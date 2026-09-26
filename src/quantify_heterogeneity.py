"""Cluster patch embeddings and score how mixed the niches are.

K-Means (or a Gaussian mixture) groups patches that look alike into
morphological niches. The niche ids are not cell types and they are not
ordered: cluster 3 is not "more heterogeneous" than cluster 2. What carries
the heterogeneity signal is the *distribution* of those labels.

Primary score
-------------
Shannon entropy of the label histogram, divided by ``log2(k)`` so the result
lies in ``[0, 1]``:

    H = -sum_i p_i log2(p_i)
    heterogeneity_score = H / log2(k)

0 means every patch fell into one niche. 1 means the patches are spread
evenly across all k niches. Raw entropy in bits is reported as well, because
that is the quantity in the definition above.

This score is global. It does not know whether two niches sit in separate
halves of the slide or are stirred together. ``spatial_mixing`` and the
heatmap are there for that second question.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# tab10 is readable for the default k=8 and does not imply a numeric order.
NICHE_COLORS = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


@dataclass
class HeterogeneityReport:
    """Numeric summary written to ``heterogeneity_report.json``."""

    n_patches: int
    feature_dim: int
    n_clusters: int
    method: str
    pca_components: int
    shannon_entropy_bits: float
    heterogeneity_score: float
    cluster_diversity_ratio: float
    spatial_mixing: float | None
    min_cluster_fraction: float
    cluster_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "n_patches": self.n_patches,
            "feature_dim": self.feature_dim,
            "n_clusters": self.n_clusters,
            "method": self.method,
            "pca_components": self.pca_components,
            "shannon_entropy_bits": self.shannon_entropy_bits,
            "heterogeneity_score": self.heterogeneity_score,
            "cluster_diversity_ratio": self.cluster_diversity_ratio,
            "spatial_mixing": self.spatial_mixing,
            "min_cluster_fraction": self.min_cluster_fraction,
            "cluster_counts": self.cluster_counts,
        }


class NicheClusterer:
    """Unsupervised grouping of patch embeddings into tissue niches.

    High-dimensional ResNet vectors are standardized and reduced with PCA
    before clustering. A 2048-d space has many low-variance directions;
    clustering on those directions is unstable, and a full-covariance mixture
    model cannot be fit there with only a few hundred patches. PCA (default
    32 components, or fewer when there are not enough patches) is the usual
    fix. K-Means remains the default. Gaussian mixtures are available with
    diagonal covariances in the PCA space.

    Args:
        n_clusters: Requested niche count. Lowered automatically when a slide
            has fewer patches than ``k``.
        method: ``kmeans`` or ``gmm``.
        pca_components: Upper bound on PCA dimensions. ``0`` disables PCA.
        min_cluster_fraction: A niche counts toward the diversity ratio only
            when it holds at least this share of patches. This stops a handful
            of outlier tiles from inflating diversity.
        random_state: Seed for PCA, K-Means, and the mixture model.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        method: str = "kmeans",
        pca_components: int = 32,
        min_cluster_fraction: float = 0.02,
        random_state: int = 0,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if method not in {"kmeans", "gmm"}:
            raise ValueError("method must be 'kmeans' or 'gmm'")
        if pca_components < 0:
            raise ValueError("pca_components must be >= 0")
        if not 0.0 <= min_cluster_fraction <= 1.0:
            raise ValueError("min_cluster_fraction must be in [0, 1]")

        self.n_clusters = n_clusters
        self.method = method
        self.pca_components = pca_components
        self.min_cluster_fraction = min_cluster_fraction
        self.random_state = random_state
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.model = None
        self.resolved_k = n_clusters
        self.resolved_pca_components = 0

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign one integer niche id to each row of ``embeddings``."""

        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("embeddings must have shape (N_patches, feature_dim)")
        n_samples, n_features = matrix.shape
        if n_samples == 0:
            raise ValueError("embeddings is empty")

        self.resolved_k = min(self.n_clusters, n_samples)
        if self.resolved_k < self.n_clusters:
            logger.warning(
                "Requested k=%s but only %s patches are available. Using k=%s.",
                self.n_clusters,
                n_samples,
                self.resolved_k,
            )
        if self.resolved_k == 1:
            self.scaler = None
            self.pca = None
            self.model = None
            self.resolved_pca_components = 0
            return np.zeros(n_samples, dtype=np.int32)

        self.scaler = StandardScaler()
        reduced = self.scaler.fit_transform(matrix)
        self.resolved_pca_components = _pca_rank(
            self.pca_components, n_samples, n_features
        )
        if self.resolved_pca_components >= 1:
            self.pca = PCA(
                n_components=self.resolved_pca_components,
                random_state=self.random_state,
            )
            reduced = self.pca.fit_transform(reduced)
        else:
            self.pca = None

        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.resolved_k,
                n_init=10,
                random_state=self.random_state,
            )
        else:
            # Diagonal covariances stay well-defined when a niche has only a
            # few dozen patches in PCA space.
            self.model = GaussianMixture(
                n_components=self.resolved_k,
                covariance_type="diag",
                random_state=self.random_state,
            )
        labels = self.model.fit_predict(reduced)
        return np.asarray(labels, dtype=np.int32)

    def score(self, labels: np.ndarray, coordinates: pd.DataFrame | None = None) -> HeterogeneityReport:
        """Entropy, diversity ratio, and optional neighbor mixing for one slide."""

        labels = np.asarray(labels, dtype=np.int32)
        k = int(self.resolved_k)
        counts = np.bincount(labels, minlength=k)
        mixing = None
        if coordinates is not None:
            mixing = spatial_mixing(
                coordinates["x"].to_numpy(),
                coordinates["y"].to_numpy(),
                labels,
            )
        feature_dim = 0
        if self.scaler is not None:
            feature_dim = int(len(self.scaler.mean_))
        return HeterogeneityReport(
            n_patches=int(labels.size),
            feature_dim=feature_dim,
            n_clusters=k,
            method=self.method,
            pca_components=int(self.resolved_pca_components),
            shannon_entropy_bits=shannon_entropy(labels, n_clusters=k),
            heterogeneity_score=normalized_shannon_entropy(labels, n_clusters=k),
            cluster_diversity_ratio=cluster_diversity_ratio(
                labels,
                n_clusters=k,
                min_fraction=self.min_cluster_fraction,
            ),
            spatial_mixing=mixing,
            min_cluster_fraction=self.min_cluster_fraction,
            cluster_counts={str(i): int(counts[i]) for i in range(k)},
        )


def shannon_entropy(labels: np.ndarray, n_clusters: int | None = None) -> float:
    """Shannon entropy ``H = -sum p_i log2(p_i)`` of a label histogram, in bits.

    Empty bins do not change the value (the convention ``0 log 0 = 0``).
    ``n_clusters`` only matters as the length of the histogram; extra empty
    bins beyond the largest label are ignored by the sum, which is correct.
    """

    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1
    counts = np.bincount(labels.astype(int), minlength=n_clusters).astype(np.float64)
    if counts.sum() == 0:
        return 0.0
    # scipy.stats.entropy defaults to natural log. base=2 reports bits.
    return float(entropy(counts, base=2))


def normalized_shannon_entropy(labels: np.ndarray, n_clusters: int) -> float:
    """Entropy divided by ``log2(k)``, so scores from different ``k`` share a 0-1 scale.

    This is the heterogeneity score the command-line runner prints. With
    ``k == 1`` the denominator is zero and the score is defined as 0.
    """

    if n_clusters <= 1:
        return 0.0
    raw = shannon_entropy(labels, n_clusters=n_clusters)
    return float(raw / np.log2(n_clusters))


def cluster_diversity_ratio(
    labels: np.ndarray,
    n_clusters: int,
    min_fraction: float = 0.02,
) -> float:
    """Share of the ``k`` niches that actually appear on the slide.

    A niche is counted when its patch fraction is at least ``min_fraction``.
    The ratio is ``n_present / k`` and therefore sits in ``[0, 1]``. A slide
    that uses 2 niches out of 8 scores 0.25 even if those two niches are
    perfectly balanced (that balanced case still has high entropy). Read the
    ratio together with the entropy, not instead of it.
    """

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels.astype(int), minlength=n_clusters).astype(np.float64)
    fractions = counts / counts.sum()
    present = int(np.count_nonzero(fractions >= min_fraction))
    return float(present / n_clusters)


def spatial_mixing(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> float | None:
    """Fraction of grid-neighbor pairs whose niche labels differ.

    Neighbors are the next occupied patch one grid step to the right or
    below. The grid step is the smallest positive gap in ``x`` and in ``y``.
    The value is in ``[0, 1]``. Scores near 0 mean each niche occupies a
    solid region. Scores near 1 mean neighboring patches often disagree,
    i.e. the niches are spatially intermixed. Returns None when no two
    patches touch on that grid.
    """

    if len(labels) == 0:
        return None
    xs = np.asarray(x, dtype=np.int64)
    ys = np.asarray(y, dtype=np.int64)
    labs = np.asarray(labels, dtype=np.int32)
    points = {(int(px), int(py)): int(lab) for px, py, lab in zip(xs, ys, labs)}
    dx = _min_positive_gap(xs)
    dy = _min_positive_gap(ys)
    steps: list[tuple[int, int]] = []
    if dx > 0:
        steps.append((dx, 0))
    if dy > 0:
        steps.append((0, dy))
    if not steps:
        return None

    compared = 0
    disagreed = 0
    for (px, py), lab in points.items():
        for step_x, step_y in steps:
            neighbor = points.get((px + step_x, py + step_y))
            if neighbor is None:
                continue
            compared += 1
            if neighbor != lab:
                disagreed += 1
    if compared == 0:
        return None
    return float(disagreed / compared)


def save_cluster_heatmap(
    coordinates: pd.DataFrame,
    labels: np.ndarray,
    output_path: str | Path,
    slide_width: int,
    slide_height: int,
    patch_size_level0: int,
    thumbnail_path: str | Path | None = None,
) -> Path:
    """Draw niche ids back onto slide coordinates, next to the tissue thumbnail.

    Cluster numbers are categories. The colormap is discrete on purpose so a
    reader does not treat the ids as a severity scale.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int32)
    canvas, scale = _rasterize_labels(
        x=coordinates["x"].to_numpy(),
        y=coordinates["y"].to_numpy(),
        labels=labels,
        slide_width=slide_width,
        slide_height=slide_height,
        patch_size_level0=patch_size_level0,
    )
    k = int(labels.max()) + 1 if labels.size else 0
    colors = [NICHE_COLORS[i % len(NICHE_COLORS)] for i in range(max(k, 1))]
    cmap = ListedColormap(colors)
    # Background stays masked so glass is not painted as niche 0.
    masked = np.ma.masked_where(canvas < 0, canvas)
    bounds = np.arange(-0.5, max(k, 1) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    thumbnail = None
    if thumbnail_path is not None and Path(thumbnail_path).is_file():
        thumbnail = plt.imread(thumbnail_path)

    ncols = 2 if thumbnail is not None else 1
    fig_width = 11 if thumbnail is not None else 6
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 5.5), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    if thumbnail is not None:
        axes[0].imshow(thumbnail)
        axes[0].set_title("Tissue thumbnail")
        axes[0].axis("off")
        heat_ax = axes[1]
    else:
        heat_ax = axes[0]

    heat_ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(0, slide_width, slide_height, 0),
        aspect="equal",
    )
    heat_ax.set_xlim(0, slide_width)
    heat_ax.set_ylim(slide_height, 0)
    heat_ax.set_title("Niche map (level-0 coordinates)")
    heat_ax.set_xlabel("x (level-0 pixels)")
    heat_ax.set_ylabel("y (level-0 pixels)")
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=f"niche {i}")
        for i in range(k)
    ]
    heat_ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Intratumoral niches — colors are labels, not a ranked scale", fontsize=12)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote niche heatmap (%s px/level-0-px scale %.5f) to %s", canvas.shape, scale, output_path)
    return output_path


def save_proportion_chart(
    labels: np.ndarray,
    output_path: str | Path,
    n_clusters: int,
) -> Path:
    """Bar chart of how many patches landed in each niche."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_clusters)
    frame = pd.DataFrame(
        {
            "niche": [str(i) for i in range(n_clusters)],
            "patches": counts.astype(int),
            "fraction": counts / max(int(counts.sum()), 1),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    sns.barplot(
        data=frame,
        x="niche",
        y="fraction",
        hue="niche",
        palette=[NICHE_COLORS[i % len(NICHE_COLORS)] for i in range(n_clusters)],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Niche")
    ax.set_ylabel("Fraction of patches")
    ax.set_title("Cluster proportions used for the entropy score")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_report(report: HeterogeneityReport, output_path: str | Path, extra: dict | None = None) -> Path:
    """Write the heterogeneity summary as JSON."""

    output_path = Path(output_path)
    payload = report.to_dict()
    if extra:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def _pca_rank(requested: int, n_samples: int, n_features: int) -> int:
    """Largest PCA rank that sklearn will actually fit on this matrix."""

    if requested <= 0:
        return 0
    # PCA needs strictly fewer components than the sample count.
    return max(0, min(requested, n_features, n_samples - 1))


def _min_positive_gap(values: np.ndarray) -> int:
    unique = np.unique(values.astype(np.int64))
    if unique.size < 2:
        return 0
    gaps = np.diff(unique)
    positive = gaps[gaps > 0]
    if positive.size == 0:
        return 0
    return int(positive.min())


def _rasterize_labels(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    slide_width: int,
    slide_height: int,
    patch_size_level0: int,
    canvas_max: int = 900,
) -> tuple[np.ndarray, float]:
    """Paint patches onto a small canvas. Empty pixels are -1."""

    scale = canvas_max / float(max(slide_width, slide_height, 1))
    canvas_w = max(1, int(round(slide_width * scale)))
    canvas_h = max(1, int(round(slide_height * scale)))
    canvas = np.full((canvas_h, canvas_w), -1, dtype=np.int16)
    paint_w = max(1, int(round(patch_size_level0 * scale)))
    paint_h = paint_w
    for px, py, lab in zip(x, y, labels):
        x0 = int(np.clip(round(float(px) * scale), 0, canvas_w - 1))
        y0 = int(np.clip(round(float(py) * scale), 0, canvas_h - 1))
        x1 = min(canvas_w, x0 + paint_w)
        y1 = min(canvas_h, y0 + paint_h)
        canvas[y0:y1, x0:x1] = int(lab)
    return canvas, scale
