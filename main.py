"""Run the intratumoral-heterogeneity pipeline on one whole-slide image.

Example:
    python main.py --demo --output-dir outputs/demo
    python main.py --slide data/biopsy.svs --output-dir outputs/biopsy --k 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.dataset import (
    NoTissueFoundError,
    OpenSlideUnavailableError,
    extract_patches,
)
from src.extract_features import extract_embeddings, save_embeddings
from src.quantify_heterogeneity import (
    NicheClusterer,
    save_cluster_heatmap,
    save_proportion_chart,
    save_report,
)
from src.synthetic import generate_synthetic_wsi

logger = logging.getLogger("ith")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tile a whole-slide image, embed the tissue patches, cluster "
            "morphological niches, and report a heterogeneity (entropy) score."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--slide", type=Path, help="Path to a .svs, .tif, or other OpenSlide WSI")
    source.add_argument(
        "--demo",
        action="store_true",
        help="Generate a synthetic H&E-like slide and run the pipeline on it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where patches, embeddings, figures, and the JSON report are written",
    )
    parser.add_argument("--patch-size", type=int, default=256, help="Patch side length in pixels at the target magnification")
    parser.add_argument("--magnification", type=float, default=20.0, help="Target objective magnification for each patch")
    parser.add_argument("--stride", type=int, default=None, help="Grid step in pixels at the target magnification. Defaults to the patch size.")
    parser.add_argument("--tissue-threshold", type=float, default=0.5, help="Minimum tissue fraction required to keep a patch")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional cap. Tiles are subsampled evenly across the slide.")
    parser.add_argument("--patch-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--model", choices=("resnet50", "dinov2_vits14"), default="resnet50")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--k", type=int, default=8, help="Number of K-Means / GMM niches")
    parser.add_argument("--method", choices=("kmeans", "gmm"), default="kmeans")
    parser.add_argument("--pca-components", type=int, default=32, help="PCA dimensions before clustering. 0 disables PCA.")
    parser.add_argument(
        "--min-cluster-fraction",
        type=float,
        default=0.02,
        help="Minimum patch share for a niche to count in the diversity ratio",
    )
    parser.add_argument(
        "--embedding-format",
        choices=("npy", "parquet", "both"),
        default="npy",
        help="How to store the N_patches x feature_dim matrix",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--demo-size",
        type=int,
        default=3072,
        help="Side length of the synthetic slide. Use a multiple of 256.",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    """Execute tiling, embedding, clustering, and figure export. Return the report dict."""

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs/demo") if args.demo else Path("outputs") / args.slide.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        slide_path = generate_synthetic_wsi(
            output_dir / "synthetic_slide.tif",
            size=args.demo_size,
            seed=args.seed,
        )
    else:
        slide_path = Path(args.slide)

    manifest = extract_patches(
        slide_path,
        output_dir,
        patch_size=args.patch_size,
        target_magnification=args.magnification,
        stride=args.stride,
        tissue_threshold=args.tissue_threshold,
        max_patches=args.max_patches,
        patch_format=args.patch_format,
    )
    embeddings, device, encoder_name = extract_embeddings(
        manifest,
        model_name=args.model,
        weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    matrix_path = save_embeddings(
        embeddings,
        manifest,
        output_dir,
        fmt=args.embedding_format,
    )

    clusterer = NicheClusterer(
        n_clusters=args.k,
        method=args.method,
        pca_components=args.pca_components,
        min_cluster_fraction=args.min_cluster_fraction,
        random_state=args.seed,
    )
    labels = clusterer.fit_predict(embeddings)
    np_labels_path = output_dir / "cluster_labels.npy"
    _save_labels(np_labels_path, labels)

    report = clusterer.score(labels, coordinates=manifest)
    geometry = json.loads((output_dir / "geometry.json").read_text())
    save_cluster_heatmap(
        coordinates=manifest,
        labels=labels,
        output_path=output_dir / "cluster_heatmap.png",
        slide_width=int(geometry["slide_width"]),
        slide_height=int(geometry["slide_height"]),
        patch_size_level0=int(geometry["patch_size_level0"]),
        thumbnail_path=output_dir / "thumbnail.png",
    )
    save_proportion_chart(
        labels,
        output_dir / "cluster_proportions.png",
        n_clusters=report.n_clusters,
    )
    payload = save_report(
        report,
        output_dir / "heterogeneity_report.json",
        extra={
            "slide_path": str(slide_path),
            "slide_id": slide_path.stem,
            "model": encoder_name,
            "weights": args.weights,
            "device": str(device),
            "target_magnification": args.magnification,
            "patch_size": args.patch_size,
            "embeddings_path": str(matrix_path),
            "labels_path": str(np_labels_path),
            "heatmap_path": str(output_dir / "cluster_heatmap.png"),
        },
    )
    # score() does not see the embedding width when k collapses to 1 and the
    # scaler is skipped. Fill it from the matrix so the JSON stays accurate.
    written = json.loads(payload.read_text())
    written["feature_dim"] = int(embeddings.shape[1])
    payload.write_text(json.dumps(written, indent=2))
    return written


def _save_labels(path: Path, labels) -> None:
    np.save(path, np.asarray(labels, dtype=np.int32))


def format_summary(report: dict) -> str:
    mixing = report["spatial_mixing"]
    mixing_text = "n/a" if mixing is None else f"{mixing:.3f}"
    counts = ", ".join(f"{key}:{value}" for key, value in report["cluster_counts"].items())
    return "\n".join(
        [
            f"Slide: {report['slide_id']}",
            f"Patches: {report['n_patches']}",
            f"Encoder: {report['model']} ({report['feature_dim']}-d) on {report['device']}",
            f"Clustering: {report['method']}, k={report['n_clusters']}, pca={report['pca_components']}",
            f"Shannon entropy: {report['shannon_entropy_bits']:.3f} bits",
            f"Heterogeneity score (normalized entropy): {report['heterogeneity_score']:.3f}",
            f"Cluster diversity ratio: {report['cluster_diversity_ratio']:.3f}",
            f"Spatial mixing: {mixing_text}",
            f"Niche counts: {counts}",
            f"Report: {report.get('_report_path', 'heterogeneity_report.json')}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, NoTissueFoundError, OpenSlideUnavailableError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    report_path = Path(report["embeddings_path"]).parent / "heterogeneity_report.json"
    report["_report_path"] = str(report_path)
    print(format_summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
