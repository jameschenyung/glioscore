"""Metric checks that do not need a slide or a neural network."""

import numpy as np
import pytest

from src.quantify_heterogeneity import (
    NicheClusterer,
    cluster_diversity_ratio,
    normalized_shannon_entropy,
    shannon_entropy,
    spatial_mixing,
)


def test_uniform_labels_have_maximum_entropy():
    labels = np.tile(np.arange(8), 4)
    assert shannon_entropy(labels, n_clusters=8) == pytest.approx(3.0)
    assert normalized_shannon_entropy(labels, n_clusters=8) == pytest.approx(1.0)
    assert cluster_diversity_ratio(labels, n_clusters=8) == pytest.approx(1.0)


def test_single_niche_has_zero_entropy():
    labels = np.zeros(40, dtype=int)
    assert shannon_entropy(labels, n_clusters=8) == pytest.approx(0.0)
    assert normalized_shannon_entropy(labels, n_clusters=8) == pytest.approx(0.0)
    # One occupied niche out of the eight that were requested.
    assert cluster_diversity_ratio(labels, n_clusters=8, min_fraction=0.02) == pytest.approx(1 / 8)


def test_rare_niche_does_not_inflate_diversity_ratio():
    labels = np.array([0] * 98 + [1, 2])
    # Niche 1 and 2 are 1% each, under the 2% cutoff, so only niche 0 counts.
    ratio = cluster_diversity_ratio(labels, n_clusters=8, min_fraction=0.02)
    assert ratio == pytest.approx(1 / 8)
    # Entropy still notices the rare labels, and it stays below the maximum.
    assert 0.0 < shannon_entropy(labels, n_clusters=8) < 1.0


def test_spatial_mixing_of_two_vertical_bands():
    # Two columns, two rows. Left column is niche 0, right column is niche 1.
    x = np.array([0, 10, 0, 10])
    y = np.array([0, 0, 10, 10])
    labels = np.array([0, 1, 0, 1])
    # Right-edges disagree, down-edges agree -> half of the neighbor pairs differ.
    assert spatial_mixing(x, y, labels) == pytest.approx(0.5)


def test_kmeans_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    first = rng.normal(loc=0.0, scale=0.1, size=(40, 16))
    second = rng.normal(loc=5.0, scale=0.1, size=(40, 16))
    embeddings = np.vstack([first, second])
    labels = NicheClusterer(n_clusters=2, pca_components=4, random_state=0).fit_predict(embeddings)
    # Each blob should be pure. Which id is "0" depends on K-Means initialization.
    assert len(set(labels[:40].tolist())) == 1
    assert len(set(labels[40:].tolist())) == 1
    assert labels[0] != labels[-1]
    score = normalized_shannon_entropy(labels, n_clusters=2)
    assert score == pytest.approx(1.0)
