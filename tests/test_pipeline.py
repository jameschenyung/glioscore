"""End-to-end run on the synthetic slide.

The encoder is randomly initialized so the test does not download ImageNet
weights. It checks that tiling, embedding, clustering, and the report files
line up. It does not check that niches match the painted regions — that
depends on the pretrained ResNet50 checkpoint used by ``python main.py --demo``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from main import main


def test_demo_pipeline_writes_a_bounded_score(tmp_path: Path):
    output = tmp_path / "run"
    exit_code = main(
        [
            "--demo",
            "--demo-size",
            "1024",
            "--output-dir",
            str(output),
            "--weights",
            "none",
            "--k",
            "2",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--embedding-format",
            "both",
        ]
    )
    assert exit_code == 0

    report_path = output / "heterogeneity_report.json"
    assert report_path.is_file()
    assert (output / "embeddings.npy").is_file()
    assert (output / "embeddings.parquet").is_file()
    assert (output / "cluster_heatmap.png").is_file()
    assert (output / "cluster_proportions.png").is_file()
    assert (output / "tissue_mask.png").is_file()

    embeddings = np.load(output / "embeddings.npy")
    labels = np.load(output / "cluster_labels.npy")
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == labels.shape[0]
    assert embeddings.shape[0] >= 4
    assert embeddings.shape[1] == 2048

    # The glass margin must not be saved as tissue. A corner patch would start
    # at (0, 0); the inner field of a 1024 slide starts at 256.
    manifest_x = np.loadtxt(output / "manifest.csv", delimiter=",", skiprows=1, usecols=1)
    assert np.min(manifest_x) >= 256

    sample = Image.open(next((output / "patches").glob("*.png")))
    assert sample.size == (256, 256)

    import json

    report = json.loads(report_path.read_text())
    assert 0.0 <= report["heterogeneity_score"] <= 1.0
    assert 0.0 <= report["cluster_diversity_ratio"] <= 1.0
    assert report["shannon_entropy_bits"] >= 0.0
    assert report["n_patches"] == embeddings.shape[0]
