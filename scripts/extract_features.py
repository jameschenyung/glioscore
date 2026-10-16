#!/usr/bin/env python3
"""
Stage 2 — per-tile foundation-model embeddings for glioma tiles.

Reads the Stage 1 patient split lists and the tile metadata CSV, keeps every
tile whose patient_id is in train or test (never re-splits), and embeds each
tile with a timm DINOv2 backbone. Train and test are processed separately with
the same model weights so Stage 3 XGBoost can train/evaluate without leakage.

Default model
-------------
  vit_small_patch14_dinov2.lvd142m  (timm; embedding dim 384)

Outputs (under outputs/ by default)
-----------------------------------
  train_embeddings.npy   float32 (N_train, D)
  train_labels.npy       int64   (N_train,)
  test_embeddings.npy    float32 (N_test, D)
  test_labels.npy        int64   (N_test,)
  feature_manifest.json  model name, dim, and row-aligned metadata for both splits

CSV schema (same as Stage 1): patient_id, slide_id, tile_path, label

Usage
-----
  python scripts/extract_features.py
  python scripts/extract_features.py --metadata data/example_metadata.csv \\
      --synthetic-tiles --no-pretrained   # smoke test without real WSIs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Required metadata columns (aligned with Stage 1).
REQUIRED_COLUMNS = ("patient_id", "slide_id", "tile_path", "label")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_TRAIN_PATIENTS = DEFAULT_OUTPUT_DIR / "train_patients.txt"
DEFAULT_TEST_PATIENTS = DEFAULT_OUTPUT_DIR / "test_patients.txt"
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_BATCH_SIZE = 32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-tile DINOv2 embeddings for train/test patient splits. "
            "Patients are never mixed across splits."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Tile metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--train-patients",
        type=Path,
        default=DEFAULT_TRAIN_PATIENTS,
        help=f"Stage 1 train patient list (default: {DEFAULT_TRAIN_PATIENTS})",
    )
    parser.add_argument(
        "--test-patients",
        type=Path,
        default=DEFAULT_TEST_PATIENTS,
        help=f"Stage 1 test patient list (default: {DEFAULT_TEST_PATIENTS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for embeddings / labels / manifest (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"timm model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto | cpu | cuda | mps (default: auto)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained timm weights (default: True; use --no-pretrained for smoke)",
    )
    parser.add_argument(
        "--synthetic-tiles",
        action="store_true",
        help=(
            "If a tile_path is missing, write a small synthetic RGB PNG there "
            "so the pipeline can be smoke-tested without real WSI tiles."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the neural net; write random float32 embeddings of the model's "
            "feature dim. Useful for wiring Stage 3 without downloading weights."
        ),
    )
    parser.add_argument(
        "--tile-root",
        type=Path,
        default=REPO_ROOT / "data",
        help=(
            "Root for relative tile_path values in the CSV "
            f"(default: {REPO_ROOT / 'data'})"
        ),
    )
    return parser.parse_args(argv)


def resolve_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if preference != "auto":
        raise ValueError(f"Unknown device preference: {preference}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_patient_list(path: Path) -> list[str]:
    """Load one patient_id per line; skip blanks; preserve file order uniqueness."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Patient list not found: {path}\n"
            "Run Stage 1 first: python scripts/split_patients.py"
        )
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        pid = line.strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
    if not ids:
        raise ValueError(f"Patient list is empty: {path}")
    return ids


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}\n"
            "Expected columns: patient_id, slide_id, tile_path, label"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read CSV {path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"Metadata CSV is empty: {path}")
    if df["patient_id"].isna().any():
        raise ValueError(f"Found missing patient_id values in {path}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def filter_split(
    df: pd.DataFrame,
    patient_ids: list[str],
    split_name: str,
) -> pd.DataFrame:
    """Keep only rows for patients in this split; preserve CSV row order."""
    allowed = set(patient_ids)
    subset = df[df["patient_id"].isin(allowed)].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"No metadata rows for {split_name} patients {sorted(allowed)}. "
            "Check that metadata.csv patient_ids match the Stage 1 lists."
        )
    # Guarding against accidental leakage if a caller passes overlapping lists.
    return subset


def resolve_tile_path(tile_path: str, tile_root: Path) -> Path:
    """Absolute paths stay as-is; relative paths are resolved under tile_root."""
    p = Path(tile_path)
    if p.is_absolute():
        return p
    return (tile_root / p).resolve()


def ensure_tile_exists(path: Path, *, synthetic: bool, seed: int) -> None:
    if path.is_file():
        return
    if not synthetic:
        raise FileNotFoundError(
            f"Tile image not found: {path}\n"
            "Provide real tiles at tile_path, or pass --synthetic-tiles for a smoke test."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Small but non-trivial RGB patch; the transform will resize to model input.
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def prepare_tile_paths(
    df: pd.DataFrame,
    tile_root: Path,
    *,
    synthetic: bool,
) -> list[Path]:
    paths: list[Path] = []
    for i, rel in enumerate(df["tile_path"].astype(str).tolist()):
        path = resolve_tile_path(rel, tile_root)
        # Deterministic synthetic content per path index so re-runs are stable.
        ensure_tile_exists(path, synthetic=synthetic, seed=10_000 + i)
        paths.append(path)
    return paths


class TileDataset(Dataset):
    """RGB tiles with a torchvision/timm transform; returns (tensor, row_index)."""

    def __init__(self, paths: list[Path], transform) -> None:
        if not paths:
            raise ValueError("TileDataset received an empty path list")
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, index


def build_model_and_transform(model_name: str, pretrained: bool):
    """Create a timm backbone (num_classes=0 → pooled embedding) and eval transform."""
    import timm
    from timm.data import create_transform, resolve_model_data_config

    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to create timm model {model_name!r} (pretrained={pretrained}): {exc}"
        ) from exc

    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    # Probe embedding dim without depending on undocumented attributes.
    # Prefer num_features when present (standard for timm feature extractors).
    feature_dim = int(getattr(model, "num_features", 0) or 0)
    if feature_dim <= 0:
        with torch.no_grad():
            c, h, w = data_config["input_size"]
            probe = model(torch.zeros(1, c, h, w))
            if isinstance(probe, (tuple, list)):
                probe = probe[0]
            if probe.ndim > 2:
                probe = torch.flatten(probe, 1)
            feature_dim = int(probe.shape[-1])

    return model, transform, feature_dim, data_config


def embed_split(
    paths: list[Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> np.ndarray:
    """Run the encoder over one split; row i matches paths[i] / metadata row i."""
    dataset = TileDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # critical: preserve row alignment with labels / manifest
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images, _indices in tqdm(loader, desc=f"embed:{split_name}", leave=False):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            chunks.append(feats.detach().cpu().numpy().astype(np.float32, copy=False))

    if not chunks:
        raise RuntimeError(f"No embeddings produced for split {split_name}")
    return np.concatenate(chunks, axis=0)


def dry_run_embeddings(n: int, feature_dim: int, seed: int) -> np.ndarray:
    """Random embeddings for pipeline wiring tests (not for real training)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, feature_dim), dtype=np.float32)


def assert_no_patient_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    overlap = set(train_df["patient_id"]) & set(test_df["patient_id"])
    if overlap:
        raise RuntimeError(
            f"Patient leakage between train and test splits: {sorted(overlap)}"
        )


def write_outputs(
    output_dir: Path,
    *,
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_paths: list[Path],
    test_paths: list[Path],
    model_name: str,
    feature_dim: int,
    pretrained: bool,
    dry_run: bool,
    device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_embeddings.npy", train_emb)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_embeddings.npy", test_emb)
    np.save(output_dir / "test_labels.npy", test_labels)

    def rows_for(df: pd.DataFrame, paths: list[Path], split: str) -> list[dict]:
        records = []
        for i, row in df.reset_index(drop=True).iterrows():
            records.append(
                {
                    "split": split,
                    "row": int(i),
                    "patient_id": str(row["patient_id"]),
                    "slide_id": str(row["slide_id"]),
                    "tile_path": str(row["tile_path"]),
                    "resolved_path": str(paths[i]),
                    "label": int(row["label"]),
                }
            )
        return records

    manifest = {
        "model_name": model_name,
        "embedding_dim": int(feature_dim),
        "pretrained": bool(pretrained),
        "dry_run": bool(dry_run),
        "device": device,
        "n_train": int(train_emb.shape[0]),
        "n_test": int(test_emb.shape[0]),
        "files": {
            "train_embeddings": "train_embeddings.npy",
            "train_labels": "train_labels.npy",
            "test_embeddings": "test_embeddings.npy",
            "test_labels": "test_labels.npy",
        },
        # Row i of train_* arrays matches train_rows[i]; same for test.
        "train_rows": rows_for(train_df, train_paths, "train"),
        "test_rows": rows_for(test_df, test_paths, "test"),
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        train_patients = read_patient_list(args.train_patients)
        test_patients = read_patient_list(args.test_patients)
        overlap = set(train_patients) & set(test_patients)
        if overlap:
            raise ValueError(
                f"Stage 1 lists overlap on patients {sorted(overlap)}; aborting."
            )

        meta = load_metadata(args.metadata)
        train_df = filter_split(meta, train_patients, "train")
        test_df = filter_split(meta, test_patients, "test")
        assert_no_patient_leakage(train_df, test_df)

        # Resolve / optionally synthesize tile images before touching the model.
        train_paths = prepare_tile_paths(
            train_df, args.tile_root, synthetic=args.synthetic_tiles
        )
        test_paths = prepare_tile_paths(
            test_df, args.tile_root, synthetic=args.synthetic_tiles
        )

        if args.dry_run:
            # Infer dim from an untrained architecture without downloading weights.
            _model, _transform, feature_dim, _cfg = build_model_and_transform(
                args.model, pretrained=False
            )
            del _model, _transform
            train_emb = dry_run_embeddings(len(train_paths), feature_dim, seed=42)
            test_emb = dry_run_embeddings(len(test_paths), feature_dim, seed=43)
            device = torch.device("cpu")
            pretrained_used = False
        else:
            device = resolve_device(args.device)
            model, transform, feature_dim, _cfg = build_model_and_transform(
                args.model, pretrained=args.pretrained
            )
            pretrained_used = args.pretrained
            train_emb = embed_split(
                train_paths,
                model,
                transform,
                device,
                args.batch_size,
                args.num_workers,
                "train",
            )
            test_emb = embed_split(
                test_paths,
                model,
                transform,
                device,
                args.batch_size,
                args.num_workers,
                "test",
            )

        train_labels = train_df["label"].to_numpy(dtype=np.int64)
        test_labels = test_df["label"].to_numpy(dtype=np.int64)

        if train_emb.shape[0] != len(train_labels) or test_emb.shape[0] != len(test_labels):
            raise RuntimeError(
                "Embedding/label length mismatch: "
                f"train {train_emb.shape[0]} vs {len(train_labels)}, "
                f"test {test_emb.shape[0]} vs {len(test_labels)}"
            )
        if train_emb.shape[1] != feature_dim or test_emb.shape[1] != feature_dim:
            raise RuntimeError(
                f"Unexpected embedding dim: train={train_emb.shape}, "
                f"test={test_emb.shape}, expected D={feature_dim}"
            )

        manifest_path = write_outputs(
            args.output_dir,
            train_emb=train_emb,
            train_labels=train_labels,
            test_emb=test_emb,
            test_labels=test_labels,
            train_df=train_df,
            test_df=test_df,
            train_paths=train_paths,
            test_paths=test_paths,
            model_name=args.model,
            feature_dim=feature_dim,
            pretrained=pretrained_used,
            dry_run=args.dry_run,
            device=str(device),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Model={args.model} dim={feature_dim} "
        f"train={train_emb.shape[0]} test={test_emb.shape[0]} "
        f"dry_run={args.dry_run} pretrained={pretrained_used}"
    )
    print(f"Wrote embeddings under {args.output_dir}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
