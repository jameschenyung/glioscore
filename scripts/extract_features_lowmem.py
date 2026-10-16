#!/usr/bin/env python3
"""
Low-memory DINOv2 embedding for CPU PoC runs (15 GB hosts).

Same contract as ``extract_features.py`` (writes train/test ``.npy`` + manifest),
but:
  - pre-allocates the output array (no chunk list growth)
  - saves train embeddings to disk before running test
  - forces ``num_workers=0`` and modest batch size
  - caps torch intra-op threads

Patient-level splits are still taken from Stage 1 lists; this script never
re-splits.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse Stage-2 helpers so behavior stays aligned.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import (  # noqa: E402
    TileDataset,
    assert_no_patient_leakage,
    build_model_and_transform,
    filter_split,
    load_metadata,
    prepare_tile_paths,
    read_patient_list,
    resolve_device,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-memory Stage-2 DINOv2 embedding")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--train-patients", type=Path, required=True)
    p.add_argument("--test-patients", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tile-root", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--model", type=str, default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-threads", type=int, default=4)
    return p.parse_args(argv)


def embed_into_array(
    paths: list[Path],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    feature_dim: int,
    split_name: str,
) -> np.ndarray:
    """Fill a pre-allocated float32 array row-by-row; avoid holding chunk lists."""
    out = np.empty((len(paths), feature_dim), dtype=np.float32)
    dataset = TileDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = model.to(device)
    model.eval()
    row = 0
    with torch.no_grad():
        for images, _indices in tqdm(loader, desc=f"embed:{split_name}"):
            images = images.to(device)
            feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            arr = feats.detach().cpu().numpy().astype(np.float32, copy=False)
            n = arr.shape[0]
            out[row : row + n] = arr
            row += n
            del images, feats, arr
    if row != len(paths):
        raise RuntimeError(f"{split_name}: filled {row} rows, expected {len(paths)}")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    torch.set_num_threads(args.torch_threads)

    try:
        train_patients = read_patient_list(args.train_patients)
        test_patients = read_patient_list(args.test_patients)
        overlap = set(train_patients) & set(test_patients)
        if overlap:
            raise ValueError(f"Patient overlap across splits: {sorted(overlap)}")

        meta = load_metadata(args.metadata)
        train_df = filter_split(meta, train_patients, "train")
        test_df = filter_split(meta, test_patients, "test")
        assert_no_patient_leakage(train_df, test_df)

        train_paths = prepare_tile_paths(train_df, args.tile_root, synthetic=False)
        test_paths = prepare_tile_paths(test_df, args.tile_root, synthetic=False)

        device = resolve_device(args.device)
        model, transform, feature_dim, _cfg = build_model_and_transform(
            args.model, pretrained=True
        )

        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)

        print(f"Embedding train ({len(train_paths)} tiles) …")
        train_emb = embed_into_array(
            train_paths, model, transform, device, args.batch_size, feature_dim, "train"
        )
        train_labels = train_df["label"].to_numpy(dtype=np.int64)
        np.save(out / "train_embeddings.npy", train_emb)
        np.save(out / "train_labels.npy", train_labels)
        print(f"Wrote {out / 'train_embeddings.npy'} shape={train_emb.shape}")
        del train_emb
        gc.collect()

        print(f"Embedding test ({len(test_paths)} tiles) …")
        test_emb = embed_into_array(
            test_paths, model, transform, device, args.batch_size, feature_dim, "test"
        )
        test_labels = test_df["label"].to_numpy(dtype=np.int64)
        np.save(out / "test_embeddings.npy", test_emb)
        np.save(out / "test_labels.npy", test_labels)
        print(f"Wrote {out / 'test_embeddings.npy'} shape={test_emb.shape}")

        manifest = {
            "model": args.model,
            "feature_dim": feature_dim,
            "pretrained": True,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": 0,
            "n_train": int(len(train_paths)),
            "n_test": int(len(test_paths)),
            "lowmem": True,
            "files": {
                "train_embeddings": "train_embeddings.npy",
                "train_labels": "train_labels.npy",
                "test_embeddings": "test_embeddings.npy",
                "test_labels": "test_labels.npy",
            },
        }
        manifest_path = out / "feature_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {manifest_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
