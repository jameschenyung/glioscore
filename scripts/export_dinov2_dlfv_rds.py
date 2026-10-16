#!/usr/bin/env python3
"""
Export DINOv2 tile embeddings as an R ``.rds`` matrix (VGG19 DLFV style).

Produces a numeric matrix where:
  - rows = tiles (samples)
  - cols = DINOv2 feature dimensions (unnamed, like VGG19)
  - rownames encode slide + Havoc region color + tile bbox so Alberto's R
    pipeline can parse PATIENT_ID with ``TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}``

Rowname scheme
--------------
From a tile path such as::

  tiles_selected_TCGA_LGG_GBM/TCGA-CS-4938-01Z-00-DX1_blue/(100,200,612,712).jpg

we write::

  TCGA-CS-4938-01Z-00-DX1_blue_100_200_612_712

which mirrors the VGG19 Havoc-region style ``TCGA-CS-4938-01Z-00-DX1_blue``
while remaining unique per tile.

Usage
-----
  # Reuse Stage-2 npy from the DINOv2 PoC (fast):
  python scripts/export_dinov2_dlfv_rds.py \\
      --metadata data/metadata_tcga_tiles.csv \\
      --tile-root data/tiles_tcga \\
      --reuse-embeddings-dir outputs/dinov2_poc \\
      --output outputs/dinov2_dlfv.rds

  # Embed from tiles (CPU PoC with caps):
  python scripts/export_dinov2_dlfv_rds.py \\
      --metadata data/metadata_tcga_tiles.csv \\
      --tile-root data/tiles_tcga \\
      --model vit_small_patch14_dinov2.lvd142m \\
      --max-tiles-per-patient 32 --seed 42 \\
      --output outputs/dinov2_dlfv.rds
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata_tcga_tiles.csv"
DEFAULT_TILE_ROOT = REPO_ROOT / "data" / "tiles_tcga"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "dinov2_dlfv.rds"
DEFAULT_SAMPLE_OUTPUT = REPO_ROOT / "outputs" / "dinov2_dlfv_sample.rds"
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_SEED = 42

PATIENT_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
# slide folder: TCGA-...-DX1_blue  OR path ending with /(x0, y0, x1, y1).jpg
BBOX_RE = re.compile(
    r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\.(?:jpg|jpeg|png|tif|tiff)$",
    re.IGNORECASE,
)
COLOR_SUFFIX_RE = re.compile(
    r"_(blue|green|yellow|orange|red|purple|black|white|pink|brown)$",
    re.IGNORECASE,
)

REQUIRED_COLUMNS = ("patient_id", "slide_id", "tile_path", "label")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embed glioma tiles with DINOv2 and write an R .rds feature matrix "
            "with patient-parsable rownames (VGG19 DLFV style)."
        )
    )
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--sample-output",
        type=Path,
        default=DEFAULT_SAMPLE_OUTPUT,
        help="Also write a small sample .rds (first N rows) for quick R checks",
    )
    p.add_argument("--sample-rows", type=int, default=50)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional PoC cap on unique patients (seeded subsample)",
    )
    p.add_argument(
        "--max-tiles-per-patient",
        type=int,
        default=None,
        help="Optional PoC cap on tiles kept per patient (seeded)",
    )
    p.add_argument(
        "--reuse-embeddings-dir",
        type=Path,
        default=None,
        help=(
            "If set, load train/test_embeddings.npy + patient lists from this "
            "Stage-2 output dir instead of re-running DINOv2 (must match metadata)."
        ),
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional JSON sidecar documenting rownames scheme + shape",
    )
    return p.parse_args(argv)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing columns {missing}; found {list(df.columns)}")
    if df.empty:
        raise ValueError(f"Metadata CSV is empty: {path}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def rowname_from_tile(tile_path: str, slide_id: str) -> str:
    """
    Build a VGG19-like rowname with unique tile bbox.

    Prefer slide_id (already ``TCGA-…-DX#_color``) + bbox coords from the filename.
    """
    path = Path(str(tile_path))
    bbox = BBOX_RE.search(path.name)
    slide = str(slide_id).strip() if slide_id is not None else ""
    if not slide:
        # Fall back to parent folder name (…/TCGA-…_blue/(coords).jpg)
        slide = path.parent.name

    if bbox:
        x0, y0, x1, y1 = bbox.groups()
        name = f"{slide}_{x0}_{y0}_{x1}_{y1}"
    else:
        # Last resort: slide + stem (still must contain TCGA barcode)
        stem = re.sub(r"[^\w.\-]+", "_", path.stem)
        name = f"{slide}_{stem}"

    if not PATIENT_ID_RE.search(name):
        raise ValueError(
            f"Rowname does not contain a TCGA PATIENT_ID: {name!r} "
            f"(from tile_path={tile_path!r}, slide_id={slide_id!r})"
        )
    return name


def subsample_dataframe(
    df: pd.DataFrame,
    *,
    max_patients: int | None,
    max_tiles_per_patient: int | None,
    seed: int,
) -> pd.DataFrame:
    out = df
    if max_patients is not None:
        if max_patients < 1:
            raise ValueError("--max-patients must be >= 1")
        patients = sorted(out["patient_id"].unique().tolist())
        if len(patients) > max_patients:
            rng = np.random.default_rng(seed)
            keep = set(rng.choice(patients, size=max_patients, replace=False).tolist())
            out = out[out["patient_id"].isin(keep)].reset_index(drop=True)
            print(f"Patient subsample: kept {len(keep)}/{len(patients)} (seed={seed})")

    if max_tiles_per_patient is not None:
        if max_tiles_per_patient < 1:
            raise ValueError("--max-tiles-per-patient must be >= 1")
        rng = np.random.default_rng(seed)
        parts: list[pd.DataFrame] = []
        for pid, group in out.groupby("patient_id", sort=True):
            g = group.sort_values("tile_path").reset_index(drop=True)
            if len(g) > max_tiles_per_patient:
                idx = rng.choice(len(g), size=max_tiles_per_patient, replace=False)
                g = g.iloc[sorted(idx.tolist())]
            parts.append(g)
        out = pd.concat(parts, ignore_index=True)
        print(
            f"Tile cap: max {max_tiles_per_patient}/patient → {len(out)} tiles "
            f"(seed={seed})"
        )
    return out.reset_index(drop=True)


def ensure_unique_rownames(names: list[str]) -> list[str]:
    """Deduplicate rownames with a stable numeric suffix if needed."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
            continue
        seen[name] += 1
        alt = f"{name}__{seen[name]}"
        while alt in seen:
            seen[name] += 1
            alt = f"{name}__{seen[name]}"
        seen[alt] = 0
        out.append(alt)
    return out


def load_reused_embeddings(
    emb_dir: Path,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Rebuild the Stage-2 row order: train patients then test patients,
    each filtered from metadata in CSV order (same as extract_features*).
    """
    train_list = emb_dir / "train_patients.txt"
    test_list = emb_dir / "test_patients.txt"
    train_npy = emb_dir / "train_embeddings.npy"
    test_npy = emb_dir / "test_embeddings.npy"
    for path in (train_list, test_list, train_npy, test_npy):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage-2 artifact: {path}")

    train_patients = [
        line.strip()
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    test_patients = [
        line.strip()
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_set, test_set = set(train_patients), set(test_patients)
    if train_set & test_set:
        raise RuntimeError(f"Patient leak in reused lists: {sorted(train_set & test_set)}")

    train_df = metadata[metadata["patient_id"].isin(train_set)].reset_index(drop=True)
    test_df = metadata[metadata["patient_id"].isin(test_set)].reset_index(drop=True)
    train_emb = np.load(train_npy)
    test_emb = np.load(test_npy)
    if train_emb.shape[0] != len(train_df) or test_emb.shape[0] != len(test_df):
        raise ValueError(
            "Reused embeddings do not match metadata row counts: "
            f"train {train_emb.shape[0]} vs {len(train_df)}, "
            f"test {test_emb.shape[0]} vs {len(test_df)}. "
            "Re-run embedding or pass matching --metadata."
        )
    X = np.concatenate([train_emb, test_emb], axis=0).astype(np.float32, copy=False)
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(
        f"Reused embeddings from {emb_dir}: "
        f"shape={X.shape} (train={len(train_df)}, test={len(test_df)})"
    )
    return X, df


def embed_tiles(
    df: pd.DataFrame,
    tile_root: Path,
    *,
    model_name: str,
    device: str,
    batch_size: int,
    torch_threads: int,
) -> np.ndarray:
    """Run DINOv2 via the low-mem embedder helpers."""
    import os

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_features import (  # noqa: WPS433 — local Stage-2 helpers
        build_model_and_transform,
        prepare_tile_paths,
        resolve_device,
    )
    from extract_features_lowmem import embed_into_array  # noqa: WPS433

    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    torch.set_num_threads(torch_threads)

    paths = prepare_tile_paths(df, tile_root, synthetic=False)
    dev = resolve_device(device)
    model, transform, feature_dim, _cfg = build_model_and_transform(
        model_name, pretrained=True
    )
    print(
        f"Embedding {len(paths)} tiles with {model_name} "
        f"(dim={feature_dim}, device={dev}, batch={batch_size}) …"
    )
    return embed_into_array(
        paths, model, transform, dev, batch_size, feature_dim, "all"
    )


def write_rds_via_rscript(
    X: np.ndarray,
    rownames: list[str],
    output: Path,
) -> None:
    """
    Write a numeric matrix RDS using base R ``saveRDS``.

    Intermediate: little-endian float64 binary + rownames text (no extra R packages).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix; got shape {X.shape}")
    if X.shape[0] != len(rownames):
        raise ValueError(
            f"Row count mismatch: X={X.shape[0]} vs rownames={len(rownames)}"
        )
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R (r-base-core) to write .rds files, "
            "e.g. `sudo apt-get install -y r-base-core`."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n, d = int(X.shape[0]), int(X.shape[1])

    with tempfile.TemporaryDirectory(prefix="dinov2_rds_") as tmp:
        tmp_path = Path(tmp)
        bin_path = tmp_path / "embeddings_f64le.bin"
        names_path = tmp_path / "rownames.txt"
        r_script = tmp_path / "write_rds.R"

        # row-major float64 little-endian
        np.ascontiguousarray(X, dtype="<f8").tofile(bin_path)
        names_path.write_text("\n".join(rownames) + "\n", encoding="utf-8")

        r_script.write_text(
            f"""
n <- {n}L
d <- {d}L
con <- file("{bin_path.as_posix()}", "rb")
on.exit(close(con), add = TRUE)
vals <- readBin(con, what = "double", n = n * d, size = 8L, endian = "little")
if (length(vals) != n * d) {{
  stop(sprintf("readBin length %d != expected %d", length(vals), n * d))
}}
m <- matrix(vals, nrow = n, ncol = d, byrow = TRUE)
rn <- readLines("{names_path.as_posix()}", warn = FALSE)
if (length(rn) != n) {{
  stop(sprintf("rownames length %d != nrow %d", length(rn), n))
}}
rownames(m) <- rn
# Leave colnames NULL (matches typical VGG19 DLFV matrix style).
saveRDS(m, file = "{output.as_posix()}")
cat(sprintf("Wrote RDS %s  dim=%d x %d\\n", "{output.as_posix()}", n, d))
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(r_script)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Rscript failed while writing RDS:\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        print(proc.stdout.strip())


def verify_rds(path: Path, expected_rows: int, expected_cols: int) -> list[str]:
    """Round-trip check via R: dim + first few rownames + PATIENT_ID parse."""
    r_code = f"""
m <- readRDS("{path.as_posix()}")
cat(sprintf("dim=%d,%d\\n", nrow(m), ncol(m)))
rn <- rownames(m)
cat("rownames_head\\n")
cat(paste(head(rn, 5), collapse = "\\n"), "\\n")
"""
    proc = subprocess.run(
        ["Rscript", "--vanilla", "-e", r_code],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to readRDS {path}:\n{proc.stderr}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    dim_line = next((ln for ln in lines if ln.startswith("dim=")), None)
    if dim_line is None:
        raise RuntimeError(f"Unexpected R verify output:\n{proc.stdout}")
    nr, nc = map(int, dim_line.replace("dim=", "").split(","))
    if nr != expected_rows or nc != expected_cols:
        raise RuntimeError(
            f"RDS shape mismatch: got {nr}x{nc}, expected {expected_rows}x{expected_cols}"
        )
    # Collect rownames after marker
    try:
        idx = lines.index("rownames_head")
        examples = lines[idx + 1 :]
    except ValueError:
        examples = []
    for name in examples[:5]:
        if not PATIENT_ID_RE.search(name):
            raise RuntimeError(f"Rowname missing PATIENT_ID: {name!r}")
    print(f"Verified {path}: dim={nr}x{nc}; example rownames: {examples[:3]}")
    return examples[:5]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        meta = load_metadata(args.metadata)

        if args.reuse_embeddings_dir is not None:
            X, df = load_reused_embeddings(args.reuse_embeddings_dir, meta)
            model_name = args.model
            # Prefer model name from Stage-2 manifest when present.
            manifest_path = args.reuse_embeddings_dir / "feature_manifest.json"
            if manifest_path.is_file():
                try:
                    model_name = json.loads(manifest_path.read_text()).get(
                        "model", model_name
                    )
                except json.JSONDecodeError:
                    pass
        else:
            df = subsample_dataframe(
                meta,
                max_patients=args.max_patients,
                max_tiles_per_patient=args.max_tiles_per_patient,
                seed=args.seed,
            )
            X = embed_tiles(
                df,
                args.tile_root,
                model_name=args.model,
                device=args.device,
                batch_size=args.batch_size,
                torch_threads=args.torch_threads,
            )
            model_name = args.model

        rownames = ensure_unique_rownames(
            [
                rowname_from_tile(tp, sid)
                for tp, sid in zip(
                    df["tile_path"].astype(str).tolist(),
                    df["slide_id"].astype(str).tolist(),
                )
            ]
        )
        # Sanity: every row parses to a patient id.
        bad = [n for n in rownames if not PATIENT_ID_RE.search(n)]
        if bad:
            raise RuntimeError(f"{len(bad)} rownames lack PATIENT_ID; e.g. {bad[:3]}")

        write_rds_via_rscript(X, rownames, args.output)

        sample_n = min(int(args.sample_rows), X.shape[0])
        if args.sample_output is not None and sample_n > 0:
            write_rds_via_rscript(
                X[:sample_n], rownames[:sample_n], args.sample_output
            )

        examples = verify_rds(args.output, X.shape[0], X.shape[1])
        if args.sample_output is not None and args.sample_output.is_file():
            verify_rds(args.sample_output, sample_n, X.shape[1])

        n_patients = int(df["patient_id"].nunique())
        sidecar = {
            "rds_path": str(args.output),
            "sample_rds_path": str(args.sample_output) if args.sample_output else None,
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "n_patients": n_patients,
            "model": model_name,
            "rowname_scheme": (
                "{slide_id}_{x0}_{y0}_{x1}_{y1} where slide_id is "
                "TCGA-…-DX#_color (Havoc region), bbox from tile filename; "
                "PATIENT_ID = first TCGA-XX-XXXX match"
            ),
            "patient_id_regex": PATIENT_ID_RE.pattern,
            "example_rownames": examples,
            "reuse_embeddings_dir": (
                str(args.reuse_embeddings_dir) if args.reuse_embeddings_dir else None
            ),
            "metadata": str(args.metadata),
            "tile_root": str(args.tile_root),
            "seed": args.seed,
            "max_patients": args.max_patients,
            "max_tiles_per_patient": args.max_tiles_per_patient,
        }
        manifest_out = args.manifest_out or (
            args.output.parent / "dinov2_dlfv_manifest.json"
        )
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote manifest → {manifest_out}")
        print(
            f"Done: {args.output}  shape={X.shape[0]} x {X.shape[1]}  "
            f"patients={n_patients}"
        )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
