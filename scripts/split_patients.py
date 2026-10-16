#!/usr/bin/env python3
"""
Stage 1 — deterministic patient-level train/test split for glioma tiles.

Reads a tile metadata CSV and assigns *patients* (not tiles or slides) to
train (~2/3) or test (~1/3). Every tile and slide belonging to the same
patient stays in exactly one split, which prevents data leakage.

CSV schema (data/metadata.csv by default)
-----------------------------------------
Required columns:
  patient_id  — unique patient identifier (split key)
  slide_id    — whole-slide image id (kept in CSV; not used for splitting)
  tile_path   — path to a tile image (kept in CSV; not used for splitting)
  label       — supervised label (kept in CSV; not used for splitting)

Usage
-----
  python scripts/split_patients.py
  python scripts/split_patients.py --metadata data/example_metadata.csv

Outputs (created under outputs/ if missing):
  outputs/train_patients.txt  — one patient_id per line
  outputs/test_patients.txt   — one patient_id per line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Columns required in the metadata CSV.
REQUIRED_COLUMNS = ("patient_id", "slide_id", "tile_path", "label")

# Repo root is one level above scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 1.0 / 3.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split unique patient IDs into train (~2/3) and test (~1/3) "
            "with a fixed seed. Never splits tiles/slides within a patient."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Path to metadata CSV (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for train/test patient lists (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic splits (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of patients for the test set (default: {DEFAULT_TEST_SIZE})",
    )
    return parser.parse_args(argv)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load and validate the tile metadata CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}\n"
            "Expected columns: patient_id, slide_id, tile_path, label"
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 — surface parse errors clearly
        raise ValueError(f"Failed to read CSV {path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"Metadata CSV is empty: {path}")

    if df["patient_id"].isna().any():
        n_na = int(df["patient_id"].isna().sum())
        raise ValueError(f"Found {n_na} row(s) with missing patient_id in {path}")

    return df


def unique_patient_ids(df: pd.DataFrame) -> list[str]:
    """Return sorted unique patient IDs as strings (stable order before shuffle)."""
    # Sorted first so the set of IDs is deterministic before sklearn shuffles.
    patients = sorted(df["patient_id"].astype(str).unique().tolist())
    if len(patients) < 2:
        raise ValueError(
            f"Need at least 2 unique patients to split; found {len(patients)}: {patients}"
        )
    return patients


def split_patients(
    patients: list[str],
    *,
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Assign unique patient IDs to train / test.

    Uses sklearn.train_test_split on the patient list only — never on tiles
    or slides — so all material for one patient stays in one split.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")

    train_ids, test_ids = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    # Stable line order in output files (sorted), independent of shuffle order.
    return sorted(train_ids), sorted(test_ids)


def write_patient_list(path: Path, patient_ids: list[str]) -> None:
    """Write one patient_id per line (UTF-8, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(patient_ids) + ("\n" if patient_ids else "")
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df = load_metadata(args.metadata)
        patients = unique_patient_ids(df)
        train_ids, test_ids = split_patients(
            patients,
            test_size=args.test_size,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Sanity: no overlap between splits.
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        print(f"error: patient leakage across splits: {sorted(overlap)}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_patients.txt"
    test_path = output_dir / "test_patients.txt"
    write_patient_list(train_path, train_ids)
    write_patient_list(test_path, test_ids)

    n = len(patients)
    print(
        f"Split {n} patients "
        f"(train={len(train_ids)} ~{len(train_ids)/n:.0%}, "
        f"test={len(test_ids)} ~{len(test_ids)/n:.0%}; "
        f"seed={args.seed}, test_size={args.test_size})"
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
