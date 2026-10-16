#!/usr/bin/env python3
"""
VGG19-RDS baseline — XGBoost on Havoc-region VGG19 features from an R .rds file.

Loads the manager-provided VGG19-by-region ``.rds``. Default search order when
``--rds`` is omitted: ``data/lgggbm_cnn_byRegion_VGG19.rds`` (cloud/Linux), then
the canonical user path
``C:\\Users\\james\\Downloads\\lgggbm_cnn_byRegion_VGG19.rds`` if that file
exists. No copy into ``data/`` is required. Rows are Havoc regions (e.g. blue /
green / yellow) across slides and patients; columns are VGG19 feature
dimensions. Splits and evaluates at the **patient** level only.

PATIENT_ID rule
---------------
From each row name (or a patient-id column if present), extract the TCGA
patient barcode prefix with the regex::

    TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}

That is the first three hyphenated fields (e.g. ``TCGA-FG-A87N`` from
``TCGA-FG-A87N-01Z-00-DX1_blue`` or ``LGG_TCGA-FG-A87N_yellow``). All region
rows whose names share the same prefix stay in the same split.

Label rule
----------
Prefer an explicit label column among common names (``label``, ``class``,
``y``, ``diagnosis``, ``group``, ``subtype``, ``histology``, …). Otherwise
look for ``LGG`` / ``GBM`` tokens in the row names (case-insensitive). If
neither source yields labels, the script exits with a clear error.

Outputs
-------
  outputs/xgboost_vgg19_rds_model.json
  outputs/xgboost_vgg19_rds_metrics.json
  outputs/train_patients_vgg19.txt
  outputs/test_patients_vgg19.txt

Usage
-----
  python scripts/train_vgg19_rds_baseline.py
  python scripts/train_vgg19_rds_baseline.py --rds data/lgggbm_cnn_byRegion_VGG19.rds
  python scripts/train_vgg19_rds_baseline.py --rds "C:\\Users\\james\\Downloads\\lgggbm_cnn_byRegion_VGG19.rds"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
# Cloud / Linux default (optional local copy — not required).
DEFAULT_RDS = REPO_ROOT / "data" / "lgggbm_cnn_byRegion_VGG19.rds"
# Canonical location on the user's Windows machine (preferred; leave in place).
CANONICAL_USER_RDS = Path(r"C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 1.0 / 3.0

# Standard TCGA patient barcode: tissue-source-site + participant
# (e.g. TCGA-FG-A87N). Longer barcodes append sample/vial/etc. after this.
PATIENT_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)

# Column names treated as labels when present (case-insensitive match).
LABEL_COLUMN_CANDIDATES = (
    "label",
    "labels",
    "class",
    "classes",
    "y",
    "diagnosis",
    "group",
    "subtype",
    "histology",
    "histology_label",
    "outcome",
    "response",
    "tumor_type",
    "disease",
)

# Non-feature metadata columns that may appear alongside VGG19 dims.
METADATA_COLUMN_CANDIDATES = (
    "patient_id",
    "patient",
    "barcode",
    "sample_id",
    "slide_id",
    "slide",
    "region",
    "color",
    "havoc_region",
    "rowname",
    "rownames",
    *LABEL_COLUMN_CANDIDATES,
)

# Underscore-separated tokens (e.g. LGG_TCGA-...) are common; do not use \b
# because "_" is a word character in Python regex.
LGG_GBM_RE = re.compile(r"(?<![A-Za-z0-9])(LGG|GBM)(?![A-Za-z0-9])", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost on VGG19 Havoc-region features from an .rds file "
            "with a strict patient-level train/test split (seed=42)."
        )
    )
    parser.add_argument(
        "--rds",
        type=Path,
        default=None,
        help=(
            "Path to VGG19-by-region .rds. If omitted, tries "
            f"{DEFAULT_RDS} then the canonical user path "
            f"{CANONICAL_USER_RDS} when that file exists. "
            "Override freely; no copy into data/ is required."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for model, metrics, and patient lists (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for patient split and XGBoost (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Fraction of *patients* held out for test (default: {DEFAULT_TEST_SIZE})",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel threads for XGBoost (-1 = all; default: -1)",
    )
    return parser.parse_args(argv)


def resolve_rds_path(cli_path: Path | None) -> Path:
    """
    Resolve the RDS path.

    Explicit ``--rds`` always wins. Otherwise prefer ``data/...`` when present
    (cloud/Linux), then the canonical Windows Downloads path if that file
    exists. If neither is found, return the data/ default so the missing-file
    error still names a concrete path.
    """
    if cli_path is not None:
        return cli_path.expanduser()
    if DEFAULT_RDS.is_file():
        return DEFAULT_RDS
    if CANONICAL_USER_RDS.is_file():
        return CANONICAL_USER_RDS
    return DEFAULT_RDS


def require_rds(path: Path) -> None:
    if path.is_file():
        return
    raise FileNotFoundError(
        f"VGG19 RDS file not found: {path}\n"
        "Pass --rds to the file on disk, or leave it at the canonical path:\n"
        f"  {CANONICAL_USER_RDS}\n"
        "Optional cloud/Linux location (no copy required):\n"
        f"  {DEFAULT_RDS}"
    )


def load_rds_frame(path: Path) -> pd.DataFrame:
    """Load the first tabular object from an .rds via pyreadr."""
    require_rds(path)
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required to load .rds files. "
            "Install with: pip install pyreadr"
        ) from exc

    try:
        result = pyreadr.read_r(str(path))
    except Exception as exc:  # noqa: BLE001 — surface R-read errors clearly
        raise ValueError(f"Failed to read RDS {path}: {exc}") from exc

    if not result:
        raise ValueError(f"RDS file is empty (no objects): {path}")

    # Prefer a clearly named matrix; otherwise take the first object.
    preferred_keys = (
        "features",
        "X",
        "x",
        "data",
        "mat",
        "matrix",
        "cnn",
        "vgg19",
        "lgggbm_cnn_byRegion_VGG19",
    )
    frame = None
    chosen_key = None
    for key in preferred_keys:
        if key in result and isinstance(result[key], pd.DataFrame):
            frame = result[key]
            chosen_key = key
            break
    if frame is None:
        chosen_key = next(iter(result.keys()))
        obj = result[chosen_key]
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(
                f"RDS object '{chosen_key}' is not a data frame/matrix "
                f"(type={type(obj)!r}). Need a tabular feature matrix."
            )
        frame = obj

    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(
            f"RDS object '{chosen_key}' has empty shape {frame.shape}; "
            "expected rows=regions and columns=VGG19 features."
        )

    key_label = "unnamed" if chosen_key is None else str(chosen_key)
    print(
        f"Loaded RDS object '{key_label}' "
        f"shape={frame.shape[0]} rows x {frame.shape[1]} cols "
        f"from {path}"
    )
    return frame.copy()


def extract_patient_id(text: str) -> str | None:
    """Return TCGA-XX-XXXX patient barcode prefix, or None if not found."""
    match = PATIENT_ID_RE.search(str(text))
    if not match:
        return None
    return match.group(1).upper()


def _index_looks_like_default_r_rownames(index: pd.Index) -> bool:
    """True for RangeIndex or 1..N style labels that pyreadr uses when R rownames are absent."""
    if isinstance(index, pd.RangeIndex):
        return True
    values = [str(v) for v in index.tolist()]
    if not values:
        return True
    if all(v.isdigit() for v in values):
        return True
    # No TCGA tokens anywhere in the index → not usable as barcode source.
    if not any(PATIENT_ID_RE.search(v) for v in values):
        return True
    return False


def resolve_row_labels(frame: pd.DataFrame) -> pd.Series:
    """
    Build a string Series of per-row identifiers used for PATIENT_ID parsing.

    Prefer an explicit patient/barcode column; otherwise use the DataFrame index
    when it carries real R rownames (TCGA barcodes). Fall back to a
    ``rowname`` / ``id`` column if the index is the default 1..N sequence.
    """
    col_map = {str(c).lower(): c for c in frame.columns}
    for name in ("patient_id", "patient", "barcode", "sample_id"):
        if name in col_map:
            return frame[col_map[name]].astype(str)

    if not _index_looks_like_default_r_rownames(frame.index):
        return pd.Series(frame.index.astype(str), index=frame.index, name="rowname")

    for name in ("rowname", "rownames", "id", "ids"):
        if name in col_map:
            return frame[col_map[name]].astype(str)

    raise ValueError(
        "No usable row names or patient_id column found in the RDS table. "
        "Need R rownames (or a patient_id / barcode / rowname column) containing "
        "TCGA barcodes such as TCGA-FG-A87N."
    )


def assign_patient_ids(raw: pd.Series) -> pd.Series:
    patients = raw.map(extract_patient_id)
    missing = int(patients.isna().sum())
    if missing:
        examples = raw[patients.isna()].head(5).tolist()
        raise ValueError(
            f"Could not parse PATIENT_ID (TCGA-XX-XXXX) from {missing} row(s). "
            f"Rule: match regex {PATIENT_ID_RE.pattern!r} in row names / "
            f"patient columns. Examples that failed: {examples}"
        )
    return patients.astype(str)


def find_label_column(frame: pd.DataFrame) -> str | None:
    col_map = {str(c).lower(): c for c in frame.columns}
    for name in LABEL_COLUMN_CANDIDATES:
        if name in col_map:
            return col_map[name]
    # Also accept a column whose values are exclusively LGG/GBM-like tokens.
    for col in frame.columns:
        series = frame[col]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            vals = {
                str(v).strip().upper()
                for v in series.dropna().unique()
                if str(v).strip()
            }
            if vals and vals <= {"LGG", "GBM", "0", "1"}:
                return col
    return None


def labels_from_rownames(raw_ids: pd.Series) -> pd.Series | None:
    found: list[str | None] = []
    for text in raw_ids.astype(str):
        match = LGG_GBM_RE.search(text)
        found.append(match.group(1).upper() if match else None)
    series = pd.Series(found, index=raw_ids.index, name="label")
    if series.isna().any():
        return None
    return series


def infer_labels(frame: pd.DataFrame, raw_ids: pd.Series) -> tuple[pd.Series, str]:
    """
    Return (label Series, documentation string describing the source).

    Raises ValueError with a clear message if labels cannot be inferred.
    """
    label_col = find_label_column(frame)
    if label_col is not None:
        labels = frame[label_col].astype(str).str.strip()
        if labels.isna().any() or (labels == "").any() or (labels == "nan").any():
            raise ValueError(
                f"Label column '{label_col}' has missing/empty values; "
                "cannot train without complete labels."
            )
        doc = (
            f"Labels taken from column '{label_col}'. "
            f"Unique values: {sorted(labels.unique().tolist())}."
        )
        return labels, doc

    from_names = labels_from_rownames(raw_ids)
    if from_names is not None:
        doc = (
            "Labels inferred from LGG/GBM tokens in row names "
            f"(regex {LGG_GBM_RE.pattern!r}). "
            f"Unique values: {sorted(from_names.unique().tolist())}."
        )
        return from_names, doc

    raise ValueError(
        "Could not infer labels for the VGG19 RDS matrix.\n"
        "Need one of:\n"
        "  1) A label column (e.g. label / class / diagnosis / group) with "
        "LGG vs GBM (or other class names), or\n"
        "  2) Row names that contain LGG or GBM tokens "
        "(e.g. 'LGG_TCGA-FG-A87N_blue').\n"
        f"Columns found: {list(frame.columns)[:30]}"
        + (" ..." if len(frame.columns) > 30 else "")
    )


def feature_matrix(frame: pd.DataFrame, label_col: str | None) -> np.ndarray:
    """Select numeric feature columns; drop metadata / label columns."""
    drop = set()
    col_map = {str(c).lower(): c for c in frame.columns}
    for name in METADATA_COLUMN_CANDIDATES:
        if name in col_map:
            drop.add(col_map[name])
    if label_col is not None:
        drop.add(label_col)

    feature_cols = [c for c in frame.columns if c not in drop]
    if not feature_cols:
        raise ValueError("No feature columns left after dropping metadata/label columns.")

    # Keep only numeric columns for XGBoost.
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely non-numeric after coercion.
    keep = [c for c in numeric.columns if numeric[c].notna().any()]
    numeric = numeric[keep]
    if numeric.shape[1] == 0:
        raise ValueError(
            "No numeric VGG19 feature columns found. "
            f"Non-dropped columns were: {feature_cols[:20]}"
        )
    if numeric.isna().any().any():
        n_nan = int(numeric.isna().sum().sum())
        raise ValueError(
            f"Feature matrix has {n_nan} NaN value(s) after coercion; "
            "fix or drop incomplete rows before training."
        )
    return numeric.to_numpy(dtype=np.float32)


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Map string labels to 0..C-1. Prefer LGG=0, GBM=1 when both present."""
    unique = sorted({str(v) for v in labels.tolist()})
    upper = {u: u.upper() for u in unique}
    if set(upper.values()) == {"LGG", "GBM"}:
        class_names = ["LGG", "GBM"]
        name_to_idx = {"LGG": 0, "GBM": 1}
        y = np.array([name_to_idx[str(v).upper()] for v in labels], dtype=np.int64)
        return y, name_to_idx, class_names

    class_names = unique
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([name_to_idx[str(v)] for v in labels], dtype=np.int64)
    return y, name_to_idx, class_names


def patient_level_split(
    patient_ids: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Split unique patient IDs (~test_size test, rest train), then expand to rows.

    Stratify on the per-patient majority label when every class has ≥2 patients.
    """
    unique_patients = np.array(sorted(set(patient_ids.tolist())))
    if len(unique_patients) < 2:
        raise ValueError(
            f"Need at least 2 patients for a train/test split; found {len(unique_patients)}."
        )

    # Majority label per patient for optional stratification.
    patient_label = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        vals, counts = np.unique(y[mask], return_counts=True)
        patient_label[pid] = int(vals[np.argmax(counts)])
    strata = np.array([patient_label[p] for p in unique_patients])

    stratify = strata
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < 2:
        print(
            "Warning: cannot stratify patient split (a class has <2 patients); "
            "using an unstratified patient split."
        )
        stratify = None

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_patients = sorted(train_patients.tolist())
    test_patients = sorted(test_patients.tolist())
    train_set = set(train_patients)
    test_set = set(test_patients)

    # Hard guarantee: no patient appears in both splits.
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"Patient leak between splits: {sorted(overlap)}")

    train_idx = np.array(
        [i for i, p in enumerate(patient_ids) if p in train_set], dtype=np.int64
    )
    test_idx = np.array(
        [i for i, p in enumerate(patient_ids) if p in test_set], dtype=np.int64
    )
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Patient split produced an empty train or test row set.")

    return train_idx, test_idx, train_patients, test_patients


def choose_average(y_train: np.ndarray, y_test: np.ndarray) -> str:
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    return "binary" if n_classes == 2 else "macro"


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
):
    from xgboost import XGBClassifier

    average = choose_average(y_train, y_test)
    n_classes = len(class_names)
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        objective="binary:logistic" if n_classes == 2 else "multi:softprob",
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    label_order = list(range(n_classes))
    cm = confusion_matrix(y_test, y_pred, labels=label_order)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average=average, zero_division=0)
        ),
        "average": average,
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(n_classes),
        "classes": class_names,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.astype(int).tolist(),
        },
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "seed": args.seed,
        },
        "split": {
            "test_size_patients": args.test_size,
            "seed": args.seed,
            "level": "patient",
        },
        "patient_id_rule": PATIENT_ID_RE.pattern,
    }
    return model, metrics, y_pred


def print_report(metrics: dict, label_doc: str, train_patients: list[str], test_patients: list[str]) -> None:
    print("=== VGG19-RDS XGBoost baseline (held-out test patients) ===")
    print(label_doc)
    print(
        f"n_train_rows={metrics['n_train_rows']}  n_test_rows={metrics['n_test_rows']}  "
        f"D={metrics['n_features']}  classes={metrics['classes']}"
    )
    print(
        f"n_train_patients={len(train_patients)}  n_test_patients={len(test_patients)}"
    )
    print(f"accuracy = {metrics['accuracy']:.4f}")
    print(f"precision ({metrics['average']}) = {metrics['precision']:.4f}")
    print(f"recall    ({metrics['average']}) = {metrics['recall']:.4f}")
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    print("confusion matrix (rows=true, cols=pred):")
    header = "true\\pred " + " ".join(f"{str(c):>8s}" for c in labels)
    print(header)
    for true_lab, row in zip(labels, matrix):
        print(f"{str(true_lab):>9s} " + " ".join(f"{v:>8d}" for v in row))


def write_patient_list(path: Path, patients: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(patients) + ("\n" if patients else ""), encoding="utf-8")


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rds_path = resolve_rds_path(args.rds)

    try:
        print(f"Using RDS: {rds_path}")
        frame = load_rds_frame(rds_path)
        raw_ids = resolve_row_labels(frame)
        patient_ids = assign_patient_ids(raw_ids)
        labels, label_doc = infer_labels(frame, raw_ids)
        print(label_doc)

        label_col = find_label_column(frame)
        X = feature_matrix(frame, label_col)
        y, _name_to_idx, class_names = encode_labels(labels)
        patient_arr = patient_ids.to_numpy()

        train_idx, test_idx, train_patients, test_patients = patient_level_split(
            patient_arr, y, args.test_size, args.seed
        )

        # Safety: every row of a patient is entirely in one split.
        train_p = set(patient_arr[train_idx])
        test_p = set(patient_arr[test_idx])
        if train_p & test_p:
            raise RuntimeError("Internal error: patient IDs overlap across splits.")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model, metrics, _y_pred = train_and_eval(
            X_train, y_train, X_test, y_test, class_names, args
        )
        metrics["n_train_patients"] = len(train_patients)
        metrics["n_test_patients"] = len(test_patients)
        metrics["label_source"] = label_doc
        metrics["rds_path"] = str(rds_path)

        print_report(metrics, label_doc, train_patients, test_patients)

        model_path = out / "xgboost_vgg19_rds_model.json"
        metrics_path = out / "xgboost_vgg19_rds_metrics.json"
        train_list = out / "train_patients_vgg19.txt"
        test_list = out / "test_patients_vgg19.txt"

        save_model(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        write_patient_list(train_list, train_patients)
        write_patient_list(test_list, test_patients)

        print(f"Wrote model   → {model_path}")
        print(f"Wrote metrics → {metrics_path}")
        print(f"Wrote train patients → {train_list}")
        print(f"Wrote test patients  → {test_list}")
        return 0
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
