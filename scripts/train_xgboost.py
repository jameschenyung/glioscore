#!/usr/bin/env python3
"""
Stage 3 — XGBoost classifier on Stage 2 tile embeddings.

Loads train/test embeddings and labels written by scripts/extract_features.py,
fits an XGBoost classifier on the training set only, and evaluates on the
held-out test set. Patient-level separation was already enforced in Stage 1;
this script never re-splits or mixes patients.

Metrics
-------
Accuracy, precision, recall, and a confusion matrix via sklearn.
For multi-class labels, precision/recall use average='macro' (documented in
the printed report and metrics JSON). Binary 0/1 labels use the binary
averaging path automatically.

Outputs
-------
  outputs/xgboost_model.json     — native XGBoost booster JSON
  outputs/xgboost_metrics.json   — accuracy / precision / recall / confusion matrix

Usage
-----
  python scripts/train_xgboost.py
  python scripts/train_xgboost.py --output-dir outputs --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_SEED = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost on Stage 2 embeddings and evaluate on the "
            "held-out patient-level test set (no re-splitting)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory with Stage 2 npy files and for model/metrics (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train-embeddings",
        type=Path,
        default=None,
        help="Override path to train_embeddings.npy",
    )
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=None,
        help="Override path to train_labels.npy",
    )
    parser.add_argument(
        "--test-embeddings",
        type=Path,
        default=None,
        help="Override path to test_embeddings.npy",
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=None,
        help="Override path to test_labels.npy",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Path for xgboost_model.json (default: <output-dir>/xgboost_model.json)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Path for xgboost_metrics.json (default: <output-dir>/xgboost_metrics.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for XGBoost (default: {DEFAULT_SEED})",
    )
    # Sensible small-data defaults; override via CLI as needed.
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


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    out = args.output_dir
    return {
        "train_embeddings": args.train_embeddings or (out / "train_embeddings.npy"),
        "train_labels": args.train_labels or (out / "train_labels.npy"),
        "test_embeddings": args.test_embeddings or (out / "test_embeddings.npy"),
        "test_labels": args.test_labels or (out / "test_labels.npy"),
        "model_out": args.model_out or (out / "xgboost_model.json"),
        "metrics_out": args.metrics_out or (out / "xgboost_metrics.json"),
    }


def load_array(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {name}: {path}\n"
            "Run Stage 2 first: python scripts/extract_features.py"
        )
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to load {name} from {path}: {exc}") from exc
    if arr.size == 0:
        raise ValueError(f"{name} is empty: {path}")
    return arr


def validate_xy(X: np.ndarray, y: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f"{split} embeddings must be 2-D (N, D); got shape {X.shape}")
    if y.ndim != 1:
        # Allow (N, 1) from some exporters.
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            raise ValueError(f"{split} labels must be 1-D; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split} row count mismatch: embeddings {X.shape[0]} vs labels {y.shape[0]}"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{split} embeddings contain NaN or Inf")
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def choose_average(y_train: np.ndarray, y_test: np.ndarray) -> str:
    """
    Binary (exactly two unique labels across train+test) → average='binary'.
    Otherwise → average='macro' (unweighted mean over classes).
    """
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    return "binary" if n_classes == 2 else "macro"


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
):
    """Fit XGBClassifier on train; score on held-out test only."""
    from xgboost import XGBClassifier

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: train D={X_train.shape[1]} vs test D={X_test.shape[1]}"
        )

    average = choose_average(y_train, y_test)
    # Map labels to 0..C-1 for XGBoost if needed (e.g. sparse class ids).
    classes = np.unique(y_train)
    missing_in_train = sorted(set(y_test.tolist()) - set(classes.tolist()))
    if missing_in_train:
        raise ValueError(
            f"Test labels {missing_in_train} never appear in training; "
            "cannot evaluate those classes."
        )
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_train_enc = np.array([class_to_idx[int(c)] for c in y_train], dtype=np.int64)
    y_test_enc = np.array([class_to_idx[int(c)] for c in y_test], dtype=np.int64)

    n_classes = len(classes)
    # Binary logistic vs multi:softprob; sklearn wrapper picks objective from y.
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
    model.fit(X_train, y_train_enc)
    y_pred_enc = model.predict(X_test)
    # Map predictions back to original label values for human-readable metrics.
    idx_to_class = {i: int(c) for c, i in class_to_idx.items()}
    y_pred = np.array([idx_to_class[int(i)] for i in y_pred_enc], dtype=np.int64)

    # Confusion matrix over the sorted union of original labels for stable axes.
    label_order = sorted(int(c) for c in classes)
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
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(n_classes),
        "classes": label_order,
        "confusion_matrix": {
            "labels": label_order,
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
    }
    return model, metrics, y_pred


def print_report(metrics: dict) -> None:
    print("=== Stage 3 XGBoost (held-out test) ===")
    print(f"n_train={metrics['n_train']}  n_test={metrics['n_test']}  "
          f"D={metrics['n_features']}  classes={metrics['classes']}")
    print(f"accuracy = {metrics['accuracy']:.4f}")
    print(f"precision ({metrics['average']}) = {metrics['precision']:.4f}")
    print(f"recall    ({metrics['average']}) = {metrics['recall']:.4f}")
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    print("confusion matrix (rows=true, cols=pred):")
    header = "true\\pred " + " ".join(f"{c:>6d}" for c in labels)
    print(header)
    for true_lab, row in zip(labels, matrix):
        print(f"{true_lab:>9d} " + " ".join(f"{v:>6d}" for v in row))


def save_model(model, path: Path) -> None:
    """Persist via native XGBoost JSON booster format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # sklearn wrapper exposes get_booster(); save_model writes JSON when suffix is .json
    model.get_booster().save_model(str(path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)

    try:
        X_train = load_array(paths["train_embeddings"], "train_embeddings")
        y_train = load_array(paths["train_labels"], "train_labels")
        X_test = load_array(paths["test_embeddings"], "test_embeddings")
        y_test = load_array(paths["test_labels"], "test_labels")

        X_train, y_train = validate_xy(X_train, y_train, "train")
        X_test, y_test = validate_xy(X_test, y_test, "test")

        model, metrics, _y_pred = train_and_eval(
            X_train, y_train, X_test, y_test, args
        )

        save_model(model, paths["model_out"])
        paths["metrics_out"].parent.mkdir(parents=True, exist_ok=True)
        paths["metrics_out"].write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(metrics)
    print(f"Wrote model   → {paths['model_out']}")
    print(f"Wrote metrics → {paths['metrics_out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
