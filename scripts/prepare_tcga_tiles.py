#!/usr/bin/env python3
"""
Prepare TCGA LGG/GBM tile archive for DINOv2 PoC comparison.

Workflow
--------
1. Index members of ``tiles_selected_TCGA_LGG_GBM.tar.gz`` (no full extract).
2. Parse PATIENT_ID (TCGA-XX-XXXX) and LGG/GBM labels from member paths.
3. Optionally subsample a fraction of *patients* (seeded) for a PoC.
4. Extract only those patients' tiles under ``data/tiles_tcga/``.
5. Write ``data/metadata_tcga_tiles.csv`` (patient_id, slide_id, tile_path, label).

Splits are *not* done here — use ``scripts/split_patients.py`` on the CSV so
train/test assignment stays patient-level and deterministic (seed 42).

Usage
-----
  python scripts/prepare_tcga_tiles.py \\
      --archive data/tiles_selected_TCGA_LGG_GBM.tar.gz \\
      --patient-fraction 0.333 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "tiles_selected_TCGA_LGG_GBM.tar.gz"
DEFAULT_EXTRACT_DIR = REPO_ROOT / "data" / "tiles_tcga"
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata_tcga_tiles.csv"
DEFAULT_INDEX = REPO_ROOT / "outputs" / "tcga_tiles_archive_index.json"
DEFAULT_SUBSET_DOC = REPO_ROOT / "outputs" / "dinov2_poc_subset.json"
DEFAULT_SEED = 42
DEFAULT_PATIENT_FRACTION = 1.0 / 3.0

PATIENT_ID_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
# Underscore-separated tokens (e.g. LGG_TCGA-...) — avoid \b because "_" is a word char.
LGG_GBM_RE = re.compile(r"(?<![A-Za-z0-9])(LGG|GBM)(?![A-Za-z0-9])", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Index / selectively extract TCGA glioma tiles and write metadata CSV "
            "with patient-level labels (never split by region color)."
        )
    )
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    p.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--index-out", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--subset-doc", type=Path, default=DEFAULT_SUBSET_DOC)
    p.add_argument(
        "--patient-fraction",
        type=float,
        default=DEFAULT_PATIENT_FRACTION,
        help=(
            "Fraction of unique patients to keep for the PoC "
            f"(default: {DEFAULT_PATIENT_FRACTION:.3f} ≈ 1/3). Use 1.0 for all."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Build/save the archive index JSON and exit (no extract).",
    )
    p.add_argument(
        "--reuse-index",
        type=Path,
        default=None,
        help="Reuse a previously written archive index JSON instead of re-scanning.",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Assume tiles already extracted; only (re)write metadata from index+subset.",
    )
    p.add_argument(
        "--gdc-fallback",
        action="store_true",
        help=(
            "If a patient has no LGG/GBM token in paths, resolve label via GDC "
            "project_id (TCGA-LGG / TCGA-GBM)."
        ),
    )
    p.add_argument(
        "--max-tiles-per-patient",
        type=int,
        default=None,
        help=(
            "Optional PoC speed cap: keep at most N tiles per patient "
            "(deterministic sample, seed). Omit to keep all tiles for kept patients."
        ),
    )
    return p.parse_args(argv)


def is_image_member(name: str) -> bool:
    path = Path(name)
    if path.name.startswith(".") or name.endswith("/"):
        return False
    return path.suffix.lower() in IMAGE_SUFFIXES


def extract_patient_id(text: str) -> str | None:
    m = PATIENT_ID_RE.search(text)
    return m.group(1).upper() if m else None


def extract_label_token(text: str) -> str | None:
    """
    Infer LGG/GBM from path components.

    Important: the archive root ``tiles_selected_TCGA_LGG_GBM/`` contains *both*
    tokens and must be ignored — otherwise every tile falsely matches LGG.
    Prefer an exclusive path segment (``.../LGG/...`` or ``.../GBM/...``).
    """
    parts = Path(text).parts
    searchable: list[str] = []
    for i, part in enumerate(parts):
        upper = part.upper()
        # Skip the combined archive / dataset folder name.
        if "LGG" in upper and "GBM" in upper:
            continue
        searchable.append(part)
    for part in searchable:
        m = LGG_GBM_RE.fullmatch(part.strip())
        if m:
            return m.group(1).upper()
    # Also allow tokens embedded in a component (e.g. LGG_TCGA-...) but not both.
    for part in searchable:
        upper = part.upper()
        if "LGG" in upper and "GBM" in upper:
            continue
        m = LGG_GBM_RE.search(part)
        if m:
            return m.group(1).upper()
    return None


def slide_id_from_path(member: str, patient_id: str) -> str:
    """Best-effort slide id: longest path component containing the patient barcode."""
    parts = Path(member).parts
    candidates = [p for p in parts if patient_id.lower() in p.lower()]
    if candidates:
        # Prefer a component that looks like a full slide barcode.
        for c in sorted(candidates, key=len, reverse=True):
            if len(c) > len(patient_id):
                return c
        return candidates[-1]
    stem = Path(member).stem
    return stem or patient_id


def index_archive(archive: Path) -> list[dict]:
    """Stream-list image members; return records with patient_id / path label guess."""
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")

    records: list[dict] = []
    skipped_no_patient = 0
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if not is_image_member(name):
                continue
            pid = extract_patient_id(name)
            if pid is None:
                skipped_no_patient += 1
                continue
            records.append(
                {
                    "member": name,
                    "patient_id": pid,
                    "path_label": extract_label_token(name),
                    "size": int(member.size) if member.size is not None else 0,
                }
            )
    print(
        f"Indexed {len(records)} image tiles "
        f"({skipped_no_patient} images skipped: no TCGA patient barcode in path)"
    )
    return records


def load_or_build_index(args: argparse.Namespace) -> list[dict]:
    if args.reuse_index is not None:
        path = args.reuse_index
        if not path.is_file():
            raise FileNotFoundError(f"--reuse-index not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"] if isinstance(data, dict) and "records" in data else data
        print(f"Reused index with {len(records)} tiles from {path}")
        return records

    records = index_archive(args.archive)
    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive": str(args.archive),
        "n_tiles": len(records),
        "n_patients": len({r["patient_id"] for r in records}),
        "records": records,
    }
    args.index_out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"Wrote archive index → {args.index_out}")
    return records


def majority_patient_label(records: list[dict]) -> dict[str, str | None]:
    """Per-patient label from path tokens; None if ambiguous/missing."""
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        lab = r.get("path_label")
        if lab:
            votes[r["patient_id"]][lab] += 1
    out: dict[str, str | None] = {}
    for pid in {r["patient_id"] for r in records}:
        counts = votes.get(pid, {})
        if not counts:
            out[pid] = None
            continue
        # Majority; require a unique winner.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            out[pid] = ranked[0][0]
        else:
            out[pid] = None
    return out


def fetch_gdc_labels(patient_ids: list[str]) -> dict[str, str]:
    """
    Map TCGA submitter_id → LGG/GBM via GDC projects TCGA-LGG / TCGA-GBM.

    Uses the public GDC API (no auth). Batches to stay within URL limits.
    """
    import json as _json
    import urllib.error
    import urllib.request

    mapping: dict[str, str] = {}
    # Filter cases by submitter_id in batches.
    batch_size = 80
    endpoint = "https://api.gdc.cancer.gov/cases"

    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i : i + batch_size]
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "submitter_id",
                        "value": batch,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "project.project_id",
                        "value": ["TCGA-LGG", "TCGA-GBM"],
                    },
                },
            ],
        }
        params = (
            f"filters={urllib.request.quote(_json.dumps(filters))}"
            f"&fields=submitter_id,project.project_id"
            f"&size={len(batch)}"
            f"&format=JSON"
        )
        url = f"{endpoint}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"GDC API request failed: {exc}") from exc

        hits = payload.get("data", {}).get("hits", [])
        for hit in hits:
            sid = str(hit.get("submitter_id", "")).upper()
            project = hit.get("project", {}) or {}
            project_id = str(project.get("project_id", "")).upper()
            if project_id == "TCGA-LGG":
                mapping[sid] = "LGG"
            elif project_id == "TCGA-GBM":
                mapping[sid] = "GBM"
    return mapping


def resolve_labels(
    records: list[dict],
    *,
    gdc_fallback: bool,
) -> tuple[dict[str, str], str]:
    path_labels = majority_patient_label(records)
    resolved: dict[str, str] = {}
    missing = []
    for pid, lab in path_labels.items():
        if lab in {"LGG", "GBM"}:
            resolved[pid] = lab
        else:
            missing.append(pid)

    sources = [f"path tokens for {len(resolved)} patients"]
    if missing and gdc_fallback:
        print(f"Resolving {len(missing)} patient labels via GDC…")
        gdc = fetch_gdc_labels(sorted(missing))
        for pid in missing:
            if pid in gdc:
                resolved[pid] = gdc[pid]
        still = [p for p in missing if p not in resolved]
        sources.append(f"GDC for {len(missing) - len(still)} patients")
        missing = still

    if missing:
        raise ValueError(
            f"Could not resolve LGG/GBM labels for {len(missing)} patients "
            f"(examples: {missing[:5]}). Re-run with --gdc-fallback or inspect paths."
        )

    doc = "; ".join(sources)
    return resolved, doc


def subsample_patients(
    patient_ids: list[str],
    patient_labels: dict[str, str],
    fraction: float,
    seed: int,
) -> list[str]:
    """Deterministic patient-level subsample (stratified by label when possible)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"patient_fraction must be in (0, 1]; got {fraction}")
    patients = sorted(set(patient_ids))
    if fraction >= 1.0:
        return patients

    strata = np.array([patient_labels[p] for p in patients])
    # train_test_split with test_size=fraction keeps `fraction` in the "test" arm.
    # We want a *subset* of size ≈ fraction, so take the test arm.
    stratify = strata
    _, counts = np.unique(strata, return_counts=True)
    if counts.min() < 2:
        print("Warning: cannot stratify patient subsample; using unstratified draw.")
        stratify = None

    _holdout, keep = train_test_split(
        patients,
        test_size=fraction,
        random_state=seed,
        stratify=stratify,
    )
    keep = sorted(keep)
    print(
        f"PoC subset: kept {len(keep)}/{len(patients)} patients "
        f"(fraction={fraction}, seed={seed})"
    )
    return keep


def extract_members(
    archive: Path,
    members: list[str],
    extract_dir: Path,
) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(members)
    n_done = 0
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if member.name not in wanted:
                continue
            # Safety: refuse absolute / path-traversal members.
            dest = (extract_dir / member.name).resolve()
            if not str(dest).startswith(str(extract_dir.resolve())):
                raise RuntimeError(f"Refusing path-traversal tar member: {member.name}")
            tf.extract(member, path=extract_dir)
            n_done += 1
            if n_done % 500 == 0:
                print(f"  extracted {n_done}/{len(wanted)} …")
    print(f"Extracted {n_done} tiles → {extract_dir}")


def encode_label(label: str) -> int:
    """LGG=0, GBM=1 (matches VGG19 baseline convention)."""
    lab = label.upper()
    if lab == "LGG":
        return 0
    if lab == "GBM":
        return 1
    raise ValueError(f"Unexpected label {label!r}; expected LGG or GBM")


def write_metadata(
    records: list[dict],
    patient_labels: dict[str, str],
    extract_dir: Path,
    metadata_out: Path,
) -> int:
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with metadata_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["patient_id", "slide_id", "tile_path", "label"]
        )
        writer.writeheader()
        for r in records:
            pid = r["patient_id"]
            lab = patient_labels[pid]
            member = r["member"]
            # tile_path relative to extract_dir (and usable with --tile-root).
            tile_path = str(Path(member))
            # Verify file exists when extract_dir is populated.
            abs_path = extract_dir / member
            if not abs_path.is_file():
                # Some tars strip a top-level prefix inconsistently; try basename walk skip.
                raise FileNotFoundError(
                    f"Expected extracted tile missing: {abs_path}"
                )
            writer.writerow(
                {
                    "patient_id": pid,
                    "slide_id": slide_id_from_path(member, pid),
                    "tile_path": tile_path,
                    "label": encode_label(lab),
                }
            )
            n += 1
    print(f"Wrote metadata ({n} rows) → {metadata_out}")
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = load_or_build_index(args)
        if args.index_only:
            return 0

        all_patients = sorted({r["patient_id"] for r in records})
        patient_labels, label_doc = resolve_labels(
            records, gdc_fallback=args.gdc_fallback or True
        )
        print(f"Label source: {label_doc}")
        n_lgg = sum(1 for p in all_patients if patient_labels[p] == "LGG")
        n_gbm = sum(1 for p in all_patients if patient_labels[p] == "GBM")
        print(f"Patients in archive: {len(all_patients)} (LGG={n_lgg}, GBM={n_gbm})")

        keep_patients = set(
            subsample_patients(
                all_patients,
                patient_labels,
                args.patient_fraction,
                args.seed,
            )
        )
        subset_records = [r for r in records if r["patient_id"] in keep_patients]

        # Optional per-patient tile cap (still patient-level; never mixes splits).
        tile_cap_doc = None
        if args.max_tiles_per_patient is not None:
            if args.max_tiles_per_patient < 1:
                raise ValueError("--max-tiles-per-patient must be >= 1")
            rng = np.random.default_rng(args.seed)
            by_patient: dict[str, list[dict]] = defaultdict(list)
            for r in subset_records:
                by_patient[r["patient_id"]].append(r)
            capped: list[dict] = []
            for pid in sorted(by_patient):
                tiles = by_patient[pid]
                # Stable order before RNG so the sample is reproducible.
                tiles = sorted(tiles, key=lambda x: x["member"])
                if len(tiles) > args.max_tiles_per_patient:
                    idx = rng.choice(
                        len(tiles), size=args.max_tiles_per_patient, replace=False
                    )
                    tiles = [tiles[i] for i in sorted(idx.tolist())]
                capped.extend(tiles)
            tile_cap_doc = {
                "max_tiles_per_patient": args.max_tiles_per_patient,
                "n_tiles_before_cap": len(subset_records),
                "n_tiles_after_cap": len(capped),
            }
            print(
                f"Tile cap: {len(subset_records)} → {len(capped)} "
                f"(max {args.max_tiles_per_patient}/patient, seed={args.seed})"
            )
            subset_records = capped

        members = [r["member"] for r in subset_records]
        print(
            f"Subset tiles: {len(subset_records)} "
            f"from {len(keep_patients)} patients"
        )

        # Document subset for the comparison write-up.
        args.subset_doc.parent.mkdir(parents=True, exist_ok=True)
        subset_payload = {
            "seed": args.seed,
            "patient_fraction": args.patient_fraction,
            "sampling": (
                "sklearn.train_test_split on sorted unique patient_ids, "
                "stratified by LGG/GBM when possible; kept the 'test' arm "
                f"of size ≈ {args.patient_fraction}"
            ),
            "n_patients_archive": len(all_patients),
            "n_patients_poc": len(keep_patients),
            "n_tiles_poc": len(subset_records),
            "tile_cap": tile_cap_doc,
            "label_source": label_doc,
            "patients": sorted(keep_patients),
            "label_counts": {
                "LGG": sum(1 for p in keep_patients if patient_labels[p] == "LGG"),
                "GBM": sum(1 for p in keep_patients if patient_labels[p] == "GBM"),
            },
            "archive": str(args.archive),
            "extract_dir": str(args.extract_dir),
        }
        args.subset_doc.write_text(
            json.dumps(subset_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote subset doc → {args.subset_doc}")

        if not args.skip_extract:
            extract_members(args.archive, members, args.extract_dir)

        write_metadata(
            subset_records,
            patient_labels,
            args.extract_dir,
            args.metadata_out,
        )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
