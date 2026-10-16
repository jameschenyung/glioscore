#!/usr/bin/env bash
# DINOv2 tile PoC vs VGG19 region baseline — patient-level splits only.
# Assumes archive already downloaded to data/tiles_selected_TCGA_LGG_GBM.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"

ARCHIVE="${ARCHIVE:-data/tiles_selected_TCGA_LGG_GBM.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-data/tiles_tcga}"
METADATA="${METADATA:-data/metadata_tcga_tiles.csv}"
OUT="${OUT:-outputs/dinov2_poc}"
MODEL="${MODEL:-vit_small_patch14_dinov2.lvd142m}"
PATIENT_FRACTION="${PATIENT_FRACTION:-0.333}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# CPU PoC default: cap tiles/patient so embedding finishes in reasonable time.
# Set MAX_TILES_PER_PATIENT=0 to disable the cap.
MAX_TILES_PER_PATIENT="${MAX_TILES_PER_PATIENT:-32}"

mkdir -p "$OUT" "$EXTRACT_DIR"

echo "=== 1. Prepare tiles (index → patient subset → extract → metadata) ==="
PREPARE_ARGS=(
  --archive "$ARCHIVE"
  --extract-dir "$EXTRACT_DIR"
  --metadata-out "$METADATA"
  --index-out "$OUT/tcga_tiles_archive_index.json"
  --subset-doc "$OUT/dinov2_poc_subset.json"
  --patient-fraction "$PATIENT_FRACTION"
  --seed "$SEED"
  --gdc-fallback
)
if [[ "${MAX_TILES_PER_PATIENT}" != "0" ]]; then
  PREPARE_ARGS+=(--max-tiles-per-patient "$MAX_TILES_PER_PATIENT")
fi
python3 scripts/prepare_tcga_tiles.py "${PREPARE_ARGS[@]}"

echo "=== 2. Patient-level train/test split (seed=$SEED, ~1/3 test) ==="
python3 scripts/split_patients.py \
  --metadata "$METADATA" \
  --output-dir "$OUT" \
  --seed "$SEED" \
  --test-size 0.333

echo "=== 3. DINOv2 embeddings ($MODEL) ==="
python3 scripts/extract_features.py \
  --metadata "$METADATA" \
  --train-patients "$OUT/train_patients.txt" \
  --test-patients "$OUT/test_patients.txt" \
  --output-dir "$OUT" \
  --tile-root "$EXTRACT_DIR" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --device cpu \
  --num-workers 2

echo "=== 4. XGBoost (Stage-3 defaults) ==="
python3 scripts/train_xgboost.py \
  --output-dir "$OUT" \
  --seed "$SEED"

echo "Done. Metrics: $OUT/xgboost_metrics.json"
