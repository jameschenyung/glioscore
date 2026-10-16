# Data

Place tile metadata at `metadata.csv` with columns:

| column       | meaning                                      |
|--------------|----------------------------------------------|
| `patient_id` | unique patient (split key; never leak)       |
| `slide_id`   | whole-slide image id                         |
| `tile_path`  | path to a tile image                         |
| `label`      | supervised class label                       |

`example_metadata.csv` is a tiny optional fixture for smoke-testing
`scripts/split_patients.py` and `scripts/extract_features.py`. It is not
real patient data. Tile paths are relative to `data/` unless absolute.

## VGG19 Havoc-region RDS (manager baseline)

Expected path (gitignored — do not commit the binary):

```text
data/lgggbm_cnn_byRegion_VGG19.rds
```

Copy from the user machine, for example:

```text
C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds
```

→ `data/lgggbm_cnn_byRegion_VGG19.rds`

Rows = Havoc regions across slides/patients; columns = VGG19 features.
`scripts/train_vgg19_rds_baseline.py` parses `PATIENT_ID` as the TCGA barcode
prefix `TCGA-XX-XXXX` from row names, splits **patients** (~1/3 test, seed 42),
trains XGBoost, and writes metrics under `outputs/`.

```bash
python3 scripts/train_vgg19_rds_baseline.py
```

Stage 2 smoke test (synthetic tiles, no pretrained download):

```bash
python3 scripts/split_patients.py --metadata data/example_metadata.csv
python3 scripts/extract_features.py --metadata data/example_metadata.csv \
  --synthetic-tiles --no-pretrained
# or wiring-only: add --dry-run
python3 scripts/train_xgboost.py
```
