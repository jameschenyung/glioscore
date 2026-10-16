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

**Canonical user path (preferred — leave the file here; no copy into `data/`):**

```text
C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds
```

Optional cloud/Linux location (gitignored if present):

```text
data/lgggbm_cnn_byRegion_VGG19.rds
```

`scripts/train_vgg19_rds_baseline.py` resolves the file as follows when
`--rds` is omitted: use `data/...` if it exists, otherwise use the canonical
Downloads path if that file exists. Always overrideable with `--rds`.

Rows = Havoc regions across slides/patients; columns = VGG19 features.
The script parses `PATIENT_ID` as the TCGA barcode prefix `TCGA-XX-XXXX` from
row names, splits **patients** (~1/3 test, seed 42), trains XGBoost, and
writes metrics under `outputs/`.

Windows one-liner (clone/checkout + venv + train + print metrics):

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/jameschenyung/glioscore/cursor/vgg19-rds-baseline-92d1/scripts/run_vgg19_baseline_windows.ps1 | iex"
```

Or from an existing checkout on the PR branch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_vgg19_baseline_windows.ps1
```

Direct train (venv already set up):

```bat
python scripts/train_vgg19_rds_baseline.py --rds "C:\Users\james\Downloads\lgggbm_cnn_byRegion_VGG19.rds"
```

Stage 2 smoke test (synthetic tiles, no pretrained download):

```bash
python3 scripts/split_patients.py --metadata data/example_metadata.csv
python3 scripts/extract_features.py --metadata data/example_metadata.csv \
  --synthetic-tiles --no-pretrained
# or wiring-only: add --dry-run
python3 scripts/train_xgboost.py
```
