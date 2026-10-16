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

Stage 2 smoke test (synthetic tiles, no pretrained download):

```bash
python3 scripts/split_patients.py --metadata data/example_metadata.csv
python3 scripts/extract_features.py --metadata data/example_metadata.csv \
  --synthetic-tiles --no-pretrained
# or wiring-only: add --dry-run
python3 scripts/train_xgboost.py
```
