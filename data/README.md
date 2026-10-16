# Data

Place tile metadata at `metadata.csv` with columns:

| column       | meaning                                      |
|--------------|----------------------------------------------|
| `patient_id` | unique patient (split key; never leak)       |
| `slide_id`   | whole-slide image id                         |
| `tile_path`  | path to a tile image                         |
| `label`      | supervised class label                       |

`example_metadata.csv` is a tiny optional fixture for smoke-testing
`scripts/split_patients.py`. It is not real patient data.
