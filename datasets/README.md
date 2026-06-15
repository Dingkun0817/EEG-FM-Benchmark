# Dataset preprocessing

Preprocessing scripts convert raw public EEG datasets into `EEGData` pickle files under `datasets/data/`.

## Environment variables

| Variable | Dataset | Purpose |
|----------|---------|---------|
| `TUAB_RAW_DIR` | TUAB | Raw EDF root |
| `SLEEPEDF_RAW_DIR` | Sleep-EDF | Raw sleep-cassette directory |
| `SEED_RAW_DIR` | SEED | Raw SEED `.mat` directory |
| `SEED_VIG_RAW_DIR` | SEED-VIG | Raw `.mat` EEG directory |
| `SEED_VIG_LABELS_DIR` | SEED-VIG | PERCLOS label `.mat` directory |
| `EEGMAT_RAW_DIR` | EEGMAT | Raw `.edf` directory |
| `THINGSEEG2_EEG_DIR` | ThingsEEG2 | Preprocessed EEG `.npy` tree |
| `THINGSEEG2_IMG_DIR` | ThingsEEG2 | Image feature directory |
| `CHB_MIT_RAW_DIR` | CHB-MIT | Raw CHB-MIT root |
| `CHB_MIT_OUTPUT_DIR` | CHB-MIT | Output pkl directory |

Defaults point under `datasets/data/raw/<dataset>` when unset.

## Main entry points

Run from the repository root (examples):

```bash
python datasets/TUAB/Preprocess_Dataset.py
python datasets/SleepEDF/Preprocess_Dataset.py
python datasets/SEED/Preprocess_Dataset.py
python datasets/SEED_VIG/Preprocess_Dataset.py
python datasets/EEGMAT/Preprocess_EEGMAT.py
python datasets/ThingsEEG2/Preprocess_Dataset.py
python datasets/CHB_MIT/Preprocess_Dataset.py
python datasets/Dial/Preprocess_Dial.py
python datasets/BNCI2014001/Preprocess_Dataset.py
# ... other BNCI* Preprocess_Dataset.py scripts
```

Outputs are written to `datasets/data/*.pkl` unless a script documents otherwise.

## Further improvements (not implemented)

- Extract shared helpers for the five BNCI `Preprocess_Dataset.py` templates.
- Replace repeated `sys.path.append(project_root)` with editable install (`pip install -e .`).
- Add `datasets/data/*.pkl` to `.gitignore` if they are build artifacts only.
- Remove orphaned CSV copies under `SleepEDF/` if no longer referenced.
