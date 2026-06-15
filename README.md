# EEG Benchmark

A framework for fine-tuning and evaluating EEG foundation models across multiple datasets, task modes (cross-subject / few-shot), and both deep learning and traditional ML baselines.

## Features

- **Multiple datasets**: Support for public EEG datasets (BNCI, CHB-MIT, TUAB, SleepEDF, SEED, etc.) with unified pkl-based loading.
- **Task modes**: **Cross** (e.g. LOSO or k-fold cross-validation) and **Fewshot** (per-subject few-shot with configurable train ratio).
- **Model categories**: Traditional **ML** (e.g. CSP_LDA, PSD_*), **DL** (e.g. EEGNet, ShallowConv, Conformer), and **FM** (e.g. BIOT_6D, LaBraM, BENDR).
- **Configurable pipeline**: Preprocessing, optimizer, LR schedule, and training hyperparameters via CLI.


## Environment and installation

- **Python**: 3.10.
- **Dependencies**: See `requirements.txt`. Install with pip after creating a conda environment:

```bash
conda create -n benchmark python=3.10 -y
conda activate benchmark
pip install -r requirements.txt
```

For GPU support, install PyTorch with the appropriate CUDA build from [pytorch.org](https://pytorch.org), then run `pip install -r requirements.txt`.

## Quick start

Run all commands from the repository root (directory that contains `run_finetuning.py`). Logs and metrics go under `./outputs/logs/`.


**LOSO with DL baseline (EEGNet on BNCI2014001, 30% per class per subject)**

```bash
python run_finetuning.py \
  --dataset BNCI2014001 \
  --model_name EEGNet \
  --seeds 0 1 2 \
  --gpuid 0 \
  --task_mode Cross \
  --finetune_strategy full \
  --batch_size 32 \
  --epochs 100 \
  --class_weights True \
  --optimizer_type adam \
  --lr 0.001 \
  --weight_decay 0.01 \
  --use_preprocessing_params True \
  --target_fs 250 \
  --l_freq 8.0 \
  --h_freq 32.0 \
  --norm_method car
```

**Few-shot with foundation model, full fine-tuning (CBraMod on BNCI2014001, 30% per class per subject)**

```bash
python run_finetuning.py \
  --dataset BNCI2014001 \
  --model_name CBraMod \
  --seeds 0 1 2 \
  --gpuid 0 \
  --task_mode Fewshot \
  --train_percentage 0.3 \
  --finetune_strategy full \
  --batch_size 16 \
  --epochs 20 \
  --dropout_rate 0.5 \
  --optimizer_type adamw \
  --lr 0.001 \
  --weight_decay 0.1 \
  --layer_decay 1.0 \
  --use_lr_scheduler True \
  --warmup_epochs 5 \
  --min_lr 1e-06 \
  --use_preprocessing_params True \
  --target_fs 200 \
  --l_freq 0.3 \
  --h_freq 75.0 \
  --notch_freq 60.0 \
  --norm_method car \
  --time_length 4.0
```

**Few-shot with foundation model, head only — backbone frozen (CBraMod on BNCI2014001)**

```bash
python run_finetuning.py \
  --dataset BNCI2014001 \
  --model_name CBraMod \
  --seeds 0 1 2 \
  --gpuid 0 \
  --task_mode Fewshot \
  --train_percentage 0.3 \
  --finetune_strategy head_only \
  --batch_size 16 \
  --epochs 20 \
  --dropout_rate 0.5 \
  --label_smoothing 0.0 \
  --optimizer_type adamw \
  --lr 0.001 \
  --weight_decay 0.1 \
  --use_lr_scheduler True \
  --warmup_epochs 5 \
  --min_lr 1e-06 \
  --min_weight_decay 0.0001 \
  --clip_grad_norm 1.0 \
  --use_preprocessing_params True \
  --target_fs 200 \
  --l_freq 0.3 \
  --h_freq 75.0 \
  --notch_freq 60.0 \
  --norm_method car \
  --time_length 4.0
```

**Traditional ML**

```bash
python run_finetuning.py \
  --dataset BNCI2014001 \
  --model_name CSP_LDA \
  --task_mode Cross \
  --seeds 0 1 2
```

### Hyperparameters

The commands above are **reference runs** for specific dataset–model pairs (e.g. CBraMod and EEGNet on BNCI2014001). They are not meant as defaults for every experiment.

In practice, **each dataset and each model usually needs its own hyperparameter search** to reach strong performance. Learning rate, batch size, epochs,  optimizer settings, and preprocessing settings, all interact with data characteristics and model architecture, so there is **no single universal configuration** that works well everywhere.

We encourage you to explore different hyperparameter settings rather than relying on a single fixed recipe.

See `python run_finetuning.py --help` for the full flag list. 

## Datasets

Data is read from `datasets/data/<DatasetName>.pkl`. If a pkl file is missing, the corresponding dataset module (e.g. `datasets/BNCI2014001/Preprocess_Dataset.py`) is used to download or generate it. Supported datasets include:

| Dataset       | BCI Paradigm     | Source |
|---------------|------------------|--------|
| BNCI2014001   | MI               | MOABB |
| BNCI2014004   | MI               | MOABB |
| BNCI2015001   | MI               | MOABB |
| BNCI2014008   | P300             | MOABB |
| BNCI2014009   | P300             | MOABB |
| CHB_MIT       | Clinic           | <https://physionet.org/content/chbmit/1.0.0/> |
| TUAB          | Clinic           | <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/> |
| SleepEDF      | Sleep            | <https://physionet.org/content/sleep-edfx/1.0.0/> |
| SEED          | Emotion          | <https://bcmi.sjtu.edu.cn/home/seed/downloads.html> |
| SEED_VIG      | Fatigue          | <https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html> |
| Dial          | SSVEP            | MOABB |
| ThingsEEG2    | Visual decoding  | <https://things-initiative.org/> |
| EEGMAT        | Workload         | <https://physionet.org/content/eegmat/1.0.0/> |



## Models

- **ML (traditional)**: CSP_LDA, Xdawn_LDA, PSD_Ridge, DE_LDA, TRCA, PSD_LDA, PSD_SVM. These use a separate training/evaluation path and do not require GPU.
- **DL**: EEGNet, ShallowConv, CNNTransformer, Deformer, Conformer, LMDA, FBCNet, MSCFormer, TSception.
- **FM**: LaBraM, BENDR, NeuroGPT, BrainOmni_Tiny, BrainOmni_Base, LUNA_Base, LUNA_Huge, LUNA_Large, EEGMamba, SingLEM, TFMTokenizer, CBraMod, EEGPT, BIOT_6D, BIOT_1D, BIOT_2D.

Use `--model_name` with one of the names above (see `models/config.py` for the full mapping).

## Main arguments

| Argument              | Description |
|-----------------------|-------------|
| `--dataset`           | Dataset name (e.g. CHB_MIT, BNCI2014001). |
| `--model_name`        | Model name (e.g. BIOT_6D, EEGNet, CSP_LDA). |
| `--task_mode`         | `Cross` or `Fewshot`. |
| `--finetune_strategy` | `full` (all params) or `head_only` (task head only). |
| `--gpuid`             | GPU device ID. |
| `--seeds`             | Random seeds, e.g. `0 1 2`. |
| `--epochs`             | Number of training epochs. |
| `--batch_size`        | Batch size for training and evaluation. |
| `--train_percentage`  | In Fewshot, fraction of each class per subject used for training (default 0.3). |

Run `python run_finetuning.py --help` for all options (optimizer, LR scheduler, preprocessing, etc.).

## Output layout

- **Root**: `./outputs/logs/<dataset>/<model>/<task_mode>/<finetune_strategy>/<timestamp>/`.
- **Per seed**: Subfolder named by best metric and seed (e.g. `0.85_seed_0`) containing:
  - `results.json` – metrics and config.
  - `training_log.txt` – training log.
  - `results.txt` – human-readable summary and copy-paste command.
- **Multi-seed**: CSV files in the same timestamp folder with hyperparameters and per-seed/per-subject metrics.

## Project structure

- `run_finetuning.py` – Entry point; CLI, dataset/model loading, split loops, training, and result aggregation.
- `utils/` – Data loading (`EEGDataLoader`, `dataset_split`, `model_loader`), training loop (`trainer`), metrics, logger, preprocessing, optimizer helpers.
- `models/` – Model and preprocessor implementations under `ML/`, `DL/`, and `FM/`; `config.py` maps model names to categories.
- `datasets/` – Per-dataset preprocessing and pkl generation (e.g. `Preprocess_Dataset.py`, `tool.py`).
- `outputs/` – All run outputs (logs and result files).

Reference or legacy code directories are not part of the main pipeline.


## Reference and citation

Upstream open-source repositories for DL and FM models in this benchmark:

| Model | GitHub |
|-------|--------|
| **DL** | |
| EEGNet | <https://github.com/vlawhern/arl-eegmodels> |
| ShallowConv | <https://github.com/braindecode/braindecode> |
| Deformer | <https://github.com/yi-ding-cs/EEG-Deformer> |
| Conformer | <https://github.com/eeyhsong/EEG-Conformer> |
| LMDA | <https://github.com/MiaoZhengQing/LMDA-Code> |
| FBCNet | <https://github.com/ravikiran-mane/FBCNet> |
| MSCFormer | <https://github.com/snailpt/MSCFormer> |
| TSception | <https://github.com/yi-ding-cs/TSception> |
| **FM** | |
| LaBraM | <https://github.com/935963004/LaBraM> |
| BENDR | <https://github.com/SPOClab-ca/BENDR> |
| NeuroGPT | <https://github.com/wenhui0206/NeuroGPT> |
| BrainOmni_Tiny / BrainOmni_Base | <https://github.com/OpenTSLab/BrainOmni> |
| LUNA_Base / LUNA_Huge / LUNA_Large | <https://github.com/pulp-bio/BioFoundation> |
| EEGMamba | <https://github.com/wjq-learning/EEGMamba> |
| SingLEM | <https://github.com/ttlabtuat/SingLEM> |
| TFMTokenizer | <https://github.com/Jathurshan0330/TFM-Tokenizer> |
| CBraMod | <https://github.com/wjq-learning/CBraMod> |
| EEGPT | <https://github.com/BINE022/EEGPT> |
| BIOT_6D / BIOT_1D / BIOT_2D | <https://github.com/ycq091044/BIOT> |

## License

See the repository for license information, if applicable.
