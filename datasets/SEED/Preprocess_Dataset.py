"""
SEED dataset preprocessing: load raw .mat files, compute train/test usage in memory, write pkl with usage.
Output pkl only, no intermediate npy/csv.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.insert(0, project_root)

from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl

# Raw data directory (set path as needed)
DEFAULT_DATA_DIR = os.environ.get(
    'SEED_RAW_DIR',
    os.path.join(project_root, 'datasets', 'data', 'raw', 'SEED'),
)
# Output pkl directory; default: project datasets/data
DEFAULT_OUTPUT_DIR = os.path.join(project_root, 'datasets', 'data')

SEED_CH_NAMES = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2',
]
LABEL_MAP = {1: 2, 0: 1, -1: 0}  # positive->2, neutral->1, negative->0
FIXED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
FS = 200
SEGMENT_DURATION = 1
SEGMENT_SAMPLES = FS * SEGMENT_DURATION
CHANNELS = 62


def _extract_number(key):
    numbers = re.findall(r'\d+', key)
    return int(numbers[0]) if numbers else float('inf')


def _build_subject_files(data_dir):
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    subject_files = {}
    for mat_file in mat_files:
        match = re.match(r'^(\d+)_(\d{8})\.mat$', mat_file)
        if match:
            subject_id = int(match.group(1))
            date_str = match.group(2)
            subject_files.setdefault(subject_id, []).append((mat_file, date_str))
        else:
            print(f"Warning: Cannot parse filename format: {mat_file}")

    for subject_id, files in subject_files.items():
        if len(files) != 3:
            print(f"Warning: Subject {subject_id} has {len(files)} files, expected 3")
        subject_files[subject_id] = sorted(files, key=lambda x: x[1])
    return subject_files


def _load_seed_raw_segments(data_dir, sessions=None):
    """
    Parse raw SEED .mat files into segmented arrays and in-memory meta.
    sessions: int or list to use (e.g. 1 or [1,2,3]); None means all sessions.
    """
    if sessions is not None:
        sessions = [sessions] if isinstance(sessions, int) else list(sessions)

    subject_files = _build_subject_files(data_dir)
    all_data = []
    all_labels = []
    all_subjects = []
    all_sessions = []
    all_trial_idx = []
    all_segment_idx = []

    for subject_id in sorted(subject_files.keys()):
        session_files = subject_files[subject_id]
        for session_idx, (mat_file, date_str) in enumerate(session_files, start=1):
            if sessions is not None and session_idx not in sessions:
                continue

            file_path = os.path.join(data_dir, mat_file)
            data = sio.loadmat(file_path)
            keys = [key for key in data.keys() if not key.startswith('__')]

            eeg_keys = []
            for key in keys:
                if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
                    if data[key].shape[0] == CHANNELS or data[key].shape[1] == CHANNELS:
                        eeg_keys.append(key)

            eeg_keys_sorted = sorted(eeg_keys, key=_extract_number)
            if len(eeg_keys_sorted) != 15:
                print(f"Warning: File {mat_file} has {len(eeg_keys_sorted)} EEG data blocks, expected 15")
                continue

            for trial_idx, eeg_key in enumerate(eeg_keys_sorted, start=1):
                eeg_data = data[eeg_key]
                if eeg_data.shape[0] != CHANNELS:
                    if eeg_data.shape[1] == CHANNELS:
                        eeg_data = eeg_data.T
                    else:
                        print(f"Warning: File {mat_file} {eeg_key} has abnormal shape: {eeg_data.shape}")
                        continue

                num_segments = eeg_data.shape[1] // SEGMENT_SAMPLES
                label = FIXED_LABELS[trial_idx - 1]

                for seg_idx in range(num_segments):
                    start = seg_idx * SEGMENT_SAMPLES
                    end = start + SEGMENT_SAMPLES
                    segment = eeg_data[:, start:end]

                    all_data.append(segment)
                    all_labels.append(label)
                    all_subjects.append(subject_id)
                    all_sessions.append(session_idx)
                    all_trial_idx.append(trial_idx)
                    all_segment_idx.append(seg_idx)

    data = np.array(all_data)
    labels_raw = np.array(all_labels)
    meta_df = pd.DataFrame({
        'index': range(len(all_data)),
        'subject': all_subjects,
        'session': all_sessions,
        'trial': all_trial_idx,
        'segment': all_segment_idx,
        'original_label': all_labels,
        'emotion': [LABEL_MAP[l] for l in all_labels],
        'date': [subject_files[sub][sess - 1][1] for sub, sess in zip(all_subjects, all_sessions)],
    })
    return data, labels_raw, meta_df


def _compute_usage_from_meta(meta_df, labels):
    """Per subject, per label: first trial = train, rest = test. meta_df row-aligned with labels."""
    n = len(labels)
    usage = np.full(n, 'test', dtype='U10')
    meta_df = meta_df.reset_index(drop=True)
    for subject in meta_df['subject'].unique():
        subj_mask = (meta_df['subject'] == subject).values
        subj_meta = meta_df.loc[subj_mask]
        subj_labels = labels[subj_mask]
        for label in np.unique(subj_labels):
            label_mask = subj_labels == label
            trials = subj_meta.loc[label_mask, 'trial'].unique()
            if len(trials) == 0:
                continue
            first_trial = trials[0]
            global_idx = subj_meta.index[(subj_meta['trial'] == first_trial) & (subj_labels == label)].values
            usage[global_idx] = 'train'
    return usage


def load_seed_data_from_raw(sessions=None, data_dir=None, output_dir=None):
    """
    Load from raw SEED .mat files, compute usage in memory, write pkl with usage directly.
    sessions: int or list to use (e.g. 1 or [1,2,3]); None means all sessions.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    data, labels_raw, meta_df = _load_seed_raw_segments(data_dir, sessions=sessions)
    labels = np.array([LABEL_MAP.get(int(l), l) for l in labels_raw])
    subject_ids = meta_df['subject'].values
    usage = _compute_usage_from_meta(meta_df, labels)

    eeg_data_obj = EEGData(
        dataset_name='SEED',
        eeg_data=data,
        subject_ids=subject_ids,
        channel_names=SEED_CH_NAMES,
        sampling_rate=FS,
        labels=labels,
        usage=usage,
    )
    os.makedirs(output_dir, exist_ok=True)
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=output_dir)
    return eeg_data_obj


def download_data_SEED(sessions=None, data_dir=None, output_dir=None, **kwargs):
    """Called by run_finetuning：generate with usage SEED.pkl。"""
    return load_seed_data_from_raw(sessions=sessions or [1], data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    try:
        obj = load_seed_data_from_raw(sessions=[1])
        print("SEED pkl generated; usage written.")
        out = os.path.join(DEFAULT_OUTPUT_DIR, "SEED.pkl")
        if os.path.exists(out):
            loaded = load_eeg_data_from_pkl(out)
            print(
                f"Validation: sample count={loaded.eeg_data.shape[0]}, "
                f"usage stats: train={np.sum(loaded.usage == 'train')}, "
                f"test={np.sum(loaded.usage == 'test')}"
            )
    except Exception:
        import traceback
        traceback.print_exc()
