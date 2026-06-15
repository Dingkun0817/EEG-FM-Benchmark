import os
import numpy as np
import mne
import sys
import pickle
from multiprocessing import Pool

# Directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# Add project root to Python path
sys.path.append(project_root)

# Import EEGData and save helpers from utils
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl

# Define constants
data_root = os.environ.get(
    'TUAB_RAW_DIR',
    os.path.join(project_root, 'datasets', 'data', 'raw', 'TUAB'),
)
channel_std = "01_tcp_ar"

# channels to drop
drop_channels = ['PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', \
                 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF', 'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE', 'EEG PG2-REF', 'EEG PG1-REF']
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])

# standard common channel order (20 channels, channels common to both sets)
common_channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                   'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
# clean channel names
clean_channel_names = [ch.replace('EEG ', '').replace('-REF', '') for ch in common_channels]

def process_file(file_path, label, subject_id):
    """
    process one EDF file
    
    Args:
        file_path: EDF file path
        label: sample label
        subject_id: subject ID
    
    Returns:
        list of processed samples, each sample has EEG data, label, and subject ID
    """
    # Read EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # drop unused channels
    useless_chs = []
    for ch in drop_channels:
        if ch in raw.ch_names:
            useless_chs.append(ch)
    raw.drop_channels(useless_chs)
    
    # Keep only common channels
    available_common_channels = [ch for ch in common_channels if ch in raw.ch_names]
    if len(available_common_channels) != len(common_channels):
        print(f"Warning: Missing common channels for {os.path.basename(file_path)}")
        print(f"Expected common channels: {set(common_channels)}")
        print(f"Available common channels: {set(available_common_channels)}")
        raise Exception("Missing common channels!")
    
    # reorder to common channel order
    raw.reorder_channels(common_channels)
    
    # resample
    raw.resample(250, n_jobs=5)
    
    # split into segments
    segment_len = 10 * 250  # 10 s at 250 Hz
    raw_data = raw.get_data(units='uV')
    total_len = raw_data.shape[1]
    
    # keep at most first 3 min per subject (3 min * 60 s/min * 250 Hz = 45000 samples)
    max_len = 3 * 60 * 250  # 45000 samples
    if total_len > max_len:
        print(f"File {os.path.basename(file_path)} has {total_len} samples, trimming to first {max_len} samples (3 minutes)")
        raw_data = raw_data[:, :max_len]
        total_len = max_len
    
    samples = []
    for i in range(total_len // segment_len):
        segment = raw_data[:, i * segment_len:(i + 1) * segment_len]
        samples.append((segment, label, subject_id))
    
    return samples

def process_files_in_folder(folder_path, label, subject_id_map, num_workers=4):
    """
    Process all EDF files in a folder (supports multiprocessing)
    
    Args:
        folder_path: folder paths
        label: sample label
        subject_id_map: subject ID mapping dictionary for consistency across the dataset
        num_workers: number of parallel worker processes; default is 4
    
    Returns:
        list of processed samples
    """
    all_samples = []
    
    # Get all EDF files
    edf_files = [f for f in os.listdir(folder_path) if f.endswith(".edf")]
    print(f"Found {len(edf_files)} EDF files in {folder_path}")
    
    # Extract and map all subject IDs
    for file in edf_files:
        raw_subject_id = file.split("_")[0]
        if raw_subject_id not in subject_id_map:
            subject_id_map[raw_subject_id] = len(subject_id_map)
    
    # Prepare parallel processing arguments
    process_args = []
    for file in edf_files:
        file_path = os.path.join(folder_path, file)
        raw_subject_id = file.split("_")[0]
        subject_id = subject_id_map[raw_subject_id]
        process_args.append((file_path, label, subject_id))
    
    # Process files in parallel with multiprocessing
    print(f"Processing {len(edf_files)} files with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = pool.starmap(process_file, process_args)
    
    # Collect processing results
    for samples in results:
        if samples:
            all_samples.extend(samples)
    
    print(f"Finished processing {len(edf_files)} files, got {len(all_samples)} samples")
    
    return all_samples

def preprocess_tuab(num_workers=4):
    """
    Preprocess the TUAB dataset
    
    Args:
        num_workers: number of parallel worker processes; default is 4
    
    Returns:
        EEGData object
    """
    print(f"Data root: {data_root}")
    print(f"Using {num_workers} parallel workers for processing")
    
    # Define train and evaluation data paths
    train_abnormal_path = os.path.join(data_root, 'train', 'abnormal', channel_std)
    train_normal_path = os.path.join(data_root, 'train', 'normal', channel_std)
    eval_abnormal_path = os.path.join(data_root, 'eval', 'abnormal', channel_std)
    eval_normal_path = os.path.join(data_root, 'eval', 'normal', channel_std)
    
    # Create error log file
    error_log_path = os.path.join(current_dir, "tuab-process-error-files.txt")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)
    
    # Create global subject ID mapping for dataset-wide consistency
    global_subject_id_map = {}
    
    print("Processing training data...")
    # Process training data
    train_abnormal_samples = process_files_in_folder(train_abnormal_path, 1, global_subject_id_map, num_workers)
    train_normal_samples = process_files_in_folder(train_normal_path, 0, global_subject_id_map, num_workers)
    train_samples = train_abnormal_samples + train_normal_samples
    
    print("Processing evaluation data...")
    # Process evaluation data
    eval_abnormal_samples = process_files_in_folder(eval_abnormal_path, 1, global_subject_id_map, num_workers)
    eval_normal_samples = process_files_in_folder(eval_normal_path, 0, global_subject_id_map, num_workers)
    eval_samples = eval_abnormal_samples + eval_normal_samples
    
    # Merge all samples
    all_samples = train_samples + eval_samples
    
    if not all_samples:
        print("No samples processed successfully!")
        return None
    
    # Convert to numpy arrays
    eeg_data = np.array([sample[0] for sample in all_samples])
    labels = np.array([sample[1] for sample in all_samples])
    subject_ids = np.array([sample[2] for sample in all_samples])
    
    # Print global subject ID mapping information
    print(f"Global subject ID mapping size: {len(global_subject_id_map)}")
    print(f"Total unique subjects: {len(np.unique(subject_ids))}")
    
    # Create EEGData instance
    eeg_data_obj = EEGData(
        dataset_name='TUAB',
        eeg_data=eeg_data,
        subject_ids=subject_ids,
        channel_names=clean_channel_names,
        sampling_rate=250,
        labels=labels,
        dataset_type='classification',
        split_method='10-fold',
        is_binary=True  # TUAB binary classification: normal/abnormal
    )
    
    # Save data to pkl file
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=os.path.join(project_root, 'datasets', 'data'))
    
    return eeg_data_obj

if __name__ == "__main__":
    print("=== TUAB dataset preprocessing started ===")
    
    workers = 8
    
    # Run preprocessing
    eeg_data_obj = preprocess_tuab(num_workers=workers)
    
    if eeg_data_obj:
        print("\n=== Dataset summary ===")
        print(str(eeg_data_obj))
        print("\n=== TUAB dataset preprocessing done ===")
    else:
        print("\n=== TUAB dataset preprocessing failed ===")