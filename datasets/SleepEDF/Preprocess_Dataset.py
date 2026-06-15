import numpy as np
import os
import sys
import pandas as pd

# Directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# Add project root to Python path
sys.path.append(project_root)
print(f"current directory: {current_dir}")
print(f"project root directory: {project_root}")

# Import utility functions and EEGData
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl
import importlib.util

_tool_path = os.path.join(current_dir, 'tool.py')
_spec = importlib.util.spec_from_file_location('sleepedf_tool', _tool_path)
sleepedf_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sleepedf_tool)
load_sleepedf_data = sleepedf_tool.load_sleepedf_data
combine_subject_data = sleepedf_tool.combine_subject_data
get_sleepedf_channel_names = sleepedf_tool.get_sleepedf_channel_names

def process_sleepedf_dataset(data_dir=None, subject_ids=None, include_eeg_only=True, output_dir=None, n_jobs=None):
    """
    Process Sleep-EDF dataset and convert to EEGData
    
    Args:
    data_dir: str, Sleep-EDF data directory; None uses default
    subject_ids: list or None, subject IDs to load; None loads all
    include_eeg_only: bool, whether to include EEG channels only
    output_dir: str, output directory; None uses default
    n_jobs: int or None, worker count; None uses all CPUs
    
    Returns:
    EEGData object
    """
    dataset_name = 'SleepEDF'
    
    # Setdefaultdatadirectory
    if data_dir is None:
        data_dir = os.environ.get(
            'SLEEPEDF_RAW_DIR',
            os.path.join(project_root, 'datasets', 'data', 'raw', 'sleep-cassette'),
        )
    
    print(f"Loading data from directory: {data_dir}")
    
    # LoadSleep-EDFdata
    data_dict = load_sleepedf_data(data_dir, subject_ids, include_eeg_only, n_jobs=n_jobs)
    
    if not data_dict:
        print("warning: no valid data loaded")
        return None
    
    # merge all subjects
    X, y, subject_ids_array = combine_subject_data(data_dict)
    
    if X.size == 0:
        print("warning: merged data is empty")
        return None
    
    # Getchannel names
    channel_names = get_sleepedf_channel_names(include_eeg_only)
    
    # sampling rate is 100 Hz for Sleep-EDF
    sampling_rate = 100
    
    print(f"rawData shape: {X.shape}")
    print(f"rawlabels shape: {y.shape}")
    print(f"subject ID shape: {subject_ids_array.shape}")
    print(f"channel count: {len(channel_names)}")
    print(f"channel names: {channel_names}")
    print(f"sampling rate: {sampling_rate} Hz")
    
    # unique subjects and label distribution
    unique_subjects = np.unique(subject_ids_array)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    
    print(f"unique subject count: {len(unique_subjects)}")
    print(f"subject ID list: {unique_subjects.tolist()}")
    print(f"unique label count: {len(unique_labels)}")
    print("Label distribution:")
    for label, count in zip(unique_labels, label_counts):
        stage_name = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3/4', 'REM'][label]
        print(f"  {stage_name} (labels {label}): {count} samples")
    
    # Create EEGData instance
    eeg_data_obj = EEGData(
        dataset_name=dataset_name,
        eeg_data=X,
        subject_ids=subject_ids_array,
        channel_names=channel_names,
        sampling_rate=sampling_rate,
        labels=y,
        dataset_type='classification',
        split_method='10-fold'
    )
    
    # Setoutputdirectory
    if output_dir is None:
        output_dir = os.path.join(project_root, 'datasets/data')
    
    # ensureoutputdirectoryexist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving data to: {output_dir}")
    
    
    # Save data to data directory
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=output_dir)
    
    return eeg_data_obj



if __name__ == "__main__":
    # Test script
    try:
        print("Starting SleepEDF dataset processing test...")
        
        # optional subject IDs to load(e.g. [4001, 4002] or None for all subjects)
        selected_subjects = None  # Loadallsubject
        # selected_subjects = [4001, 4002]  # load specific subjects for test only
        
        # LoadSleepEDFdata，use multiprocessing
        eeg_data_obj = process_sleepedf_dataset(subject_ids=selected_subjects, n_jobs=None)
        print("SleepEDFdataLoad complete")
        
        
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()