import os
import mne
import numpy as np
from scipy.signal import iirnotch, filtfilt
import sys

# Directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# Add project root to Python path
sys.path.append(project_root)
print(f"current directory: {current_dir}")
print(f"project root directory: {project_root}")

# Import EEGData helpers
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl



def preprocess_eegmat_data(data_root):
    """
    Preprocess EEGMAT dataset
    
    Args:
        data_root: raw data root directory
        
    Returns:
        EEGData object
    """
    datasetname = 'EEGMAT'
    
    # Configure paths
    raw_data_path = data_root or os.environ.get(
        'EEGMAT_RAW_DIR',
        os.path.join(project_root, 'datasets', 'data', 'raw', 'EEGMAT'),
    )
    processed_data_path = os.path.join(project_root, 'datasets', 'data', 'raw', 'EEGMAT', 'processed_data')
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Configure parameters
    retain_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 
                      'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']
    segment_duration = 4  # segment duration in seconds
    
    # store all data, labels, and subject IDs
    all_eeg_segments = []
    all_labels = []
    all_subjects = []
    
    print("Starting EEGMAT preprocessing...")
    print(f"raw data path: {raw_data_path}")
    print(f"processed data path: {processed_data_path}")
    
    # process 36 subjects, two conditions each
    for subject_id in range(36):
        for file_type, label in [("_1", 0), ("_2", 1)]:
            file_name = f"Subject{subject_id:02d}{file_type}.edf"
            file_path = os.path.join(raw_data_path, file_name)
            
            if os.path.exists(file_path):
                print(f"  Processing file: {file_name}")
                
                # Loadrawdata
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                
                # rename channels (strip EEG prefix)
                channel_mapping = {ch: ch.replace('EEG ', '') for ch in raw.ch_names}
                raw.rename_channels(channel_mapping)
                
                # pick retained channels
                raw.pick_channels(retain_channels)
                
               
                
                # extract target signal window
                sfreq = raw.info['sfreq']
                if file_type == "_1":  # last 60 seconds
                    start_idx = int((raw.n_times / sfreq - 60) * sfreq)
                    data = raw.get_data(start=start_idx)
                elif file_type == "_2":  # first 60 seconds
                    data = raw.get_data(stop=int(60 * sfreq))
                
                # split data into 4-second segments
                n_points = int(segment_duration * sfreq)  # samples per 4-second segment
                segments = [data[:, i * n_points:(i + 1) * n_points] for i in range(data.shape[1] // n_points)]
                
                # append to lists
                all_eeg_segments.extend(segments)
                all_labels.extend([label] * len(segments))
                all_subjects.extend([subject_id] * len(segments))
            else:
                print(f"  file does not exist: {file_name}")
    
    # stack lists into numpy arrays
    eeg_data = np.array(all_eeg_segments)
    labels = np.array(all_labels)
    subjects = np.array(all_subjects)
    
    # sampling rate from last processed raw object
    sampling_rate = raw.info['sfreq'] if 'raw' in locals() else 128  # default sampling rate
    
    print(f"\nPreprocessing done:")
    print(f"  total samples: {eeg_data.shape[0]}")
    
    if eeg_data.shape[0] > 0:
        print(f"  channels: {eeg_data.shape[1]}")
        print(f"  time points per sample: {eeg_data.shape[2]}")
        print(f"  sampling rate: {sampling_rate} Hz")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"  subject count: {len(np.unique(subjects))}")
    else:
        print("  warning: no data files found; lists are empty")
        print("  ensure raw EEGMAT files exist at:")
        print(f"  {raw_data_path}")
        print("  expected filenames: Subject00_1.edf, Subject00_2.edf, ..., Subject35_2.edf")
    
    # create EEGData only when samples exist
    if eeg_data.shape[0] > 0:
        # Create EEGData instance
        eeg_data_obj = EEGData(
            dataset_name=datasetname,
            eeg_data=eeg_data,
            subject_ids=subjects,
            channel_names=retain_channels,
            sampling_rate=sampling_rate,
            labels=labels,
            dataset_type='classification',
            split_method='LOSO'
        )
        
        # Get data directory and save data
        data_dir = os.path.abspath(os.path.join(project_root, 'datasets', 'data'))
        os.makedirs(data_dir, exist_ok=True)
        save_eeg_data_to_pkl(eeg_data_obj, output_dir=data_dir)
        
        print(f"\nEEGMAT dataset saved to: {data_dir}")
        return eeg_data_obj
    else:
        print("\nerror: no data found; cannot create EEGData instance")
        print("ensure raw EEGMAT files exist with the expected naming format")
        return None

if __name__ == "__main__":
    """Test script for Preprocess_EEGMAT.py."""
    try:
        print("="*50)
        print("Starting Preprocess_EEGMAT test...")
        print("="*50)
        
        data_root = os.environ.get(
            'EEGMAT_RAW_DIR',
            os.path.join(project_root, 'datasets', 'data', 'raw', 'EEGMAT'),
        )
        
        # preprocessdata
        eeg_data_obj = preprocess_eegmat_data(data_root)
        
        # Loading data from file
        file_path = os.path.join(project_root, 'datasets', 'data', 'EEGMAT.pkl')
        print(f"\nLoading data from file: {file_path}")
        eeg_data = load_eeg_data_from_pkl(file_path)
        
        # verify return value is EEGData
        if isinstance(eeg_data, EEGData):
            print("✓ Validation succeeded: returned EEGData object")
            
            # Dataset summary
            print("\n=== Dataset summary ===")
            print(str(eeg_data))
            
            # Data shape
            print(f"\n=== Data details ===")
            print(f"Data shape: {eeg_data.eeg_data.shape}")
            print(f"subject count: {len(np.unique(eeg_data.subject_ids))}")
            print(f"total samples: {eeg_data.eeg_data.shape[0]}")
            print(f"channel count: {len(eeg_data.channel_names)}")
            print(f"time point count: {eeg_data.eeg_data.shape[2]}")
            
            # Printchannel names
            print(f"\nChannel names: {', '.join(eeg_data.channel_names)}")
            
            # Label info
            if eeg_data.labels is not None:
                print(f"\n=== Label info ===")
                print(f"label count: {eeg_data.labels.shape[0]}")
                unique_labels, counts = np.unique(eeg_data.labels, return_counts=True)
                print("Label distribution:")
                for label, count in zip(unique_labels, counts):
                    print(f"  labels {label}: {count} samples ({count/eeg_data.labels.shape[0]:.2%})")
        else:
            print("✗ Validation failed: did not return EEGData object")
            
        print("\n" + "="*50)
        print("Test complete!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()