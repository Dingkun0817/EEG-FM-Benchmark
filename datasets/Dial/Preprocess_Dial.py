from moabb.datasets import Nakanishi2015
from moabb.paradigms import FilterBankSSVEP
import numpy as np
import os
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

def download_data_Dial():
    """
    Download and preprocess Dial dataset(Nakanishi2015)
    Returns: EEGData object
    """
    dataset = Nakanishi2015()
    datasetname = 'Dial'

    # use FilterBankSSVEP paradigm
    paradigm = FilterBankSSVEP(
        filters=[[1, 70]],  # Bandpass filter range (Hz)
        baseline=None,      # no baseline correction
    )
    
    print("Fetching Dial dataset...")
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    
    
    # Get subject IDs
    subjects = metadata["subject"].values
    
    # Setchannel names and sampling rate（per Nakanishi2015 dataset layout）
    # Nakanishi2015 uses 8 channels
    ch_names = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
    sampling_rate = 256  
    
    # Build label mapping from unique sorted labels
    unique_labels = sorted(np.unique(labels_string))
    print(f"unique labels: {unique_labels}")
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Label mapping: {label_map}")
    
    # Convert labels to integers
    labels = np.array([label_map[label] for label in labels_string])
    
    # Create EEGData instance
    eeg_data_obj = EEGData(
        dataset_name=datasetname,
        eeg_data=data,
        subject_ids=subjects,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        labels=labels,
        dataset_type='classification',
        split_method='LOSO'
    )
    
    # Get data directory and save data
    data_dir = os.path.abspath(os.path.join(project_root, 'datasets', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=data_dir)
    
    print(f"Dial dataset preprocessed and saved to: {data_dir}")
    return eeg_data_obj


if __name__ == "__main__":
    """
    Test script for Preprocess_Dial.py
    """
    try:
        print("="*50)
        print("Starting Preprocess_Dial test...")
        print("="*50)
        
        # download and preprocess data
        eeg_data_obj = download_data_Dial()
        
        # Loading data from file
        file_path = os.path.join(project_root, 'datasets', 'data', 'Dial.pkl')
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