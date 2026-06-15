from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2015_001
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

# Import utils classes and functions
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl

def download_data_BNCI2015001():
    dataset = BNCI2015_001()
    datasetname = 'BNCI2015001'

    # use MotorImagery paradigm with preprocessing params
    paradigm = MotorImagery(fmin=0.1, fmax=75)
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    sessions = metadata["session"].values
    
    # Keep only session '0train' data
    train_mask = sessions == '0train' 
    data = data[train_mask, :, :]
    labels_string = labels_string[train_mask]
    metadata = metadata[train_mask]
    
    # Get subject IDs
    subjects = metadata["subject"].values
    
    # channel names and sampling rate
    ch_names = ['FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4']
    sampling_rate = 512
    
    # Label mapping
    label_map = {
        "right_hand": 0,
        "feet": 1,
    }
    
    # Convert labels to integers
    labels = np.array([label_map[label] for label in labels_string])
    
    # Create EEGData instance
    eeg_data_obj = EEGData(
        dataset_name=datasetname,
        eeg_data=data,
        subject_ids=subjects,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        labels=labels
    )
    
    # Get data directory path
    data_dir = os.path.abspath(os.path.join(project_root, 'datasets', 'data'))
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving data to: {data_dir}")
    # Save data to data directory
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=data_dir)
    
    return eeg_data_obj


if __name__ == "__main__":
    # Test script
    try:
        print("Starting Preprocess_Dataset test...")
        # Download data
        eeg_data_obj = download_data_BNCI2015001()
        print("Data download complete")

        # Loading data from data directory
        file_path = os.path.join(project_root, 'datasets', 'data', 'BNCI2015001.pkl')
        print(f"Loading data from file: {file_path}")
        eeg_data = load_eeg_data_from_pkl(file_path)
        
        # Return type
        print(f"\nReturn type: {type(eeg_data)}")
        
        # verify return value is EEGData
        if isinstance(eeg_data, EEGData):
            print("Validation succeeded: returned EEGData object")
            
            # Dataset summary
            print("\n=== Dataset summary ===")
            print(str(eeg_data))
            
            # Data shape
            print(f"\nData shape: {eeg_data.eeg_data.shape}")
            print(f"subject ID count: {eeg_data.subject_ids.shape[0]}")
            print(f"channel count: {len(eeg_data.channel_names)}")
            # Print channel names (one line, comma-separated)
            print(f"Channel names: {', '.join([f'{i+1}. {channel}' for i, channel in enumerate(eeg_data.channel_names)])}")
            
            # Label info
            if eeg_data.labels is not None:
                print(f"label count: {eeg_data.labels.shape[0]}")
                unique_labels, counts = np.unique(eeg_data.labels, return_counts=True)
                print("Label distribution:")
                for label, count in zip(unique_labels, counts):
                    print(f"  labels {label}: {count} samples")
        else:
            print("Validation failed: did not return EEGData object")
            
        print("\nTest complete!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()