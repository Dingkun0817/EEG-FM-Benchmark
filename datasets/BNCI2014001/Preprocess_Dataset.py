from moabb.paradigms import MotorImagery
from moabb.datasets import BNCI2014_001
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
print(f"Current dir: {current_dir}")
print(f"Project root: {project_root}")

from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl

def download_data_BNCI2014001():
    dataset = BNCI2014_001()
    datasetname = 'BNCI2014001'

    # MotorImagery paradigm and preprocessing params
    paradigm = MotorImagery(fmin=0.1, fmax=75)
    alldata = paradigm.get_data(dataset)
    data, labels_string, metadata = alldata
    sessions = metadata["session"].values
    
    # Keep only session 0train
    train_mask = sessions == '0train' 
    data = data[train_mask, :, :]
    labels_string = labels_string[train_mask]
    metadata = metadata[train_mask]

    data = data[:, :, :1000]
    
    # Subject ids
    subjects = metadata["subject"].values
    
    # Channel names and sampling rate
    ch_names = ["Fz", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1",
                 "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"]
    sampling_rate = 250
    
    # Label mapping
    label_map = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3
    }
    
    # Map labels to integers
    labels = np.array([label_map[label] for label in labels_string])
    
    # Build EEGData instance
    eeg_data_obj = EEGData(
        dataset_name=datasetname,
        eeg_data=data,
        subject_ids=subjects,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        labels=labels
    )
    
    data_dir = os.path.abspath(os.path.join(project_root, 'datasets', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving data to: {data_dir}")
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=data_dir)
    
    return eeg_data_obj


if __name__ == "__main__":
    try:
        print("Testing Preprocess_Dataset...")
        #eeg_data_obj = download_data_BNCI2014001()
        print("Data download complete")

        file_path = os.path.join(project_root, 'datasets', 'data', 'BNCI2014001.pkl')
        print(f"Loading from: {file_path}")
        eeg_data = load_eeg_data_from_pkl(file_path)
        print(f"Return type: {type(eeg_data)}")
        if isinstance(eeg_data, EEGData):
            print("OK: EEGData instance")
            print("\n=== Dataset info ===")
            print(str(eeg_data))
            print(f"\nData shape: {eeg_data.eeg_data.shape}")
            print(f"Subjects: {eeg_data.subject_ids.shape[0]}")
            print(f"Channels: {len(eeg_data.channel_names)}")
            print(f"Channel names: {', '.join(eeg_data.channel_names)}")
            if eeg_data.labels is not None:
                print(f"Labels count: {eeg_data.labels.shape[0]}")
                unique_labels, counts = np.unique(eeg_data.labels, return_counts=True)
                print("Label distribution:")
                for label, count in zip(unique_labels, counts):
                    print(f"  Label {label}: {count} samples")
        else:
            print("Failed: not an EEGData instance")
        print("\nTest done.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()