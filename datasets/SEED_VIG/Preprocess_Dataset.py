import os
import sys
import numpy as np
from scipy.io import loadmat

# Directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# Add project root to Python path
sys.path.append(project_root)

# Import utils classes and functions
from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl

def download_data_SEED_VIG():
    """
    preprocessSEED-VIGdataset
    Returns: 
        EEGData object，preprocessed EEG data and regression targets
    """
    datasetname = 'SEED_VIG'
    
    raw_data_path = os.environ.get(
        'SEED_VIG_RAW_DIR',
        os.path.join(project_root, 'datasets', 'data', 'raw', 'SEED_VIG', 'raw_data'),
    )
    labels_path = os.environ.get(
        'SEED_VIG_LABELS_DIR',
        os.path.join(project_root, 'datasets', 'data', 'raw', 'SEED_VIG', 'perclos_labels'),
    )
    
    # Checkpathwhetherexist
    for path in [raw_data_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"datapathdoes not exist: {path}")
    
    # lists for EEG data and regression targets
    all_eeg_data = []
    all_regression_values = []
    all_subject_ids = []
    all_channel_names = None  # read from first file
    final_sampling_rate = None  # read from first file
    
    # label filenames used to pair with raw files
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.mat')]
    
    # iterate label files
    for idx, file_name in enumerate(label_files):
        print(f"Processing file {idx+1}/{len(label_files)}")
        
        # build matching raw data path
        raw_file_path = os.path.join(raw_data_path, file_name)
        label_file_path = os.path.join(labels_path, file_name)
        
        # Readlabelsdata（PERCLOSvalue）
        label_data = loadmat(label_file_path)
        perclos_values = label_data['perclos']

        # ReadrawEEGdata
        raw_data = loadmat(raw_file_path)

        
        # get EEG structure and data
        eeg_struct = raw_data['EEG'][0, 0]
        data = eeg_struct['data']
        eeg_data = data.reshape(885, 1600, 17).transpose(0, 2, 1)
        
        chn_data = eeg_struct['chn']
        # process known data layout
        flat_channels = []
        for item in chn_data.flatten():
            for sub_item in item.flatten():
                flat_channels.extend(sub_item.flatten())
        all_channel_names = [str(ch).strip().strip('"') for ch in flat_channels]

        final_sampling_rate = 200

        

        
        # extractsubjectID
        parts = file_name.split('_')
        subject_id = int(parts[0]) 
        subject_ids = [subject_id] * eeg_data.shape[0]
        
        # append to lists
        all_eeg_data.append(eeg_data)
        all_regression_values.append(perclos_values)
        all_subject_ids.extend(subject_ids)
    
    # check any data loaded
    if not all_eeg_data:
        raise ValueError("no data loaded successfully")
    
    # Mergealldata
    combined_eeg_data = np.concatenate(all_eeg_data, axis=0)
    combined_regression_values = np.concatenate(all_regression_values, axis=0)
    combined_subject_ids = np.array(all_subject_ids)
    
  
    
    # Create EEGData instance
    eeg_data_obj = EEGData(
        dataset_name=datasetname,
        eeg_data=combined_eeg_data,
        subject_ids=combined_subject_ids,
        channel_names=all_channel_names,
        sampling_rate=final_sampling_rate,
        dataset_type='regression',
        regression_values=combined_regression_values
    )
    
    # Savedata
    data_dir = os.path.abspath(os.path.join(project_root, 'datasets', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=data_dir)
    
    return eeg_data_obj


if __name__ == "__main__":
    # run preprocess and test
    eeg_data_obj = download_data_SEED_VIG()
    print("\n===== SEED-VIG preprocessing test info =====")
    print(f"datasetname: {eeg_data_obj.dataset_name}")
    print(f"datasettype: {eeg_data_obj.dataset_type}")
    print(f"samplescount: {eeg_data_obj.get_sample_count()}")
    print(f"channel count: {eeg_data_obj.get_channel_count()}")
    print(f"time points per sample: {eeg_data_obj.eeg_data.shape[2]}")
    print(f"Data shape: {eeg_data_obj.eeg_data.shape}")
    print(f"sampling rate: {eeg_data_obj.sampling_rate} Hz")
    print(f"\nChannel names ({len(eeg_data_obj.channel_names)}):")
    print(eeg_data_obj.channel_names)
    
    if hasattr(eeg_data_obj, 'regression_values'):
        print(f"\nRegression target statistics:")
        print(f"  min: {np.min(eeg_data_obj.regression_values):.4f}")
        print(f"  max: {np.max(eeg_data_obj.regression_values):.4f}")
        print(f"  mean: {np.mean(eeg_data_obj.regression_values):.4f}")
        print(f"  std: {np.std(eeg_data_obj.regression_values):.4f}")
        print(f"  shape: {eeg_data_obj.regression_values.shape}")
        print(f"  subject id count: {len(eeg_data_obj.subject_ids)}")
    
    unique_subjects = np.unique(eeg_data_obj.subject_ids)
    print(f"\nSubject info:")
    print(f"  subject count: {len(unique_subjects)}")
    print(f"  subject ID list: {list(unique_subjects)}")
    
    print(f"\nData quality checks:")
    print(f"  contains NaN: {np.isnan(eeg_data_obj.eeg_data).any()}")
    print(f"  contains inf: {np.isinf(eeg_data_obj.eeg_data).any()}")
    print(f"  per-sample mean range: [{np.min(np.mean(eeg_data_obj.eeg_data, axis=(1,2))):.4f}, {np.max(np.mean(eeg_data_obj.eeg_data, axis=(1,2))):.4f}]")
    
    print("\n===== test output complete =====")