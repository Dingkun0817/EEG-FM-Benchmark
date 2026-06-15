import numpy as np
import os
import sys
import pickle
from typing import Optional, List, Union, Dict, Tuple

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


def load_things2_split_data(
    subjects: Optional[Union[int, List[int]]] = None,
    split: str = 'training',
    dnn_model: str = 'clip',
    eeg_data_path: Optional[str] = None,
    img_data_path: Optional[str] = None
) -> Tuple[EEGData, List[str]]:
    """
    Load ThingsEEG2 split (train or test) for all subjects
    
    Args:
    subjects: int or list, optional
        subject IDs to load; if None, auto-detect subjects
    split: str
        split type: 'train' or 'test'
    dnn_model: str
        DNN model name，default 'vgg16'
    eeg_data_path: str, optional
        EEG data root path
    img_data_path: str, optional
        image feature root path
    
    Returns:
    Tuple[EEGData, List[str]]: (EEGData object, processed subject list)
    """
    if split not in ['training']:#, 'test'
        raise ValueError("split must be 'train' or 'test'")
    
    datasetname = 'Things2'
    
    # Set data file paths (defaults)
    if eeg_data_path is None:
        eeg_data_path = os.environ.get(
            'THINGSEEG2_EEG_DIR',
            os.path.join(project_root, 'datasets', 'data', 'raw', 'ThingsEEG2', 'Preprocessed_data_1000Hz'),
        )
    if img_data_path is None:
        img_data_path = os.environ.get(
            'THINGSEEG2_IMG_DIR',
            os.path.join(project_root, 'datasets', 'data', 'raw', 'ThingsEEG2', 'DNN_feature_maps', 'clip', 'pretrained-True'),
        )
    
    print(f"=== Load{split.upper()}data ===")
    print(f"EEGdatapath: {eeg_data_path}")
    print(f"image featurespath: {img_data_path}")
    print(f"DNN model: {dnn_model}")
    
    '''
    subjects = []
    for i in range(1, 11):  # assume up to 50 subjects
        sub_dir = os.path.join(eeg_data_path, f'sub-{i:02d}')
        eeg_file = os.path.join(sub_dir, f'preprocessed_eeg_{split}.npy')
        if os.path.exists(eeg_file):
            subjects.append(i)
    '''
    
    all_eeg_data = []
    all_labels = []
    all_subject_ids = []
    all_img_features = []
    processed_subjects = []
    
    # load global image features shared by subjects
    img_file = f'{dnn_model}_feature_maps_{split}.npy'
    img_path = os.path.join(img_data_path, img_file)
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"image featuresfile does not exist: {img_path}")
    
    print(f"load global image features: {img_path}")
    global_img_features = np.load(img_path, allow_pickle=True)
    global_img_features = np.squeeze(global_img_features)
    print(f"global image feature shape: {global_img_features.shape}")
    
    # iterate subjects
    for sub_id in subjects:
        print(f"\nprocessing subject {sub_id}...")
        
        try:
            # build EEG file path
            eeg_file = f'preprocessed_eeg_{split}.npy'
            eeg_path = os.path.join(eeg_data_path, f'sub-{sub_id:02d}', eeg_file)
            
            # Checkfilewhetherexist
            if not os.path.exists(eeg_path):
                print(f"  warning: EEG file does not exist: {eeg_path}")
                continue
            
            print(f"  Loading EEG: {eeg_path}")
            
            # LoadEEGdata
            eeg_data_dict = np.load(eeg_path, allow_pickle=True)
            eeg_data = eeg_data_dict['preprocessed_eeg_data']
            eeg_data = np.mean(eeg_data, axis=1)  # shape: (n_samples, n_timepoints)
            
            #eeg_data = np.expand_dims(eeg_data, axis=1)  # shape: (n_samples, 1, n_timepoints)
            
            n_eeg_samples = eeg_data.shape[0]
            
            # check sample count matches
            if n_eeg_samples > global_img_features.shape[0]:
                print(f"  warning: EEG sample count({n_eeg_samples})exceeds image feature count({global_img_features.shape[0]})，truncate")
                eeg_data = eeg_data[:global_img_features.shape[0]]
                n_samples = global_img_features.shape[0]
            else:
                n_samples = n_eeg_samples
            
            # use matching image features
            img_features = global_img_features[:n_samples]  # use first n_samples of global image features
            
            # build labels (sample index，note: sample indices are per-subject）
            labels = np.arange(n_samples)
            
            # subject ID array for current batch
            subject_ids = np.full(n_samples, sub_id)
            
            # Savedata
            all_eeg_data.append(eeg_data)
            all_labels.append(labels)
            all_subject_ids.append(subject_ids)
            all_img_features.append(img_features)
            
            processed_subjects.append(sub_id)
            
            print(f"  loaded successfully: EEGshape={eeg_data.shape}, sample count={n_samples}")
            
        except Exception as e:
            print(f"  processing subject {sub_id} error: {str(e)}")
            continue
    
    # check any data loaded
    if not all_eeg_data:
        raise ValueError(f"failed to load any subject for{split}data")
    
    # merge all subjects
    combined_eeg_data = np.vstack(all_eeg_data)
    combined_labels = np.concatenate(all_labels)
    combined_subject_ids = np.concatenate(all_subject_ids)
    combined_img_features = np.vstack(all_img_features)
    
    print(f"\n=== {split.upper()}data summary ===")
    print(f"successprocessing subject: {processed_subjects}")
    print(f"total EEG samples: {combined_eeg_data.shape[0]}")
    print(f"EEGData shape: {combined_eeg_data.shape}")
    print(f"labels shape: {combined_labels.shape}")
    print(f"subject ID shape: {combined_subject_ids.shape}")
    print(f"image featuresshape: {combined_img_features.shape}")
    print(f"unique subjects: {np.unique(combined_subject_ids)}")
    print(f"subject count: {len(np.unique(combined_subject_ids))}")
    
    ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']
    
    # sampling rate（set per your setup）
    sampling_rate = 1000  # Hz
    
    # Create EEGData instance
    if len(processed_subjects) == 1:
        sub_str = f'_sub{processed_subjects[0]:02d}'
    else:
        sub_str = f'_{len(processed_subjects)}subjects'
    
    dataset_name = f'Things2_{split}{sub_str}_{dnn_model}'
    
    eeg_data_obj = EEGData(
        dataset_name=dataset_name,
        eeg_data=combined_eeg_data,
        subject_ids=combined_subject_ids,
        channel_names=ch_names,
        sampling_rate=sampling_rate,
        labels=combined_labels,
        dataset_type='classification',  # adjust for your task
        img_feature=combined_img_features,
        is_binary=True
    )
    
    return eeg_data_obj, processed_subjects


def save_things2_all_subjects(
    subjects: Optional[Union[int, List[int]]] = None,
    dnn_model: str = 'clip',
    output_dir: Optional[str] = None,
    eeg_data_path: Optional[str] = None,
    img_data_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Load all ThingsEEG2 subjects and save train/test pkls
    
    Args:
    subjects: int or list, optional
        subject IDs to load
    dnn_model: str
        DNN model name
    output_dir: str, optional
        output directory; None uses default
    eeg_data_path: str, optional
        EEG data root path
    img_data_path: str, optional
        image feature root path
    
    Returns:
    Dict[str, str]: saved file path dict
    """
    # Setoutputdirectory
    if output_dir is None:
        output_dir = os.path.join(project_root, 'datasets', 'data')
    
    # ensureoutputdirectoryexist
    os.makedirs(output_dir, exist_ok=True)
    
    # load train and test data separately
    train_data = None
    test_data = None
    train_subjects = []
    test_subjects = []
    
    file_paths = {}
    
    # 1. load training data
    print("=" * 60)
    print("Loading training data...")
    print("=" * 60)
    
    try:
        train_data, train_subjects = load_things2_split_data(
            subjects=subjects,
            split='training',
            dnn_model=dnn_model,
            eeg_data_path=eeg_data_path,
            img_data_path=img_data_path
        )
        
        # training output filename
        if subjects is None:
            sub_str = 'all'
        elif isinstance(subjects, int):
            sub_str = f'sub{subjects:02d}'
        else:
            if len(subjects) == 1:
                sub_str = f'sub{subjects[0]:02d}'
            else:
                sub_str = f'all_{len(train_subjects)}subjects'
        
        train_filename = f'Things2_train_{dnn_model}_{sub_str}.pkl'
        train_filepath = os.path.join(output_dir, train_filename)
        
        # Savetraindata
        with open(train_filepath, 'wb') as f:
            pickle.dump(train_data, f)
        
        file_paths['training'] = train_filepath
        print(f"training data saved to: {train_filepath}")
        print(f"training set info:")
        print(str(train_data))
        
    except Exception as e:
        print(f"error loading training data: {str(e)}")
    
    # 2. load test data
    print("\n" + "=" * 60)
    print("Loadingtestdata...")
    print("=" * 60)
    
    try:
        test_data, test_subjects = load_things2_split_data(
            subjects=subjects,
            split='test',
            dnn_model=dnn_model,
            eeg_data_path=eeg_data_path,
            img_data_path=img_data_path
        )
        
        # test output filename
        if subjects is None:
            sub_str = 'all'
        elif isinstance(subjects, int):
            sub_str = f'sub{subjects:02d}'
        else:
            if len(subjects) == 1:
                sub_str = f'sub{subjects[0]:02d}'
            else:
                sub_str = f'all_{len(test_subjects)}subjects'
        
        test_filename = f'Things2_test_{dnn_model}_{sub_str}.pkl'
        test_filepath = os.path.join(output_dir, test_filename)
        
        # Savetestdata
        with open(test_filepath, 'wb') as f:
            pickle.dump(test_data, f)
        
        file_paths['test'] = test_filepath
        print(f"test data saved to: {test_filepath}")
        print(f"test set info:")
        print(str(test_data))
        
    except Exception as e:
        print(f"error loading test data: {str(e)}")
    
    # 3. summary
    print("\n" + "=" * 60)
    print("load/save summary:")
    print("=" * 60)
    
    if train_data is not None:
        print(f"traindata:")
        print(f"  - subject count: {len(train_subjects)}")
        print(f"  - subject list: {train_subjects}")
        print(f"  - total samples: {train_data.get_sample_count()}")
        print(f"  - save path: {file_paths.get('training', 'not saved')}")
    
    if test_data is not None:
        print(f"testdata:")
        print(f"  - subject count: {len(test_subjects)}")
        print(f"  - subject list: {test_subjects}")
        print(f"  - total samples: {test_data.get_sample_count()}")
        print(f"  - save path: {file_paths.get('test', 'not saved')}")
    
    return file_paths


def verify_saved_data(file_paths: Dict[str, str]):
    """
    validate saved pkl files
    
    Args:
    file_paths: Dict[str, str], saved file path dict
    """
    print("\n" + "=" * 60)
    print("validating saved data...")
    print("=" * 60)
    
    for split, filepath in file_paths.items():
        if filepath and os.path.exists(filepath):
            print(f"\nvalidate {split} data: {os.path.basename(filepath)}")
            try:
                # Loaddata
                loaded_data = load_eeg_data_from_pkl(filepath)
                
                # validate data
                if isinstance(loaded_data, EEGData):
                    print(f"  Validation succeeded: EEGData object")
                    print(f"    datasetname: {loaded_data.dataset_name}")
                    print(f"    sample count: {loaded_data.get_sample_count()}")
                    print(f"    channel count: {loaded_data.get_channel_count()}")
                    print(f"    time points: {loaded_data.get_time_point_count()}")
                    print(f"    sampling rate: {loaded_data.sampling_rate} Hz")
                    print(f"    subject count: {loaded_data.get_subject_unique_count()}")
                    print(f"    unique subjectsID: {np.unique(loaded_data.subject_ids)}")
                    
                    if loaded_data.labels is not None:
                        print(f"    label range: {loaded_data.labels.min()} - {loaded_data.labels.max()}")
                    
                    if loaded_data.img_feature is not None:
                        print(f"    image featuresshape: {loaded_data.img_feature.shape}")
                else:
                    print(f"  Validation failed: not an EEGData object")
                    
            except Exception as e:
                print(f"  ✗ error loading file: {str(e)}")
        else:
            print(f"\n{split.upper()}file does not exist: {filepath}")
    
    print("\nvalidation complete!")


if __name__ == "__main__":
    # Test script
    print("Starting ThingsEEG2 load and save test...")
    eeg_data_path = os.environ.get('THINGSEEG2_EEG_DIR', '')
    img_data_path = os.environ.get('THINGSEEG2_IMG_DIR', '')
    output_dir = os.path.join(project_root, 'datasets', 'data')
    
    dnn_model = 'clip'
    
    # test1: load and save all subjects
    print("\n" + "=" * 60)
    print("test1: load and save all subjects")
    print("=" * 60)
    subjects=[1,2,3,4,5]#,,6,7,8,9,10
    try:
        # load and save data
        
        saved_files = save_things2_all_subjects(
            subjects=subjects,
            dnn_model=dnn_model,
            output_dir=output_dir,
            eeg_data_path=eeg_data_path,
            img_data_path=img_data_path
        )
        
        # validate saved data
        verify_saved_data(saved_files)
        
    except Exception as e:
        print(f"test1fail: {str(e)}")
        import traceback
        traceback.print_exc()
