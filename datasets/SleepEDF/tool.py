import os
import numpy as np
import mne
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def _process_single_file(args):
    """
    process one PSG file and hypnogram
    
    Args:
    args: tuple, arguments:
        data_dir: str, datadirectorypath
        psg_file: str, PSG filename
        include_eeg_only: bool, whether to include EEG channels only
    
    Returns:
    tuple: (subject_id, result_dict) or None if processing fails
    """
    data_dir, psg_file, include_eeg_only = args
    
    try:
        # extract subject and record IDs
        file_prefix = psg_file.split('-')[0]
        subject_id = int(file_prefix[3:5])  # e.g. subject ID from SC4001E0 prefix
        base_prefix = file_prefix[:6]  # e.g. base prefix from SC4001E0
        
        # build PSG and hypnogram paths
        psg_path = os.path.join(data_dir, psg_file)
        
        # find matching hypnogram file
        hyp_file = None
        for hyp_candidate in os.listdir(data_dir):
            if hyp_candidate.startswith(base_prefix) and 'Hypnogram' in hyp_candidate:
                hyp_file = hyp_candidate
                break
        
        if hyp_file is None:
            print(f"Warning: No hypnogram file found for {psg_file}")
            return None
        
        hyp_path = os.path.join(data_dir, hyp_file)
        
        # LoadPSGdata
        raw = mne.io.read_raw_edf(psg_path, preload=True)
        
        # LoadHypnogramdata
        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations)
        
        # select EEG channels (Fpz-Cz, Pz-Oz)
        if include_eeg_only:
            eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
            raw = raw.pick_channels(eeg_channels)
        
        # extract data and labels
        data, sfreq = raw.get_data(return_times=False), raw.info['sfreq']
        
        # build 30-second epoch labels
        epoch_duration = 30.0  # 30 seconds per epoch
        n_samples_per_epoch = int(epoch_duration * sfreq)
        n_epochs = data.shape[1] // n_samples_per_epoch
        
        # init labels to -1 for unmarked
        labels = np.full(n_epochs, -1, dtype=int)
        
        # map sleep stages to numeric labels
        stage_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,  # merge stages 3 and 4 into label 3
            'Sleep stage R': 4
        }
        
        # map each annotation segment
        for annot in annotations:
            start_idx = int(annot['onset'] * sfreq) // n_samples_per_epoch
            end_idx = int((annot['onset'] + annot['duration']) * sfreq) // n_samples_per_epoch
            end_idx = min(end_idx, n_epochs)  # clamp to epoch count
            
            if annot['description'] in stage_map:
                labels[start_idx:end_idx] = stage_map[annot['description']]
        
        # reshape data to (epochs, channels, time)
        data_reshaped = data[:, :n_epochs * n_samples_per_epoch]
        data_reshaped = data_reshaped.reshape(data.shape[0], n_epochs, n_samples_per_epoch)
        data_reshaped = data_reshaped.transpose(1, 0, 2)  # (epochs, channels, time)
        
        # drop unlabeled epochs (-1 means unlabeled)
        valid_epochs = labels != -1
        data_reshaped = data_reshaped[valid_epochs]
        labels = labels[valid_epochs]
        
        # strip 'EEG ' prefix from channel names
        channels = [ch.replace('EEG ', '') for ch in raw.ch_names]
        
        # return processed result
        return subject_id, {
            'data': data_reshaped,
            'labels': labels,
            'channels': channels,
            'sfreq': sfreq,
            'file': psg_file
        }
        
    except Exception as e:
        print(f"Error processing {psg_file}: {e}")
        return None

def load_sleepedf_data(data_dir, subject_ids=None, include_eeg_only=True, n_jobs=None):
    """
    Load Sleep-EDF dataset (multiprocessing version)
    
    Args:
    data_dir: str, Sleep-EDFdatadirectorypath
    subject_ids: list or None, subject IDs to load; None loads all
    include_eeg_only: bool, whether to include EEG channels only
    n_jobs: int or None, worker count; None uses all CPUs
    
    Returns:
    data_dict: dict, dict of subject data and labels
    """
    # GetallPSGfile
    psg_files = [f for f in os.listdir(data_dir) if f.endswith('-PSG.edf')]
    
    # filter files by subject_ids if given
    if subject_ids is not None:
        psg_files = [f for f in psg_files if any(f.startswith(f'SC{sid:04d}') for sid in subject_ids)]
    
    data_dict = {}
    
    # set worker count
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # prepare task args
    tasks = [(data_dir, psg_file, include_eeg_only) for psg_file in psg_files]
    
    # process files in parallel
    with Pool(n_jobs) as pool:
        # show progress with tqdm
        results = list(tqdm(pool.imap(_process_single_file, tasks), 
                           total=len(tasks), desc='Loading Sleep-EDF data (multiprocessing)'))
    
    # process results
    for result in results:
        if result is not None:
            subject_id, result_dict = result
            if subject_id not in data_dict:
                data_dict[subject_id] = []
            data_dict[subject_id].append(result_dict)
    
    return data_dict

def combine_subject_data(data_dict):
    """
    merge all subjects into arrays
    
    Args:
    data_dict: dict, dict of subject data and labels
    
    Returns:
    X: ndarray, merged EEG data with shape (n_samples, n_channels, n_times)
    y: ndarray, merged labels with shape (n_samples,)
    subject_ids: ndarray, remapped subject IDs (0,1,2,...) with shape (n_samples,)
    """
    all_data = []
    all_labels = []
    all_subject_ids = []
    
    # unique sorted subject IDs
    unique_subject_ids = sorted(data_dict.keys())
    
    # subject ID remap: raw ID -> new ID (0,1,2,...)
    subject_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_subject_ids)}
    
    for subject_id, sessions in data_dict.items():
        new_subject_id = subject_id_map[subject_id]
        for session in sessions:
            n_samples = session['data'].shape[0]
            all_data.append(session['data'])
            all_labels.append(session['labels'])
            all_subject_ids.append(np.full(n_samples, new_subject_id))
    
    if not all_data:
        return np.array([]), np.array([]), np.array([])
    
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_subject_ids)
    
    return X, y, subject_ids

def get_sleepedf_channel_names(include_eeg_only=True):
    """
    Sleep-EDF channel names
    
    Args:
    include_eeg_only: bool, whether EEG channel names only
    
    Returns:
    channel_names: list, Channel names
    """
    if include_eeg_only:
        return ['Fpz-Cz', 'Pz-Oz']
    else:
        return ['Fpz-Cz', 'Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 
                'EMG submental', 'Temp rectal', 'Event marker']