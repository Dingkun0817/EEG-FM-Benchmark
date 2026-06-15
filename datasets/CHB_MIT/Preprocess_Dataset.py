import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pyedflib

# Directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root to path
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# Add project root to Python path
sys.path.append(project_root)
print(f"current directory: {current_dir}")
print(f"project root directory: {project_root}")

# Import EEGData and save helpers from utils (pkl with usage, no csv)
try:
    from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl
except ImportError:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from utils.EEGDataLoader import EEGData, save_eeg_data_to_pkl, load_eeg_data_from_pkl

# ---------------- CONFIGURATION ----------------
pathDataSet = os.environ.get(
    'CHB_MIT_RAW_DIR',
    os.path.join(project_root, 'datasets', 'data', 'raw', 'CHB-MIT'),
)
output_dir = os.environ.get('CHB_MIT_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), '../../datasets/data'))

patients = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23"]

TARGET_CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                   'FZ-CZ', 'CZ-PZ']

sampleRate = 256
WINDOW_SECONDS = 4
INTERICTAL_DURATION = 4 * 150  # 10 minutes

# ---------------- DATA CLASSES ----------------

class FileData:
    def __init__(self, s, e, nF):
        self.start = s
        self.end = e
        self.nameFile = nF

class SeizureData:
    def __init__(self, s, e):
        self.start = s
        self.end = e

# ---------------- HELPER FUNCTIONS ----------------

def getTime(dateInString):
    """Parse time string, handle day rollover"""
    try:
        time_obj = datetime.strptime(dateInString, '%H:%M:%S')
    except ValueError:
        dateInString = " " + dateInString
        if ' 24' in dateInString:
            dateInString = dateInString.replace(' 24', '23')
            time_obj = datetime.strptime(dateInString, '%H:%M:%S') + timedelta(hours=1)
        elif ' 25' in dateInString:
            dateInString = dateInString.replace(' 25', '23')
            time_obj = datetime.strptime(dateInString, '%H:%M:%S') + timedelta(hours=2)
        elif ' 26' in dateInString:
            dateInString = dateInString.replace(' 26', '23')
            time_obj = datetime.strptime(dateInString, '%H:%M:%S') + timedelta(hours=3)
        elif ' 27' in dateInString:
            dateInString = dateInString.replace(' 27', '23')
            time_obj = datetime.strptime(dateInString, '%H:%M:%S') + timedelta(hours=4)
        else:
            time_obj = datetime.strptime(dateInString.strip(), '%H:%M:%S')
    return time_obj

def loadSummaryPatient(patient_id):
    """Read summary file for file and seizure times"""
    summary_path = os.path.join(pathDataSet, f'chb{patient_id}', f'chb{patient_id}-summary.txt')
    if not os.path.exists(summary_path):
        return None, None
    
    files = []
    seizures = []
    
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    idx_line = 0
    oldTime = datetime.min
    firstTime = True
    
    while idx_line < len(lines):
        line = lines[idx_line]
        data = line.split(':')
        
        if data[0] == "File Name":
            nF = data[1].strip()
            idx_line += 1
            start_str = lines[idx_line].split(": ")[1].strip()
            s = getTime(start_str)
            
            if firstTime:
                firstTime = False
            else:
                while s < oldTime:
                    s += timedelta(hours=24)
            oldTime = s
            
            idx_line += 1
            end_str = lines[idx_line].split(": ")[1].strip()
            e = getTime(end_str)
            while e < oldTime:
                e += timedelta(hours=24)
            oldTime = e
            
            current_file = FileData(s, e, nF)
            files.append(current_file)
            
            idx_line += 1
            if idx_line < len(lines) and "Number of Seizures" in lines[idx_line]:
                try:
                    num_seizures = int(lines[idx_line].split(':')[1])
                except:
                    num_seizures = 0
                
                for _ in range(num_seizures):
                    idx_line += 1
                    try:
                        secSt = int(lines[idx_line].split(': ')[1].split(' ')[0])
                        idx_line += 1
                        secEn = int(lines[idx_line].split(': ')[1].split(' ')[0])
                        
                        abs_start = s + timedelta(seconds=secSt)
                        abs_end = s + timedelta(seconds=secEn)
                        seizures.append(SeizureData(abs_start, abs_end))
                    except:
                        pass
        idx_line += 1
        
    return files, seizures

def loadDataFromFile(patient_id, fileName):
    """Read a single EDF file"""
    full_path = os.path.join(pathDataSet, f'chb{patient_id}', fileName)
    try:
        f = pyedflib.EdfReader(full_path)
        all_labels = f.getSignalLabels()
        nsamples = f.getNSamples()[0]
        sigbufs = np.zeros((len(TARGET_CHANNELS), nsamples))
        
        for i, target_label in enumerate(TARGET_CHANNELS):
            found_index = -1
            for j, file_label in enumerate(all_labels):
                if target_label == file_label.strip():
                    found_index = j
                    break
            if found_index != -1:
                sigbufs[i, :] = f.readSignal(found_index)
        
        f._close()
        del f
        return sigbufs
    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return None

def extract_segment_data(patient_id, start_dt, end_dt, files_info):
    """Extract data across files"""
    total_seconds = (end_dt - start_dt).total_seconds()
    if total_seconds <= 0:
        return None
        
    total_samples = int(total_seconds * sampleRate)
    result_data = np.zeros((len(TARGET_CHANNELS), total_samples))
    
    relevant_files = []
    for f in files_info:
        if not (f.end < start_dt or f.start > end_dt):
            relevant_files.append(f)
            
    if not relevant_files:
        return None

    filled_count = 0
    
    for rf in relevant_files:
        overlap_start = max(start_dt, rf.start)
        overlap_end = min(end_dt, rf.end)
        
        if overlap_end <= overlap_start:
            continue
            
        res_idx_start = int((overlap_start - start_dt).total_seconds() * sampleRate)
        res_idx_end = int((overlap_end - start_dt).total_seconds() * sampleRate)
        
        file_idx_start = int((overlap_start - rf.start).total_seconds() * sampleRate)
        file_idx_end = int((overlap_end - rf.start).total_seconds() * sampleRate)
        
        file_data = loadDataFromFile(patient_id, rf.nameFile)
        if file_data is None:
            continue
            
        if file_idx_end > file_data.shape[1]:
            file_idx_end = file_data.shape[1]
            
        len_to_copy = file_idx_end - file_idx_start
        actual_res_end = res_idx_start + len_to_copy
        
        if actual_res_end > result_data.shape[1]:
            actual_res_end = result_data.shape[1]
            len_to_copy = actual_res_end - res_idx_start
            
        result_data[:, res_idx_start:actual_res_end] = file_data[:, file_idx_start:file_idx_start+len_to_copy]
        filled_count += len_to_copy

    if filled_count < 0.9 * total_samples:
        return None
        
    return result_data

def segment_into_trails(data_array):
    """Split into trials"""
    if data_array is None or data_array.shape[1] == 0:
        return []
    
    n_samples = data_array.shape[1]
    window_samples = int(WINDOW_SECONDS * sampleRate)
    
    n_trails = n_samples // window_samples
    trails = []
    
    for i in range(n_trails):
        start = i * window_samples
        end = start + window_samples
        trail = data_array[:, start:end]
        trails.append(trail)
        
    return trails

# ---------------- MAIN PROCESS ----------------

def main(output_dir=None):
    """Process CHB-MIT raw EDF by (subject, label) mark usage per trial: first trial=train, rest=test，write pkl with usage directly (no csv)。"""
    out_dir = output_dir or os.path.join(project_root, 'datasets', 'data')
    global_trails = []
    global_labels = []
    global_subjects = []
    global_trial_ids = []
    trial_counter = defaultdict(lambda: defaultdict(int))
    n_channels = len(TARGET_CHANNELS)
    n_timepoints = int(WINDOW_SECONDS * sampleRate)

    for patient_id in patients:
        files_info, seizures = loadSummaryPatient(patient_id)
        if not files_info or not seizures:
            continue
        if patient_id == "19" and len(seizures) > 0:
            seizures.pop(0)

        for sz_idx, sz in enumerate(seizures):
            ictal_raw = extract_segment_data(patient_id, sz.start, sz.end, files_info)
            ictal_trails = segment_into_trails(ictal_raw)
            tid = trial_counter[patient_id][1]
            for _ in ictal_trails:
                global_trails.append(_)
                global_labels.append(1)
                global_subjects.append(patient_id)
                global_trial_ids.append(tid)
            trial_counter[patient_id][1] += 1

            interictal_duration_delta = timedelta(seconds=INTERICTAL_DURATION)
            start_pre = sz.start - interictal_duration_delta
            interictal_raw = extract_segment_data(patient_id, start_pre, sz.start, files_info)
            if interictal_raw is None or interictal_raw.shape[1] < int(INTERICTAL_DURATION * sampleRate * 0.95):
                start_post = sz.end
                end_post = sz.end + interictal_duration_delta
                interictal_raw = extract_segment_data(patient_id, start_post, end_post, files_info)
            if interictal_raw is not None:
                interictal_trails = segment_into_trails(interictal_raw)
                tid0 = trial_counter[patient_id][0]
                for _ in interictal_trails:
                    global_trails.append(_)
                    global_labels.append(0)
                    global_subjects.append(patient_id)
                    global_trial_ids.append(tid0)
                trial_counter[patient_id][0] += 1

    if len(global_trails) == 0:
        raise RuntimeError("No data extracted.")

    data_array = np.stack(global_trails, axis=0)
    labels_array = np.array(global_labels)
    subjects_array = np.array(global_subjects)
    usage = np.array(['train' if tid == 0 else 'test' for tid in global_trial_ids], dtype='U10')

    eeg_data_obj = EEGData(
        dataset_name='CHB_MIT',
        eeg_data=data_array,
        subject_ids=subjects_array,
        channel_names=TARGET_CHANNELS,
        sampling_rate=sampleRate,
        labels=labels_array,
        dataset_type='classification',
        split_method='LOSO',
        is_binary=True,
        usage=usage,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_eeg_data_to_pkl(eeg_data_obj, output_dir=out_dir)
    return eeg_data_obj


def download_data_CHB_MIT(output_dir=None, **kwargs):
    """Called by run_finetuning：generate with usage CHB_MIT.pkl。"""
    return main(output_dir=output_dir)


if __name__ == '__main__':
    main()