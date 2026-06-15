import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from .utils import LinearLayers, RegressionLayers
from .EEGDataLoader import load_eeg_data_from_pkl
from models.config import get_model_category


def import_dataset(dataset_name, n_jobs=None):
    """Load dataset by name: from pkl if present, else via download_data_* (n_jobs used only when pkl missing and supported). Returns (eeg_data, task_type, is_binary, split_method)."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, 'datasets', 'data')
    pkl_file_path = os.path.join(data_dir, f'{dataset_name}.pkl')
    print(f'Dataset pkl path: {pkl_file_path}')

    if os.path.exists(pkl_file_path):
        eeg_data = load_eeg_data_from_pkl(pkl_file_path)
        print(f'Successfully loaded {dataset_name} from {pkl_file_path}')
    else:
        dataset_module_path = f'datasets.{dataset_name}.Preprocess_Dataset'
        module = __import__(dataset_module_path, fromlist=['download_data_' + dataset_name])
        download_func = getattr(module, f'download_data_{dataset_name.replace("-", "_")}')
        if n_jobs is not None and n_jobs > 1:
            try:
                eeg_data = download_func(n_jobs=n_jobs)
            except TypeError:
                eeg_data = download_func()
        else:
            eeg_data = download_func()
        print(f'Successfully downloaded and loaded {dataset_name}')


    task_type = eeg_data.dataset_type
    is_binary = getattr(eeg_data, 'is_binary', False)
    if task_type == 'classification' and not is_binary:
        if eeg_data.get_label_count() == 2:
            is_binary = True
    split_method = getattr(eeg_data, 'split_method', 'LOSO')
    
    return eeg_data, task_type, is_binary, split_method


def import_model(model_name, eeg_data, **kwargs):
    """Load model by name (ML/DL/FM category); supports finetune_strategy ('full' or 'head_only') and dropout_rate for task head."""
    category = get_model_category(model_name)
    module_path = f'models.{category}.{model_name}.Loader_{model_name}'
    class_name = f'Loader_{model_name}'
    module = __import__(module_path, fromlist=[class_name])
    model_loader = getattr(module, class_name)
    model = model_loader(eeg_data, **kwargs)
    print(f'Successfully initialized model: {model_name}')
    finetune_strategy = kwargs.get('finetune_strategy', 'full')
    if finetune_strategy == 'head_only' and hasattr(model, 'set_trainable_parameters'):
        model.set_trainable_parameters('head_only')
    
    return model


def import_preprocessor(model_name, **kwargs):
    """Load model-specific preprocessor; n_workers is set on instance for parallel filtering if supported."""
    n_workers = kwargs.pop('n_workers', 1)
    category = get_model_category(model_name)
    module_path = f'models.{category}.{model_name}.Loader_{model_name}'
    class_name = f'EEGPreprocessor_{model_name}'
    module = __import__(module_path, fromlist=[class_name])
    preprocessor_class = getattr(module, class_name)
    preprocessor = preprocessor_class(**kwargs) if kwargs else preprocessor_class()
    if hasattr(preprocessor, 'n_workers'):
        preprocessor.n_workers = max(1, int(n_workers)) if n_workers is not None else 1
    return preprocessor


def apply_preprocessor(eeg_data, preprocessor, model_name, dataset_name, task_mode='Cross', train_percentage=0.3, **kwargs):
    """Apply preprocessor to EEG data; updates eeg_data.eeg_data and sampling_rate if preprocessor has target_fs."""
    if preprocessor is None:
        return eeg_data
    original_data = eeg_data.eeg_data
    original_fs = eeg_data.sampling_rate
    print(f'Applying {model_name} preprocessing to {dataset_name}')
    print(f'Original data shape: {original_data.shape}, sampling rate: {original_fs} Hz')
    processed_data = preprocessor.preprocess(eeg_data, task_mode=task_mode, train_percentage=train_percentage, **kwargs)
    print(f'Preprocessed data shape: {processed_data.shape}')
    eeg_data.eeg_data = processed_data
    if hasattr(preprocessor, 'target_fs'):
        eeg_data.sampling_rate = preprocessor.target_fs
    
    return eeg_data


if __name__ == '__main__':
    print("\n===== Testing dataset import =====")
    dataset_result = import_dataset('BNCI2014001')
    if dataset_result:
        eeg_data, task_type, is_binary, split_method = dataset_result
        print(f'Imported dataset: task_type={task_type}, split_method={split_method}, is_binary={is_binary}')
        print(f'Data shape: {eeg_data.eeg_data.shape}')
    if dataset_result:
        print("\n===== Testing model import =====")
        model = import_model('LaBraM', eeg_data)
        print(f'Imported model: {model.__class__.__name__}')
    if dataset_result:
        print("\n===== Testing preprocessing =====")
        import copy
        eeg_data_copy = copy.deepcopy(eeg_data)
        preprocessor = import_preprocessor('LaBraM')
        if preprocessor:
            apply_preprocessor(eeg_data_copy, preprocessor, 'LaBraM', 'BNCI2014001')
            print(f'Preprocessed shape: {eeg_data_copy.eeg_data.shape}')