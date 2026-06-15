import numpy as np
import pickle
import os

class EEGData:
    """Container for EEG dataset: data array, subject/trial ids, labels, and optional usage (train/test)."""

    def __init__(self, dataset_name, eeg_data, subject_ids, channel_names, sampling_rate,
                 labels=None, dataset_type='classification', regression_values=None, img_feature=None, split_method='LOSO', is_binary=False, usage=None):

        self.dataset_name = dataset_name
        self.eeg_data = np.array(eeg_data)
        self.subject_ids = np.array(subject_ids)
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate

        if labels is not None:
            if not isinstance(labels, np.ndarray):
                self.labels = np.array(labels)
            else:
                self.labels = labels
        else:
            self.labels = None

        if dataset_type not in ['classification', 'regression', 'matching']:
            raise ValueError("dataset_type must be 'classification', 'regression', or 'matching'")
        self.dataset_type = dataset_type

        if regression_values is not None:
            if not isinstance(regression_values, np.ndarray):
                self.regression_values = np.array(regression_values)
            else:
                self.regression_values = regression_values
        else:
            self.regression_values = None

        if img_feature is not None:
            if not isinstance(img_feature, np.ndarray):
                self.img_feature = np.array(img_feature)
            else:
                self.img_feature = img_feature
        else:
            self.img_feature = None

        self.split_method = split_method
        self.is_binary = is_binary

        if usage is not None:
            if not isinstance(usage,np.ndarray):
               self.usage = np.array(usage)
            else:
                self.usage = usage
        else:
            self.usage = None
    
    def get_sample_count(self):
        return self.eeg_data.shape[0]

    def get_channel_count(self):
        return self.eeg_data.shape[1]

    def get_time_point_count(self):
        return self.eeg_data.shape[2]

    def get_duration(self):
        return self.get_time_point_count() / self.sampling_rate

    def get_subject_ids(self):
        return self.subject_ids

    def get_channel_names(self):
        return self.channel_names

    def get_subject_unique_count(self):
        return len(np.unique(self.subject_ids))

    def get_label_count(self):
        if self.labels is None:
            return 0
        return len(np.unique(self.labels))

    def get_data_by_label(self, label):
        """Return EEG data for samples with the given label."""
        if self.labels is None:
            raise ValueError("Dataset has no labels")
        mask = self.labels == label
        return self.eeg_data[mask]
    
    def __str__(self):
        result = (
            f"Dataset: {self.dataset_name}\n"
            f"Samples: {self.get_sample_count()}, Channels: {self.get_channel_count()}, Time points: {self.get_time_point_count()}\n"
            f"Sampling rate: {self.sampling_rate} Hz, Duration per sample: {self.get_duration():.2f} s\n"
            f"Subjects: {self.get_subject_unique_count()}, Task: {self.dataset_type}\n"
        )
        if self.labels is not None:
            result += f"Labels: {self.get_label_count()} classes, {list(np.unique(self.labels))}\n"
        if self.regression_values is not None:
            result += f"Regression shape: {self.regression_values.shape}, range: [{self.regression_values.min():.4f}, {self.regression_values.max():.4f}]\n"
        if self.img_feature is not None:
            result += f"Image feature shape: {self.img_feature.shape}\n"
        return result


def save_eeg_data_to_pkl(eeg_data_obj, output_dir):
    """Save EEGData instance to a pkl file under output_dir (filename: dataset_name.pkl)."""
    if not isinstance(eeg_data_obj, EEGData):
        raise TypeError("Input must be an EEGData instance")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{eeg_data_obj.dataset_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(eeg_data_obj, f)
    print(f"EEG data saved to {save_path}")


def load_eeg_data_from_pkl(file_path):
    """Load an EEGData instance from a pkl file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        eeg_data_obj = pickle.load(f)
    if not isinstance(eeg_data_obj, EEGData):
        raise TypeError(f"Loaded object is not EEGData: {type(eeg_data_obj)}")
    print(f"Loaded EEG data from {file_path}")
    return eeg_data_obj