import numpy as np
import pandas as pd
from scipy import signal
from scipy.linalg import fractional_matrix_power
import mne
from multiprocessing import Pool


def _validate_2d_3d(data):
    """Require data to be 2D or 3D; raise ValueError otherwise."""
    if data.ndim not in (2, 3):
        raise ValueError("Input data must be 2D or 3D")


def _bandpass_core(data, l_freq, h_freq, fs):
    """Bandpass filter core: 2D array (..., n_times), returns same shape."""
    data = np.asarray(data, dtype=np.float64)
    if l_freq is None and h_freq is None:
        return data
    if l_freq is None:
        b, a = signal.butter(4, h_freq / (fs / 2), btype='low')
    elif h_freq is None:
        b, a = signal.butter(4, l_freq / (fs / 2), btype='high')
    else:
        b, a = signal.butter(4, [l_freq / (fs / 2), h_freq / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def _notch_core(data, notch_freq, fs, Q):
    """Notch filter core: 2D array (..., n_times), returns same shape."""
    data = np.asarray(data, dtype=np.float64)
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data, axis=-1)


def _bandpass_filter_chunk(args):
    """Worker for parallel bandpass: (data_chunk, l_freq, h_freq, fs) -> filtered chunk."""
    data_chunk, l_freq, h_freq, fs = args
    return _bandpass_core(data_chunk, l_freq, h_freq, fs)


def _notch_filter_chunk(args):
    """Worker for parallel notch: (data_chunk, notch_freq, fs, Q) -> filtered chunk."""
    data_chunk, notch_freq, fs, Q = args
    return _notch_core(data_chunk, notch_freq, fs, Q)


def _resample_chunk(args):
    """Worker for parallel resample: (data_chunk, new_fs, original_fs) -> resampled chunk."""
    data_chunk, new_fs, original_fs = args
    data_chunk = np.asarray(data_chunk, dtype=np.float64)
    return mne.filter.resample(data_chunk, down=(original_fs / new_fs))


class Preprocessor:
    """EEG preprocessing pipeline: resample, bandpass/notch, optional EA, normalization (z_score, min_max, CAR, etc.)."""
    def __init__(self,
                 target_fs=None,
                 l_freq=None,
                 h_freq=None,
                 notch_freq=None,
                 notch_Q=30,
                 normalize_method=None,
                 time_length=None,
                 apply_EA=False,
                 n_workers=1):
        """target_fs: resample to this Hz (None = no resample). l_freq/h_freq: bandpass. notch_freq/notch_Q: notch. apply_EA: covariance-based re-reference. n_workers: parallel workers for filter/resample."""
        self.target_fs = target_fs
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.notch_Q = notch_Q
        self.normalize_method = normalize_method
        self.time_length = time_length
        self.apply_EA = apply_EA
        self.n_workers = max(1, int(n_workers)) if n_workers is not None else 1

    def _map_3d_chunks(self, data, worker_func, make_task_args):
        """Map worker_func over 3D (n_samples, n_chans, n_times) in chunks; make_task_args(chunk) returns args for worker_func."""
        data = np.asarray(data, dtype=np.float64)
        n_samples = data.shape[0]
        if self.n_workers <= 1 or n_samples <= 1:
            result = np.zeros_like(data)
            for i in range(n_samples):
                chunk = data[i : i + 1]
                result[i] = worker_func(make_task_args(chunk))[0]
            return result
        n_workers = min(self.n_workers, n_samples)
        chunk_size = (n_samples + n_workers - 1) // n_workers
        chunks = [data[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]
        tasks = [make_task_args(c) for c in chunks]
        with Pool(n_workers) as pool:
            parts = pool.map(worker_func, tasks)
        return np.concatenate(parts, axis=0)

    def bandpass_filter(self, data, l_freq=None, h_freq=None, fs=250):
        """Apply bandpass filter to EEG (2D or 3D)."""
        data = np.asarray(data, dtype=np.float64)
        _validate_2d_3d(data)
        if data.ndim == 2:
            return _bandpass_core(data, l_freq, h_freq, fs)
        return self._map_3d_chunks(data, _bandpass_filter_chunk, lambda c: (c, l_freq, h_freq, fs))

    def notch_filter(self, data, notch_freq=50, fs=250, Q=50):
        """Notch filter (e.g. 50/60 Hz line noise)."""
        data = np.asarray(data, dtype=np.float64)
        _validate_2d_3d(data)
        if data.ndim == 2:
            return _notch_core(data, notch_freq, fs, Q)
        return self._map_3d_chunks(data, _notch_filter_chunk, lambda c: (c, notch_freq, fs, Q))

    def resample_data(self, data, new_fs, original_fs=250):
        """Resample to new_fs Hz (MNE-based)."""
        data = np.asarray(data, dtype=np.float64)
        _validate_2d_3d(data)
        if data.ndim == 2:
            return mne.filter.resample(data, down=(original_fs / new_fs))
        if self.n_workers <= 1 or data.shape[0] <= 1:
            return mne.filter.resample(data, down=(original_fs / new_fs))
        return self._map_3d_chunks(data, _resample_chunk, lambda c: (c, new_fs, original_fs))

    def _trim_or_pad_time(self, data, num_samples_target, time_axis):
        """Trim or pad (repeat from start) along time_axis to num_samples_target; data 3D (n, c, t)."""
        current_length = data.shape[time_axis]
        if current_length >= num_samples_target:
            return np.take(data, np.arange(num_samples_target), axis=time_axis).copy()
        shape = list(data.shape)
        shape[time_axis] = num_samples_target
        new_data = np.zeros(shape, dtype=data.dtype)
        sl = [slice(None)] * 3
        sl[time_axis] = slice(0, current_length)
        new_data[tuple(sl)] = data
        remaining = num_samples_target - current_length
        while remaining > 0:
            fill_len = min(remaining, current_length)
            start_pos = num_samples_target - remaining
            sl_src = [slice(None)] * 3
            sl_src[time_axis] = slice(0, fill_len)
            sl_dst = [slice(None)] * 3
            sl_dst[time_axis] = slice(start_pos, start_pos + fill_len)
            new_data[tuple(sl_dst)] = data[tuple(sl_src)]
            remaining -= fill_len
        return new_data

    def adjust_time_length(self, data, target_length, fs):
        """Trim or pad (repeat from start) to target_length seconds; 2D or 3D."""
        data = np.asarray(data)
        num_samples_target = int(target_length * fs)
        if data.ndim == 2:
            data_3d = data[np.newaxis, ...]
            out = self._trim_or_pad_time(data_3d, num_samples_target, time_axis=2)
            return out[0]
        if data.ndim == 3:
            return self._trim_or_pad_time(data, num_samples_target, time_axis=2)
        raise ValueError("Input data must be 2D or 3D")

    def exponential_moving_standardize(self, data, ems_factor=0.001, eps=1e-4):
        """Exponential moving standardization."""
        data = np.asarray(data)
        _validate_2d_3d(data)
        if data.ndim == 2:
            data = data.T
            df = pd.DataFrame(data)
            meaned = df.ewm(alpha=ems_factor).mean()
            demeaned = df - meaned
            squared = demeaned * demeaned
            square_ewmed = squared.ewm(alpha=ems_factor).mean()
            standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
            return standardized.T
            
        elif data.ndim == 3:
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                result[i] = self.exponential_moving_standardize(data[i], ems_factor, eps)
            return result
        raise ValueError("Input data must be 2D or 3D")

    def percentile95_normalize(self, data, axis=-1, q=0.95):
        """Normalize by 95th percentile of absolute values."""
        percentile = np.quantile(np.abs(data), q=q, axis=axis, keepdims=True)
        return data / (percentile + 1e-8)
    
    def common_average_reference(self, data):
        """Common average reference (CAR): subtract channel mean per sample/time."""
        if len(data.shape) == 3:
            avg_ref = np.mean(data, axis=1, keepdims=True)
            data = data - avg_ref
        else:
            avg_ref = np.mean(data, axis=0, keepdims=True)
            data = data - avg_ref
        return data
    
    def EA(self, x):
        """
        Parameters
        ----------
        x : numpy array
            data of shape (num_samples, num_channels, num_time_samples)

        Returns
        ----------
        XEA : numpy array
            data of shape (num_samples, num_channels, num_time_samples)
        """
        cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            cov[i] = np.cov(x[i])
        refEA = np.mean(cov, 0)
        sqrtRefEA = fractional_matrix_power(refEA, -0.5) 
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
            XEA[i] = np.dot(sqrtRefEA, x[i])
        return XEA
    
    def _apply_EA(self, data, subject, labels, task_mode, train_percentage=0.3):
        """Apply EA (covariance-based re-reference) per subject; in Fewshot, split train/test per subject/label for EA."""
        unique_subject_ids = np.unique(subject)
        if task_mode == 'Cross':
            for subject_id in unique_subject_ids:
                subject_mask = (subject == subject_id)
                data[subject_mask] = self.EA(data[subject_mask])
                
        elif task_mode == 'Fewshot':
            for subject_id in unique_subject_ids:

                subject_mask = (subject == subject_id)
                subject_indices = np.where(subject_mask)[0]
                subject_labels = labels[subject_mask]

                train_mask = np.zeros_like(subject, dtype=bool)
                test_mask = np.zeros_like(subject, dtype=bool)  

                for label in np.unique(subject_labels):

                    label_mask = subject_labels == label
                    label_indices = subject_indices[label_mask]

                    train_size = max(1, int(len(label_indices) * train_percentage))
                    train_indices = label_indices[:train_size]
                    test_indices = label_indices[train_size:]

                    train_mask[train_indices] = True
                    test_mask[test_indices] = True

                data[train_mask] = self.EA(data[train_mask])
                data[test_mask] = self.EA(data[test_mask])

        return data
    
    def _norm_z_score(self, data):
        data = np.asarray(data, dtype=np.float64)
        axis = 2 if data.ndim == 3 else 1
        mu = np.expand_dims(np.mean(data, axis=axis), axis=axis)
        sigma = np.expand_dims(np.std(data, axis=axis), axis=axis)
        return (data - mu) / (sigma + 1e-8)

    def _norm_min_max(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 2:
            min_v = np.expand_dims(np.min(data, axis=1), axis=1)
            max_v = np.expand_dims(np.max(data, axis=1), axis=1)
        else:
            min_v = np.expand_dims(np.expand_dims(np.min(data, axis=(0, 2)), axis=0), axis=2)
            max_v = np.expand_dims(np.expand_dims(np.max(data, axis=(0, 2)), axis=0), axis=2)
        return (data - min_v) / (max_v - min_v + 1e-8) * 2 - 1

    def normalize_data(self, data, normalize_method='none', ems_factor=0.001, factor=100, axis=-1):
        """Apply normalization: z_score, min_max, ems, 0.1mv, 95, car, or none."""
        data = np.asarray(data, dtype=np.float64)
        method = (normalize_method or 'none').lower()
        if method == 'z_score':
            return self._norm_z_score(data)
        if method == 'min_max':
            return self._norm_min_max(data)
        if method == 'ems':
            return self.exponential_moving_standardize(data, ems_factor=ems_factor)
        if method == '0.1mv':
            return data / factor
        if method == '95':
            return self.percentile95_normalize(data, axis=axis)
        if method == 'car':
            return self.common_average_reference(data)
        if method == 'none':
            return data
        raise ValueError(f"Unsupported normalize_method: {normalize_method}")

    def preprocess(self, eeg_data, task_mode='Cross', train_percentage=0.3, **kwargs):
        """Full pipeline: resample, trim/pad, bandpass, notch, optional EA, then normalization."""
        data = eeg_data.eeg_data
        fs = eeg_data.sampling_rate
        subject = eeg_data.subject_ids
        label = eeg_data.labels
        data = np.asarray(data)
        current_original_fs = fs
        current_target_fs = self.target_fs
        fs = current_target_fs if current_target_fs is not None else current_original_fs
        if current_target_fs is not None:
            data = self.resample_data(data, current_target_fs, current_original_fs)
        if self.time_length is not None:
            data = self.adjust_time_length(data, self.time_length, fs)
        if self.l_freq is not None or self.h_freq is not None:
            data = self.bandpass_filter(data, self.l_freq, self.h_freq, fs)
        if self.notch_freq is not None:
            data = self.notch_filter(data, self.notch_freq, fs, self.notch_Q)
        if self.apply_EA:
            data = self._apply_EA(data, subject, label, task_mode, train_percentage)
        if self.normalize_method is not None and self.normalize_method.lower() != 'none':
            data = self.normalize_data(data, self.normalize_method)
        
        return data
    
    def __call__(self, eeg_data, task_mode='Cross', train_percentage=0.3):
        return self.preprocess(eeg_data, task_mode, train_percentage)

