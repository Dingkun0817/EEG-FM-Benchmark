import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import numpy as np
from .Model_PSD_SVM import PSD_SVM
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_PSD_SVM(ModelLoader):
    """PSD + SVM loader (classification only)."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, n_freq_bins, fs, fmin, fmax, C, kernel, gamma.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)

        self.n_freq_bins = kwargs.get('n_freq_bins', 20)
        self.fs = kwargs.get('fs', 100)
        self.fmin = kwargs.get('fmin', 0.5)
        self.fmax = kwargs.get('fmax', 30.0)

        self.C = kwargs.get('C', 1.0)
        self.kernel = kwargs.get('kernel', 'rbf')
        self.gamma = kwargs.get('gamma', 'scale')

        super().__init__(eeg_dataset, finetune_strategy)

        if self.task_type == 'classification':
            self.psd_svm = PSD_SVM(
                n_freq_bins=self.n_freq_bins,
                fs=self.fs,
                fmin=self.fmin,
                fmax=self.fmax,
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma
            )
        else:
            raise ValueError("PSD_SVM supports classification only")

        self.is_trained = False

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).
            y: (n_trials,).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        else:
            y = np.array(y)

        self.psd_svm.fit(X, y)
        self.is_trained = True

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        return self.psd_svm.predict(X)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Class probabilities (n_trials, n_classes).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        return self.psd_svm.predict_proba(X)


class EEGPreprocessor_PSD_SVM(Preprocessor):
    """Preprocessor defaults for PSD_SVM (0.5–30 Hz, z-score, target 100 Hz)."""

    def __init__(
        self,
        target_fs=100,
        l_freq=0.5,
        h_freq=30.0,
        notch_freq=None,
        normalize_method='z_score',
        time_length=None,
        apply_EA=False
    ):
        """
        Args:
            target_fs: Resampling target (Hz).
            l_freq / h_freq: Bandpass.
            notch_freq: Optional notch (Hz).
            normalize_method: Normalization name.
            time_length: Segment length (seconds).
        """
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA
        )
