import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import numpy as np
from .Model_PSD_Ridge import PSD_Ridge
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_PSD_Ridge(ModelLoader):
    """PSD + Ridge/Logistic regression; features via MNE Welch PSD."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, n_freq_bins, etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        n_freq_bins = kwargs.get('n_freq_bins', 20)
        super().__init__(eeg_dataset, finetune_strategy)

        self.n_freq_bins = n_freq_bins

        self.psd_ridge = PSD_Ridge(n_freq_bins=n_freq_bins, task_type=self.task_type)

        self.is_trained = False

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).
            y: (n_trials,) or (n_trials, n_outputs).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        else:
            y = np.array(y)

        self.psd_ridge.fit(X, y)
        self.is_trained = True

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        return self.psd_ridge.predict(X)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Class probabilities (classification only).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        return self.psd_ridge.predict_proba(X)


class EEGPreprocessor_PSD_Ridge(Preprocessor):
    """Preprocessor tuned for PSD_Ridge (default band 1–40 Hz)."""

    def __init__(
        self,
        target_fs=250,
        l_freq=1.0,
        h_freq=40.0,
        notch_freq=None,
        normalize_method=None,
        time_length=None,
        apply_EA=False
    ):
        """
        Args:
            target_fs: Resampling target (Hz).
            l_freq: Bandpass low (Hz); default 1 for PSD pipeline.
            h_freq: Bandpass high (Hz); default 40 for PSD pipeline.
            notch_freq: Optional notch (Hz).
            normalize_method: Normalization name.
            time_length: Crop length in seconds.
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
