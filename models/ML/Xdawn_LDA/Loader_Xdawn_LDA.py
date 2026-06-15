import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import numpy as np
from .Model_Xdawn_LDA import Xdawn_LDA
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_Xdawn_LDA(ModelLoader):
    """Xdawn + LDA loader (classification)."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance (optionally with .fs for sampling rate).
            **kwargs: finetune_strategy, dropout_rate, n_components.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        kwargs.get('dropout_rate', 0.1)
        n_components = kwargs.get('n_components', 6)
        super().__init__(eeg_dataset, finetune_strategy)

        self.sfreq = getattr(eeg_dataset, 'fs', 250)

        self.n_components = n_components

        if self.task_type == 'classification':
            self.xdawn_lda = Xdawn_LDA(n_components=n_components, sfreq=self.sfreq)
        else:
            raise ValueError("Xdawn_LDA supports classification only")

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

        self.xdawn_lda.fit(X, y)
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

        return self.xdawn_lda.predict(X)

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

        return self.xdawn_lda.predict_proba(X)


class EEGPreprocessor_Xdawn_LDA(Preprocessor):
    """Bandpass defaults for Xdawn / ERP-friendly preprocessing (1–24 Hz, CAR)."""

    def __init__(
        self,
        target_fs=250,
        l_freq=1,
        h_freq=24,
        notch_freq=None,
        normalize_method='car',
        time_length=1,
        apply_EA=False
    ):
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA
        )
