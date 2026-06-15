import os
import sys
import numpy as np
import torch

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

from utils.preprocessing import Preprocessor
from .Model_TRCA import TRCA


class EEGPreprocessor_TRCA(Preprocessor):
    """Lightweight preprocessor for TRCA (mostly passthrough; optional extension point)."""

    def __init__(
        self,
        target_fs=250,
        l_freq=8.0,
        h_freq=40.0,
        notch_freq=None,
        normalize_method=None,
        time_length=None,
        apply_EA=False,
    ):
        """
        Args:
            target_fs: Target sampling rate (Hz).
            l_freq: Low-frequency cutoff for bandpass (if parent applies).
            h_freq: High-frequency cutoff.
            notch_freq: Notch frequency.
            normalize_method: Normalization method name.
            time_length: Window length in seconds.
            apply_EA: Whether to apply Euclidean alignment-style steps in parent.
        """
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA,
        )

    def fit(self, X, y=None):
        """No-op fit (TRCA pipeline does not fit this preprocessor)."""
        return self

    def transform(self, X):
        """
        Args:
            X: EEG (n_trials, n_channels, n_time_points).

        Returns:
            Same shape, float64 numpy.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class Loader_TRCA:
    """TRCA wrapper for SSVEP-style classification."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: Object with sampling_rate, dataset_type, optional target_frequencies, label count.
            **kwargs: n_components, dropout_rate (unused), etc.
        """
        kwargs.get('dropout_rate', 0.1)
        n_components = kwargs.get('n_components', 1)

        self.n_components = n_components
        self.sampling_rate = getattr(eeg_dataset, 'sampling_rate', 250)

        self.task_type = getattr(eeg_dataset, 'dataset_type', 'classification')

        self.frequencies = getattr(eeg_dataset, 'target_frequencies', None)

        self.num_classes = getattr(eeg_dataset, 'num_classes', None)
        if self.num_classes is None and hasattr(eeg_dataset, 'get_label_count'):
            self.num_classes = eeg_dataset.get_label_count()

        if self.task_type == 'classification':
            self.trca = TRCA(
                n_components=n_components,
                sampling_rate=self.sampling_rate,
                frequencies=self.frequencies
            )
        else:
            raise ValueError("TRCA supports classification only")

        self.preprocessor = EEGPreprocessor_TRCA(target_fs=self.sampling_rate)

        self.is_trained = False

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
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

        X_processed = self.preprocessor.fit_transform(X)

        self.trca.fit(X_processed, y)
        self.is_trained = True

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        X_processed = self.preprocessor.transform(X)

        return self.trca.predict(X_processed)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Class probabilities (n_trials, n_classes).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)

        X_processed = self.preprocessor.transform(X)

        return self.trca.predict_proba(X_processed)

    def save_model(self, save_path):
        """Persist TRCA filters and metadata to disk (pickle)."""
        import pickle

        model_data = {
            'n_components': self.n_components,
            'sampling_rate': self.sampling_rate,
            'frequencies': self.frequencies,
            'phases': getattr(self.trca, 'phases', None),
            'filters': self.trca.filters,
            'reference_signals': self.trca.reference_signals,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"TRCA model saved to {save_path}")

    def load_model(self, load_path):
        """Load TRCA state from pickle."""
        import pickle

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        self.trca = TRCA(
            n_components=model_data['n_components'],
            sampling_rate=model_data['sampling_rate'],
            frequencies=model_data.get('frequencies'),
        )
        if model_data.get('phases') is not None:
            self.trca.phases = model_data['phases']
        self.trca.filters = model_data.get('filters')
        self.trca.reference_signals = model_data.get('reference_signals')
        self.n_components = model_data['n_components']
        self.sampling_rate = model_data['sampling_rate']
        self.frequencies = model_data.get('frequencies')
        self.is_trained = True

        print(f"TRCA model loaded from {load_path}")

        return self.trca
