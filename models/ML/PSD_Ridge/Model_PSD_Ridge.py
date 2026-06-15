import numpy as np
import mne
from sklearn.linear_model import LogisticRegression, Ridge


class PSD_Ridge:
    """
    Power spectral density (Welch) features with Ridge regression or logistic regression.
    """

    def __init__(self, n_freq_bins=100, task_type='classification', alpha=1.0):
        """
        Args:
            n_freq_bins: Target number of frequency bins after downsampling PSD.
            task_type: 'classification' (LogisticRegression) or 'regression' (Ridge).
            alpha: Ridge penalty (regression only).
        """
        self.n_freq_bins = n_freq_bins
        self.task_type = task_type
        self.alpha = alpha

        if task_type == 'classification':
            self.model = LogisticRegression()
        elif task_type == 'regression':
            self.model = Ridge(alpha=alpha)
        else:
            raise ValueError(f"Unsupported task type: {task_type}; use 'classification' or 'regression'")

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,) or (n_trials, n_outputs) for regression.
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        X_psd = self._extract_psd_features(X)

        self.model.fit(X_psd, y)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Model predictions.
        """
        X = np.array(X).astype(np.float64)

        X_psd = self._extract_psd_features(X)

        return self.model.predict(X_psd)

    def predict_proba(self, X):
        """
        Classification only.

        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_classes) probabilities.
        """
        if self.task_type != 'classification':
            raise RuntimeError("predict_proba only applies to classification")

        X = np.array(X).astype(np.float64)

        X_psd = self._extract_psd_features(X)

        return self.model.predict_proba(X_psd)

    def _extract_psd_features(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_channels * n_freq_bins) flattened PSD.
        """
        n_trials, n_channels, n_time_points = X.shape
        fs = 200

        psd_features = []
        for trial in X:
            psds, freqs = mne.time_frequency.psd_array_welch(
                trial, sfreq=fs, fmin=1.0, fmax=40.0, n_fft=256, n_per_seg=256
            )

            if len(freqs) > self.n_freq_bins:
                step = len(freqs) // self.n_freq_bins
                psd = psds[:, ::step][:, :self.n_freq_bins]
            else:
                psd = psds

            psd_flat = psd.flatten()
            psd_features.append(psd_flat)

        return np.array(psd_features)
