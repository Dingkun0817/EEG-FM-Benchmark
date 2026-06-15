import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class PSD_LDA:
    """
    Welch PSD features + Linear Discriminant Analysis (e.g. sleep staging).
    """

    def __init__(self, n_freq_bins=20, fs=100):
        """
        Args:
            n_freq_bins: Number of frequency bins after downsampling.
            fs: Sampling rate (Hz) passed to Welch.
        """
        self.n_freq_bins = n_freq_bins
        self.fs = fs
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        X_psd = self._extract_psd_features(X)

        self.lda.fit(X_psd, y)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        X = np.array(X).astype(np.float64)

        X_psd = self._extract_psd_features(X)

        return self.lda.predict(X_psd)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_classes) probabilities.
        """
        X = np.array(X).astype(np.float64)

        X_psd = self._extract_psd_features(X)

        return self.lda.predict_proba(X_psd)

    def _extract_psd_features(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Flattened PSD (n_trials, n_channels * n_freq_bins).
        """
        n_trials, n_channels, n_time_points = X.shape

        fmin = 0.5
        fmax = 30.0

        psd_features = []
        for trial in X:
            psds, freqs = mne.time_frequency.psd_array_welch(
                trial, sfreq=self.fs, fmin=fmin, fmax=fmax, n_fft=256, n_per_seg=256
            )

            if len(freqs) > self.n_freq_bins:
                step = len(freqs) // self.n_freq_bins
                psd = psds[:, ::step][:, :self.n_freq_bins]
            else:
                psd = psds

            psd_flat = psd.flatten()
            psd_features.append(psd_flat)

        return np.array(psd_features)
