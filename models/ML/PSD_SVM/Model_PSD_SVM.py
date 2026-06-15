import numpy as np
import mne
from sklearn.svm import SVC


class PSD_SVM:
    """
    Welch PSD features (MNE) + RBF (or other kernel) SVM; common for sleep staging-style tasks.
    """

    def __init__(self, n_freq_bins=20, fs=100, fmin=0.5, fmax=30.0, C=1.0, kernel='rbf', gamma='scale'):
        """
        Args:
            n_freq_bins: Reserved for future binning (currently full PSD is flattened).
            fs: Sampling rate (Hz).
            fmin / fmax: Welch band.
            C, kernel, gamma: sklearn SVC arguments.
        """
        self.n_freq_bins = n_freq_bins
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax

        self.svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

    def _extract_psd_features(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Flattened PSD per trial.
        """
        n_trials, n_channels, n_time_points = X.shape
        psd_features = []

        for trial in range(n_trials):
            psds, freqs = mne.time_frequency.psd_array_welch(
                X[trial],
                sfreq=self.fs,
                fmin=self.fmin,
                fmax=self.fmax,
                n_fft=256,
                n_per_seg=256,
                n_overlap=128
            )

            psd_flat = psds.flatten()
            psd_features.append(psd_flat)

        return np.array(psd_features)

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        X_psd = self._extract_psd_features(X)

        self.svm.fit(X_psd, y)

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

        return self.svm.predict(X_psd)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_classes) probabilities.
        """
        X = np.array(X).astype(np.float64)

        X_psd = self._extract_psd_features(X)

        return self.svm.predict_proba(X_psd)
