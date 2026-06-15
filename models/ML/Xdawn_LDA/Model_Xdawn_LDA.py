import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Xdawn_LDA:
    """
    MNE Xdawn (ERP-enhancing spatial filters) + Linear Discriminant Analysis.
    """

    def __init__(self, n_components=6, sfreq=250, reg=0.1):
        """
        Args:
            n_components: Number of Xdawn components.
            sfreq: Sampling rate (Hz).
            reg: Regularization for Xdawn (stabilizes covariance).
        """
        self.n_components = n_components
        self.sfreq = sfreq
        self.reg = reg
        self.xdawn = mne.preprocessing.Xdawn(n_components=n_components, reg=reg)
        self.lda = LinearDiscriminantAnalysis()
        self.ch_names = [f'EEG_{i}' for i in range(64)]

    def _create_epochs(self, X, y=None):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: Optional (n_trials,) event ids for Epochs.

        Returns:
            mne.EpochsArray
        """
        if X.ndim != 3 or X.shape[2] == 0:
            raise ValueError(f"Invalid X shape {X.shape}; expected (n_trials, n_channels, n_time_points)")

        n_channels = X.shape[1]
        self.ch_names = [f'EEG_{i}' for i in range(n_channels)]

        info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=['eeg'] * n_channels)

        trials = []
        for i in range(X.shape[0]):
            data = X[i].astype(np.float64)

            raw = mne.io.RawArray(data, info)

            events = np.array([[0, 0, 1]])

            epochs = mne.Epochs(raw, events, tmin=0, tmax=(data.shape[1] - 1) / self.sfreq, baseline=None)
            trials.append(epochs.get_data()[0])

        all_data = np.array(trials)
        events = np.zeros((X.shape[0], 3), dtype=int)
        events[:, 0] = np.arange(X.shape[0])
        events[:, 2] = y if y is not None else np.ones(X.shape[0])

        epochs = mne.EpochsArray(all_data, info, events=events)
        return epochs

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        epochs = self._create_epochs(X, y)

        X_xdawn = self.xdawn.fit_transform(epochs)

        if X_xdawn.ndim == 3:
            n_trials = X_xdawn.shape[0]
            X_xdawn = X_xdawn.reshape(n_trials, -1)

        self.lda.fit(X_xdawn, y)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        X = np.array(X).astype(np.float64)

        epochs = self._create_epochs(X)

        X_xdawn = self.xdawn.transform(epochs)

        if X_xdawn.ndim == 3:
            n_trials = X_xdawn.shape[0]
            X_xdawn = X_xdawn.reshape(n_trials, -1)

        return self.lda.predict(X_xdawn)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_classes) probabilities.
        """
        X = np.array(X).astype(np.float64)

        epochs = self._create_epochs(X)

        X_xdawn = self.xdawn.transform(epochs)

        if X_xdawn.ndim == 3:
            n_trials = X_xdawn.shape[0]
            X_xdawn = X_xdawn.reshape(n_trials, -1)

        return self.lda.predict_proba(X_xdawn)
