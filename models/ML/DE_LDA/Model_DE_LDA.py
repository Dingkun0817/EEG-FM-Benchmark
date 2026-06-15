import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DE_LDA:
    """
    Sliding-window differential entropy features + Linear Discriminant Analysis.
    """

    def __init__(self, fs=250, window_size=1, step_size=0.5):
        """
        Args:
            fs: Sampling rate (Hz).
            window_size: Window length in seconds.
            step_size: Hop between windows in seconds.
        """
        self.fs = fs
        self.window_size = window_size
        self.step_size = step_size
        self.lda = LinearDiscriminantAnalysis()

    def _compute_de(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_features) differential-entropy features.
        """
        n_trials, n_channels, n_time_points = X.shape

        window_samples = int(self.window_size * self.fs)
        step_samples = int(self.step_size * self.fs)

        n_windows = (n_time_points - window_samples) // step_samples + 1

        de_features = []

        for trial in X:
            trial_de = []

            for channel in trial:
                channel_de = []

                for i in range(n_windows):
                    start_idx = i * step_samples
                    end_idx = start_idx + window_samples
                    window_data = channel[start_idx:end_idx]

                    variance = np.var(window_data)

                    de = 0.5 * np.log(2 * np.pi * np.e * variance)
                    channel_de.append(de)

                trial_de.extend(channel_de)

            de_features.append(trial_de)

        return np.array(de_features)

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        X_de = self._compute_de(X)

        self.lda.fit(X_de, y)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        X = np.array(X).astype(np.float64)

        X_de = self._compute_de(X)

        return self.lda.predict(X_de)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            (n_trials, n_classes) probabilities.
        """
        X = np.array(X).astype(np.float64)

        X_de = self._compute_de(X)

        return self.lda.predict_proba(X_de)
