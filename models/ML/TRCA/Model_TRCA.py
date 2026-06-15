import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import solve, norm


class TRCA:
    """
    Task-Related Component Analysis (TRCA) for EEG classification,
    commonly used for steady-state visual evoked potential (SSVEP) decoding.
    """

    def __init__(self, n_components=1, sampling_rate=250, frequencies=None, phases=None):
        """
        Args:
            n_components: Number of TRCA spatial filters to keep.
            sampling_rate: Sampling rate in Hz.
            frequencies: List of SSVEP stimulus frequencies.
            phases: Phase per frequency (same length as frequencies).
        """
        self.n_components = n_components
        self.sampling_rate = sampling_rate
        self.frequencies = frequencies if frequencies is not None else []
        self.phases = phases if phases is not None else [0] * len(self.frequencies)
        self.filters = None
        self.reference_signals = None
        self.scaler = StandardScaler()

    def _compute_covariance(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            cov_avg: Mean trial covariance (n_channels, n_channels).
            cov_sum: Mean cross-trial covariance sum (n_channels, n_channels).
        """
        n_trials, n_channels, n_time_points = X.shape

        cov_avg = np.zeros((n_channels, n_channels))
        for trial in range(n_trials):
            X_trial = X[trial]
            cov_avg += X_trial @ X_trial.T / n_time_points
        cov_avg /= n_trials

        cov_sum = np.zeros((n_channels, n_channels))
        for trial_i in range(n_trials):
            X_i = X[trial_i]
            for trial_j in range(n_trials):
                if trial_i != trial_j:
                    X_j = X[trial_j]
                    cov_sum += X_i @ X_j.T / n_time_points
        cov_sum /= (n_trials * (n_trials - 1))

        return cov_avg, cov_sum

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        self.filters = []
        self.reference_signals = []

        for class_label in unique_classes:
            class_trials = X[y == class_label]

            cov_avg, cov_sum = self._compute_covariance(class_trials)

            eigenvalues, eigenvectors = np.linalg.eig(solve(cov_avg, cov_sum))

            idx = eigenvalues.argsort()[::-1][:self.n_components]
            filters_class = eigenvectors[:, idx]

            for i in range(filters_class.shape[1]):
                filters_class[:, i] /= norm(filters_class[:, i])

            self.filters.append(filters_class)

            if self.n_components == 1:
                reference = np.zeros((n_classes, class_trials.shape[2]))
                for trial in range(class_trials.shape[0]):
                    reference += (filters_class.T @ class_trials[trial]).squeeze()
                reference /= class_trials.shape[0]
                self.reference_signals.append(reference)
            else:
                self.reference_signals.append(None)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted class index per trial (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        n_trials = X.shape[0]

        predictions = []

        for trial in range(n_trials):
            X_trial = X[trial]
            correlations = []

            for i, (filters_class, reference) in enumerate(zip(self.filters, self.reference_signals)):
                projected = filters_class.T @ X_trial

                if self.n_components == 1:
                    corr = np.corrcoef(projected.squeeze(), reference)[0, 1]
                    correlations.append(corr)
                else:
                    corr_sum = 0
                    for comp in range(projected.shape[0]):
                        corr_sum += np.max(np.corrcoef(projected[comp], X_trial[0])[0, 1:])
                    correlations.append(corr_sum / self.n_components)

            pred_class = np.argmax(correlations)
            predictions.append(pred_class)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Softmax-normalized correlation scores (n_trials, n_classes).
        """
        X = np.array(X).astype(np.float64)
        n_trials = X.shape[0]
        n_classes = len(self.filters)

        probabilities = np.zeros((n_trials, n_classes))

        for trial in range(n_trials):
            X_trial = X[trial]
            correlations = []

            for i, (filters_class, reference) in enumerate(zip(self.filters, self.reference_signals)):
                projected = filters_class.T @ X_trial

                if self.n_components == 1:
                    corr = np.corrcoef(projected.squeeze(), reference)[0, 1]
                    correlations.append(corr)
                else:
                    corr_sum = 0
                    for comp in range(projected.shape[0]):
                        corr_sum += np.max(np.corrcoef(projected[comp], X_trial[0])[0, 1:])
                    correlations.append(corr_sum / self.n_components)

            exp_corr = np.exp(np.array(correlations) - np.max(correlations))
            probabilities[trial] = exp_corr / exp_corr.sum()

        return probabilities
