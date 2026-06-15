import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class CSP_LDA:
    """
    Common Spatial Patterns (CSP) with Linear Discriminant Analysis (LDA).
    Uses MNE for CSP feature extraction.
    """

    def __init__(self, n_components=10):
        """
        Args:
            n_components: Number of CSP spatial filters.
        """
        self.n_components = n_components
        self.csp = mne.decoding.CSP(n_components=n_components)
        self.lda = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).
            y: (n_trials,).
        """
        X = np.array(X).astype(np.float64)
        y = np.array(y)

        X_csp = self.csp.fit_transform(X, y)

        self.lda.fit(X_csp, y)

        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        X = np.array(X).astype(np.float64)

        X_csp = self.csp.transform(X)

        return self.lda.predict(X_csp)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, n_channels, n_time_points).

        Returns:
            Class probabilities (n_trials, n_classes).
        """
        X = np.array(X).astype(np.float64)

        X_csp = self.csp.transform(X)

        return self.lda.predict_proba(X_csp)
