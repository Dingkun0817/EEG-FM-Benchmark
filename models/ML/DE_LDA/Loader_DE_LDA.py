import torch
import numpy as np
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from .Model_DE_LDA import DE_LDA


class Loader_DE_LDA(ModelLoader):
    """DE + LDA sklearn wrapper with optional preprocessing for finetune-style API."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate, window_size, step_size, configs for preprocessor.
        """
        finetune_strategy = kwargs.get('finetune_strategy', 'full')
        dropout_rate = kwargs.get('dropout_rate', 0.5)

        super().__init__(eeg_dataset, finetune_strategy)

        self.model_name = "DE_LDA"

        self.dropout_rate = dropout_rate

        self.main_model = DE_LDA(
            fs=self.sampling_rate,
            window_size=kwargs.get('window_size', 1),
            step_size=kwargs.get('step_size', 0.5)
        )

        configs = kwargs.get('configs', {})
        self.preprocessor = EEGPreprocessor_DE_LDA(**configs) if configs else EEGPreprocessor_DE_LDA()

    def forward(self, x):
        """
        No-op for API parity: DE_LDA uses numpy in fit/predict.

        Args:
            x: Tensor or array (batch, channels, time).

        Returns:
            Unmodified input tensor.
        """
        return x

    def fit(self, X, y):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).
            y: (n_trials,).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        else:
            y = np.array(y)

        self.main_model.fit(X, y)
        return self

    def predict(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Predicted labels (n_trials,).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)

        return self.main_model.predict(X)

    def predict_proba(self, X):
        """
        Args:
            X: (n_trials, num_channels, num_time_points).

        Returns:
            Class probabilities (n_trials, n_classes).
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        else:
            X = np.array(X)

        return self.main_model.predict_proba(X)

    def finetune(self, train_loader, val_loader, save_path, patience=10):
        """
        Collect numpy batches, optional preprocess transform, fit DE_LDA, evaluate on val, pickle model.

        Args:
            train_loader: Yields (x, y) batches.
            val_loader: Validation batches.
            save_path: Pickle output path.
            patience: Unused (kept for API compatibility).

        Returns:
            Validation accuracy (float).
        """
        x_train = []
        y_train = []

        for batch in train_loader:
            x, y = batch
            x_train.append(x.numpy())
            y_train.append(y.numpy())

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        x_val = []
        y_val = []

        for batch in val_loader:
            x, y = batch
            x_val.append(x.numpy())
            y_val.append(y.numpy())

        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        x_train = self.preprocessor.transform(x_train)
        x_val = self.preprocessor.transform(x_val)

        self.main_model.fit(x_train, y_train)

        y_pred = self.main_model.predict(x_val)
        accuracy = np.mean(y_pred == y_val)

        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(self.main_model, f)

        return accuracy


class EEGPreprocessor_DE_LDA(Preprocessor):
    """Default bandpass for DE_LDA pipeline (1–24 Hz, CAR)."""

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

    def transform(self, X):
        """Numpy passthrough for Loader_DE_LDA.finetune (full EEGData pipeline uses __call__)."""
        return np.asarray(X, dtype=np.float64)
