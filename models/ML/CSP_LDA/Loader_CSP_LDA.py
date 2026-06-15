import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
import mne
from models.ML.CSP_LDA.Model_CSP_LDA import CSP_LDA
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor

class Loader_CSP_LDA(ModelLoader):
    """CSP_LDA model loader (classification only), using MNE CSP."""
    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        n_components = kwargs.get('n_components', 10)
        super().__init__(eeg_dataset, finetune_strategy)
        self.n_components = n_components
        if self.task_type == 'classification':
            self.csp_lda = CSP_LDA(n_components=n_components)
        else:
            raise ValueError("CSP_LDA supports classification only")
        self.is_trained = False

    def fit(self, X, y):
        """Train CSP_LDA on EEG data X and labels y."""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)
            
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        else:
            y = np.array(y)
        self.csp_lda.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X):
        """Return predicted labels for EEG data X."""
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)
        
        return self.csp_lda.predict(X)

    def predict_proba(self, X):
        """Return class probabilities for EEG data X."""
        if not self.is_trained:
            raise RuntimeError("Model not trained; call fit first")
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy().astype(np.float64)
        else:
            X = np.array(X).astype(np.float64)
        
        return self.csp_lda.predict_proba(X)

class EEGPreprocessor_CSP_LDA(Preprocessor):
    """Preprocessor for CSP_LDA (MNE CSP): target_fs=250, bandpass 8–32 Hz."""
    def __init__(self,
                 target_fs=250,
                 l_freq=8.0,
                 h_freq=32.0,
                 notch_freq=None,
                 normalize_method=None,
                 time_length=None,
                 apply_EA=False):
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA
        )
        

