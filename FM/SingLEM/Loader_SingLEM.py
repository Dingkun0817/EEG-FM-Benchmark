import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from models.FM.SingLEM.Model_SingLEM import EEGEncoder, Config
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class EEGPreprocessor_SingLEM(Preprocessor):
    """Preprocessor defaults for SingLEM."""

    def __init__(self, target_fs=128, l_freq=0.5, h_freq=50, notch_freq=50, normalize_method=None, **kwargs):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, **kwargs)

    def preprocess(self, eeg_data, task_mode='Cross', train_percentage=0.3, **kwargs):
        """
        Args:
            eeg_data: EEGData instance.
            task_mode: 'Cross' or 'Within'.
            train_percentage: Train fraction for few-shot-style splits.
            **kwargs: Passed to base preprocess.

        Returns:
            Preprocessed array (batch, channels, time).
        """
        return super().preprocess(eeg_data, task_mode, train_percentage, **kwargs)


class Loader_SingLEM(ModelLoader):
    """SingLEM EEGEncoder loader with linear readout."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate, readout ('flatten' or 'pooling'), etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        self.config = Config()
        self.config.mask_prob = 0.0
        self.readout = readout

        current_seq_len = self.num_time_points // self.config.token_len
        if self.readout == 'flatten':
            self.feature_dim = self.config.rep_dim * current_seq_len * self.num_channels
        else:
            self.feature_dim = self.config.rep_dim

        self.main_model = EEGEncoder(config=self.config)

        from utils.utils import get_pretrained_models_path
        pretrained_path = get_pretrained_models_path('singlem_pretrained.pt')
        self.load_pretrained_weights(pretrained_path, strict=True)

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)

        self.apply_finetune_strategy()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time).

        Returns:
            Task head output (batch, nb_classes).
        """
        x = x * 10000

        batch_size, num_channels, num_time_points = x.shape
        token_len = self.config.token_len
        current_seq_len = num_time_points // token_len

        x = x.reshape(batch_size, num_channels, current_seq_len, token_len)

        x = x.reshape(-1, 1, token_len)

        features, _, _ = self.main_model(x)

        features = features.reshape(batch_size, num_channels, current_seq_len, self.config.rep_dim)
        if self.readout == 'pooling':
            features = features.mean(dim=(1, 2))
        else:
            features = features.reshape(batch_size, current_seq_len * num_channels * self.config.rep_dim)

        output = self.task_head(features)

        return output

    def load_pretrained_weights(self, pretrained_path, strict=True):
        """Load checkpoint; strips ``module.`` and optional ``encoder.`` prefixes."""
        if not os.path.exists(pretrained_path):
            print(f"Warning: pretrained weights not found at {pretrained_path}")
            return False

        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        if 'encoder.' in next(iter(state_dict.keys())):
            state_dict = {k[8:]: v for k, v in state_dict.items()}

        self.main_model.load_state_dict(state_dict, strict=strict)
        return True
