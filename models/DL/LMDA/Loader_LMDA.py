import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.DL.LMDA.Model_LMDA import LMDA
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_LMDA(ModelLoader):
    """LMDA loader for classification and regression."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate, etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        super().__init__(eeg_dataset, finetune_strategy)
        if self.task_type == 'matching':
            self.nb_classes = eeg_dataset.img_feature.shape[1]
        self.main_model = LMDA(
            num_classes=self.nb_classes,
            chans=self.num_channels,
            samples=self.num_time_points,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, num_time_points).

        Returns:
            Classification or regression output.
        """
        x = x.unsqueeze(1)
        output = self.main_model(x)
        return output


class EEGPreprocessor_LMDA(Preprocessor):
    """Preprocessor for LMDA: default target_fs=250, normalize_method='car', time_length=4."""

    def __init__(self, target_fs=250, l_freq=8.0, h_freq=32, notch_freq=None, normalize_method='car', time_length=4, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
