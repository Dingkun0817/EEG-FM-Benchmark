import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.FM.CBraMod.Model_CBraMod import CBraMod
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_CBraMod(ModelLoader):
    """CBraMod loader: patch-based backbone + linear task head on flattened features."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, from_pretrain, dropout_rate,
                readout ('flatten' or 'pooling'), etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        from_pretrain = kwargs.get('from_pretrain', True)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        self.from_pretrain = from_pretrain
        self.readout = readout
        if self.readout == 'flatten':
            self.feature_dim = self.num_time_points * self.num_channels
        else:
            self.feature_dim = 200
        self.input_chans = self.num_channels

        self.main_model = CBraMod()

        if self.from_pretrain:
            from utils.utils import get_pretrained_models_path
            pretrained_path = get_pretrained_models_path('CBraMod.pth')
            if os.path.exists(pretrained_path):
                self.main_model.load_state_dict(
                    torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False),
                    strict=True
                )
                print(f"Loaded pretrained CBraMod from {pretrained_path}")
            else:
                print(f"Warning: pretrained weights not found at {pretrained_path}")

        self.main_model.proj_out = nn.Identity()

        self.task_head = nn.Identity()
        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, num_time_points).

        Returns:
            Task head output.
        """
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        output = self.main_model(x)
        if self.readout == 'pooling':
            output = output.mean(dim=(1, 2))
        output = self.task_head(output)

        return output


class EEGPreprocessor_CBraMod(Preprocessor):
    """Preprocessor for CBraMod: default target_fs=200, bandpass 4–32 Hz, notch 60 Hz, normalize 'car'."""

    def __init__(self, target_fs=200, l_freq=4, h_freq=32.0, notch_freq=60.0, normalize_method='car', time_length=None, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, apply_EA=apply_EA, time_length=time_length)
