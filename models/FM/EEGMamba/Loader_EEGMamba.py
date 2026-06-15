import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.FM.EEGMamba.Model_EEGMamba import EEGMamba
from utils.EEGDataLoader import EEGData
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_EEGMamba(ModelLoader):
    """EEGMamba loader with optional EEGMamba.pth pretrained weights."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate, readout ('flatten' or 'pooling'), etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        readout = kwargs.get('readout', 'pooling')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        if not isinstance(eeg_dataset, EEGData):
            raise TypeError("eeg_dataset must be an EEGData instance")

        self.dataset_name = eeg_dataset.dataset_name
        self.num_t = eeg_dataset.get_duration()
        self.readout = readout
        patch_size = 200
        seq_len = self.num_time_points // patch_size
        if self.readout == 'flatten':
            self.feature_dim = int(200 * len(self.ch_names) * seq_len)
        else:
            self.feature_dim = 200

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.main_model = EEGMamba().to(self.device)

        from utils.utils import get_pretrained_models_path
        pretrained_path = get_pretrained_models_path('EEGMamba.pth')
        self._load_pretrained_weights(pretrained_path, strict=True)

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()
        self.task_head.to(self.device)

    def _load_pretrained_weights(self, pretrained_path, strict=True):
        if not os.path.exists(pretrained_path):
            print(f"Warning: pretrained weights not found at {pretrained_path}")
            return

        print(f"Loading pretrained weights from {pretrained_path}")

        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        for k in state_dict:
            state_dict[k] = state_dict[k].to(self.device)

        self.main_model.load_state_dict(state_dict, strict=strict)
        print("Pretrained weights loaded.")

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        x = self.main_model(x)
        if self.readout == 'pooling':
            h = x.mean(dim=(1, 2))
        else:
            h = x.flatten(1)
        h = self.task_head(h)

        return h


class EEGPreprocessor_EEGMamba(Preprocessor):
    """Preprocessor for EEGMamba: default target_fs=200, l_freq=0.5, h_freq=50, apply_EA=True."""

    def __init__(self, target_fs=200, l_freq=0.5, h_freq=50, notch_freq=None, normalize_method=None, time_length=4, apply_EA=True):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
