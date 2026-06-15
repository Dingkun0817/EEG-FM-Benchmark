import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.FM.BIOT_6D.Model_BIOT_6D import BIOTClassifier
from utils.utils import Conv1dWithConstraint
from utils.ModelLoader import ModelLoader
from utils.utils import get_input_chans
from utils.preprocessing import Preprocessor


class Loader_BIOT_1D(ModelLoader):
    """BIOT_1D variant (16-channel PREST checkpoint); maps input channels to 16 channels before the backbone."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: from_pretrain, finetune_strategy, dropout_rate, etc.
        """
        from_pretrain = kwargs.get('from_pretrain', True)
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', None)
        readout = kwargs.get('readout', 'pooling')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        self.input_chans = get_input_chans(self.ch_names)

        self.from_pretrain = from_pretrain
        self.readout = readout

        in_channels = 16
        self.chan_conv = Conv1dWithConstraint(self.num_channels, in_channels, 1, max_norm=1)

        model = BIOTClassifier(
            n_classes=self.nb_classes,
            n_channels=in_channels,
            n_fft=200,
            hop_length=100,
            emb_size=256,
            heads=8,
            depth=4
        )

        from utils.utils import get_pretrained_models_path
        pretrained_path = get_pretrained_models_path('EEG-PREST-16-channels.ckpt')
        if self.from_pretrain:
            model.biot.load_state_dict(torch.load(pretrained_path, weights_only=False))
            print(f"Loaded pretrained BIOT_1D from {pretrained_path}")
        else:
            print(f"Warning: from_pretrain=False; skipped pretrained weights at {pretrained_path}")

        model.classifier = nn.Identity()
        self.main_model = model

        with torch.no_grad():
            xd = torch.randn(1, self.num_channels, self.num_time_points)
            xd = self.chan_conv(xd)
            o0 = self.main_model(xd, readout=self.readout)
            self.feature_dim = int(o0.numel() // o0.shape[0])

        self.set_task_head(self.dataset_type, self.nb_classes)

        self.apply_finetune_strategy()

    def forward(self, x):
        """Map input channels to 16 channels, then run BIOT_1D."""
        x = self.chan_conv(x)
        output = self.main_model(x, readout=self.readout)
        output = self.task_head(output)
        return output


class EEGPreprocessor_BIOT_1D(Preprocessor):
    """Preprocessor for BIOT_1D: default target_fs=200, normalize_method='car'; see Preprocessor."""

    def __init__(self, target_fs=200, l_freq=None, h_freq=None, notch_freq=None, normalize_method="car", time_length=None, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, apply_EA=apply_EA, time_length=time_length)

    def preprocess(self, eeg_data, task_mode='Cross', train_percentage=0.3, **kwargs):
        """Run base preprocess."""
        data = super().preprocess(eeg_data, task_mode=task_mode, train_percentage=train_percentage, **kwargs)
        return data
