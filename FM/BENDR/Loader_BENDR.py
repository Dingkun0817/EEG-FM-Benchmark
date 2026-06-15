import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from models.FM.BENDR.Model_BENDR import ConvEncoderBENDR
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import Conv1dWithConstraint, get_pretrained_models_path


class Loader_BENDR(ModelLoader):
    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: from_pretrain, finetune_strategy, dropout_rate, channel_filling,
                readout ('flatten' or 'pooling'), pool_length (used when readout is 'pooling'), etc.
        """
        from_pretrain = kwargs.get('from_pretrain', True)
        finetune_strategy = kwargs.get('finetune_strategy', 'full')
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.channel_filling = kwargs.get('channel_filling', 'Conv')
        readout = kwargs.get('readout', 'pooling')
        pool_length = kwargs.get('pool_length', 1)
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))

        super().__init__(eeg_dataset, finetune_strategy=finetune_strategy)

        self.from_pretrain = from_pretrain
        self.readout = readout
        self.pool_length = pool_length
        encoded_samples = ConvEncoderBENDR(20).downsampling_factor(self.num_time_points)
        if self.readout == 'flatten':
            self.feature_dim = 512 * encoded_samples
        else:
            self.feature_dim = 512 * pool_length
            self.pooling = nn.AdaptiveAvgPool1d(pool_length)
        self.input_chans = self.num_channels
        self.scale_param = torch.nn.Parameter(torch.tensor(1.))

        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        if self.from_pretrain:
            pretrained_path = get_pretrained_models_path('encoder.pt')
            encoder.load(pretrained_path)
            print(f"Loaded pretrained weights from {pretrained_path}")

        self.main_model = encoder

        in_channels = 19
        self.chan_conv = Conv1dWithConstraint(self.num_channels, in_channels, 1, max_norm=1)

        self.task_head = nn.Identity()
        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()

    def forward(self, x):
        x = self.chan_conv(x)
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        output = self.main_model(x)
        if self.readout == 'pooling':
            output = self.pooling(output)
        output = output.flatten(1)
        return self.task_head(output)


class EEGPreprocessor_BENDR(Preprocessor):
    """Preprocessor for BENDR: default target_fs=256, normalize_method='car', time_length=4."""

    def __init__(self, target_fs=256, l_freq=None, h_freq=None, notch_freq=None, normalize_method='car', time_length=4, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)

    def preprocess(self, eeg_data, task_mode='Cross', train_percentage=0.3, **kwargs):
        data = super().preprocess(eeg_data, task_mode=task_mode, train_percentage=train_percentage, **kwargs)
        return data
