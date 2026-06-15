import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.DL.ShallowConv.Model_ShallowConv import ShallowConvNet, DeepConvNet
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_ShallowConv(ModelLoader):
    """ShallowConvNet / DeepConvNet loader for classification and regression."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance with channels, samples, labels, etc.
            **kwargs: finetune_strategy, dropout_rate, model_type ('ShallowConvNet' or 'DeepConvNet').
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        model_type = kwargs.get('model_type', 'ShallowConvNet')
        super().__init__(eeg_dataset, finetune_strategy)
        if self.task_type == 'matching':
            self.nb_classes = eeg_dataset.img_feature.shape[1]
        ModelClass = ShallowConvNet if model_type == 'ShallowConvNet' else DeepConvNet

        self.main_model = ModelClass(
            n_classes=self.nb_classes,
            Chans=self.num_channels,
            Samples=self.num_time_points,
            dropoutRate=dropout_rate,
            bn_track=True,
            TemporalKernel_Times=1
        )

    def forward(self, x):
        """
        Args:
            x: EEG tensor (batch_size, num_channels, num_time_points).

        Returns:
            Classification logits/probs or regression outputs.
        """
        x = x.unsqueeze(1)
        output = self.main_model(x)

        return output


class EEGPreprocessor_ShallowConv(Preprocessor):
    """Preprocessor for ShallowConv: default target_fs=250, l_freq=8, h_freq=32; see Preprocessor."""

    def __init__(self, target_fs=250, l_freq=8.0, h_freq=32, notch_freq=None, normalize_method=None, time_length=None, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
