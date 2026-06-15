import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from functools import partial
from models.FM.EEGPT.Model_EEGPT import EEGTransformer
from utils.EEGDataLoader import EEGData
from utils.utils import Conv1dWithConstraint
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_EEGPT(ModelLoader):
    """EEGPT (EEGTransformer) loader with optional pretrained target_encoder weights."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate, readout ('flatten' or 'pooling'), etc.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        if not isinstance(eeg_dataset, EEGData):
            raise TypeError("eeg_dataset must be an EEGData instance")

        self.num_t = eeg_dataset.get_duration()
        self.readout = readout

        self.main_model = EEGTransformer(
            img_size=[len(self.ch_names), self.num_time_points],
            patch_size=64,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        from utils.utils import get_pretrained_models_path
        pretrained_path = get_pretrained_models_path('eegpt.ckpt')

        if os.path.exists(pretrained_path):
            pretrain_ckpt = torch.load(pretrained_path, weights_only=False)
            target_encoder_stat = {}
            for k, v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]] = v
            self.main_model.load_state_dict(target_encoder_stat, strict=True)
        else:
            print(f"Warning: pretrained file not found at {pretrained_path}; using random init")

        self.chans_id = self.main_model.prepare_chan_ids(self.ch_names)
        self.chan_conv = Conv1dWithConstraint(len(self.ch_names), len(self.ch_names), 1, max_norm=1)

        with torch.no_grad():
            x0 = torch.randn(1, len(self.ch_names), self.num_time_points)
            x0 = self.chan_conv(x0)
            out0 = self.main_model(x0)
            if self.readout == 'flatten':
                self.feature_dim = int(out0.numel() // out0.shape[0])
            else:
                self.feature_dim = int(out0.shape[-1])

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, num_time_points).

        Returns:
            Task head output.
        """
        x = self.chan_conv(x)
        x = self.main_model(x, self.chans_id.to(x))
        if self.readout == 'pooling':
            h = x.mean(dim=(1, 2))
        else:
            h = x.flatten(2).flatten(1)
        h = self.task_head(h)
        return h


class EEGPreprocessor_EEGPT(Preprocessor):
    """Preprocessor for EEGPT: default target_fs=256, l_freq=0.5, h_freq=50, apply_EA=True."""

    def __init__(self, target_fs=256, l_freq=0.5, h_freq=50, notch_freq=None, normalize_method=None, time_length=4, apply_EA=True):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
