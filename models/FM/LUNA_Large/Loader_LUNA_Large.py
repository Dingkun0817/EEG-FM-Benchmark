import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import numpy as np
from safetensors.torch import load_file

from models.FM.LUNA_Large.Model_LUNA_Large import LUNA
from models.FM.LUNA_Base.Loader_LUNA_Base import (
    generate_fake_mask,
    get_channel_locations,
    pad_time_to_patch,
)
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import get_pretrained_models_path


class Loader_LUNA_Large(ModelLoader):
    """LUNA_Large loader with channel-location embeddings."""

    PRETRAINED_FILENAME = 'LUNA_large.safetensors'

    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get('finetune_strategy', None)
        from_pretrain = kwargs.get('from_pretrain', True)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'pooling')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        self.from_pretrain = from_pretrain
        self.readout = readout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.main_model = LUNA().to(self.device)
        locs = get_channel_locations(self.ch_names)
        self.channel_locations = torch.from_numpy(np.stack(locs, axis=0)).float().to(self.device)

        if self.from_pretrain:
            self._load_pretrained(get_pretrained_models_path(self.PRETRAINED_FILENAME))

        with torch.no_grad():
            self.main_model.eval()
            x_dummy = pad_time_to_patch(
                torch.randn(1, self.num_channels, self.num_time_points, device=self.device)
            )
            mask = generate_fake_mask(1, self.num_channels, x_dummy.shape[2]).to(self.device)
            out = self.main_model(x_dummy, mask, self.channel_locations, readout=self.readout)
        if self.readout == 'pooling':
            self.feature_dim = out.shape[-1]
        else:
            self.feature_dim = int(out.numel() // out.shape[0])

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()
        self.task_head.to(self.device)

    def _load_pretrained(self, pretrained_path):
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        state_dict = load_file(pretrained_path)
        state_dict = {
            k: v for k, v in state_dict.items()
            if 'decoder_head' not in k and 'channel_emb' not in k
        }
        self.main_model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        x = pad_time_to_patch(x)
        mask = generate_fake_mask(x.shape[0], x.shape[1], x.shape[2]).to(self.device)
        output = self.main_model(x, mask, self.channel_locations, readout=self.readout)
        return self.task_head(output)


class EEGPreprocessor_LUNA_Large(Preprocessor):
    """Preprocessor for LUNA_Large: default target_fs=256, bandpass 0.5–50 Hz."""

    def __init__(self, target_fs=256, l_freq=0.5, h_freq=50, notch_freq=None, normalize_method=None, time_length=4, apply_EA=True):
        super().__init__(
            target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq,
            normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA,
        )
