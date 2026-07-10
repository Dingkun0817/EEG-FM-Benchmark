import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import mne
from safetensors.torch import load_file

from models.FM.LUNA_Base.Model_LUNA_Base import LUNA
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import get_pretrained_models_path

TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'Fpz-Cz', 'Pz-Oz',
]


def generate_fake_mask(batch_size, c, t):
    return torch.zeros(batch_size, c, t, dtype=torch.bool)


def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list({part for ch in channel_names for part in ch.split('-')})
    else:
        names = channel_names

    info = mne.create_info(ch_names=names, sfreq=256, ch_types=['eeg'] * len(names))
    info = info.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={'cb1': 'POO7', 'cb2': 'POO8'},
    )
    montage_positions = info.get_montage().get_positions()['ch_pos']
    locs = []
    for name in channel_names:
        if name in TARGET_CHANNELS:
            electrode1, electrode2 = name.split('-')
            loc1 = montage_positions[electrode1]
            loc2 = montage_positions[electrode2]
            locs.append((loc1 + loc2) / 2)
        else:
            locs.append(montage_positions[name])
    return locs


def pad_time_to_patch(x, target_patch=40):
    remainder = x.shape[2] % target_patch
    if remainder != 0:
        x = F.pad(x, (0, target_patch - remainder), mode='constant', value=0)
    return x


class Loader_LUNA_Base(ModelLoader):
    """LUNA_Base loader with channel-location embeddings."""

    PRETRAINED_FILENAME = 'LUNA_base.safetensors'

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


class EEGPreprocessor_LUNA_Base(Preprocessor):
    """Preprocessor for LUNA_Base: default target_fs=256, bandpass 0.5–50 Hz."""

    def __init__(self, target_fs=256, l_freq=0.5, h_freq=50, notch_freq=None, normalize_method=None, time_length=4, apply_EA=True):
        super().__init__(
            target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq,
            normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA,
        )
