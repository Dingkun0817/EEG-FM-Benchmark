import os
import sys
import json
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from models.FM.BrainOmni_Base.Model_BrainOmni_Base import BrainOmni
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import get_pretrained_models_path
import mne


class Loader_BrainOmni_Base(ModelLoader):
    """BrainOmni_Base loader: EEG + sensor positions + types."""

    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get('finetune_strategy', None)
        from_pretrain = kwargs.get('from_pretrain', True)
        ckpt_path = get_pretrained_models_path('BrainOmni', 'base')
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))

        dropout_rate = kwargs.get('dropout_rate', 0.1)
        super().__init__(eeg_dataset, finetune_strategy)

        self.from_pretrain = from_pretrain
        self.ckpt_path = ckpt_path
        self.readout = readout
        self.num_t = eeg_dataset.get_duration()
        self.num_channels = eeg_dataset.get_channel_count()
        self.input_chans = self.num_channels
        self.main_model, self.lm_dim = self.get_brainomni(
            pretrained=self.from_pretrain,
            ckpt_path=self.ckpt_path,
        )

        self.nb_classes = eeg_dataset.get_label_count()
        if hasattr(self.main_model, 'head'):
            self.main_model.head = nn.Identity()
        ch_types = ["eeg" for _ in self.ch_names]
        info = mne.create_info(ch_names=self.ch_names, sfreq=250, ch_types=ch_types)
        self.pos, self.sensor_type = extract_pos_sensor_type(info)
        eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(self.sensor_type)
        self.pos = normalize_pos(self.pos, eeg_mask, mag_mask)
        self.pos = torch.from_numpy(self.pos)
        self.sensor_type = torch.from_numpy(self.sensor_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos = self.pos.to(device)
        self.sensor_type = self.sensor_type.to(device)

        with torch.no_grad():
            self.main_model.eval()
            pos = self.pos.unsqueeze(0).expand(1, -1, -1).to('cpu')
            sensor_type = self.sensor_type.unsqueeze(0).expand(1, -1).to('cpu')
            data_tem = torch.randn(1, self.num_channels, self.num_time_points).to('cpu')
            out_tem = self.main_model(data_tem, pos, sensor_type)
        if self.readout == 'pooling':
            out_tem = out_tem.mean(dim=(1, 2))
            self.feature_dim = out_tem.shape[1]
        else:
            self.feature_dim = out_tem.shape[1] * out_tem.shape[2] * out_tem.shape[3]
        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        self.apply_finetune_strategy()

    def forward(self, x):
        b, n, t = x.shape
        pos = self.pos.unsqueeze(0).expand(b, -1, -1)
        sensor_type = self.sensor_type.unsqueeze(0).expand(b, -1)
        output = self.main_model(x, pos, sensor_type)
        if self.readout == 'pooling':
            output = output.mean(dim=(1, 2))
        output = self.task_head(output)
        return output

    def get_brainomni(self, pretrained: bool = True, ckpt_path: Optional[str] = None):
        if ckpt_path is None:
            ckpt_path = get_pretrained_models_path('BrainOmni', 'base')

        model_config_path = os.path.join(ckpt_path, "model_cfg.json")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")

        with open(model_config_path, 'r') as f:
            model_config = json.load(f)

        model_config.update({
            'num_channels': self.num_channels,
            'num_time_points': self.num_t,
        })

        model = BrainOmni(**model_config)

        if pretrained:
            checkpoint_path = os.path.join(ckpt_path, "BrainOmni.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Pretrained weights file not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint, strict=True)

            for p in model.tokenizer.parameters():
                p.requires_grad = False

        return model, model.lm_dim


class EEGPreprocessor_BrainOmni_Base(Preprocessor):
    """Preprocessor for BrainOmni_Base: default target_fs=256, l_freq=0.1, h_freq=96, normalize_method='z_score'."""

    def __init__(self, target_fs=256, l_freq=0.1, h_freq=96.0, notch_freq=50.0, normalize_method="z_score", time_length=4, apply_EA=True):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)


SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}
CUSTOM_MONTAGE_PATH = os.path.join(current_dir, 'custom_montages')


def extract_pos_sensor_type(info):
    pos = []
    sensor_type = []

    montage = mne.channels.make_standard_montage("standard_1005")
    montage_positions = montage.get_positions()['ch_pos']
    montage_positions_lower = {k.lower(): v for k, v in montage_positions.items()}

    for i in info["chs"]:
        ch_name = i["ch_name"]
        ch_name_lower = ch_name.lower()

        if ch_name_lower in montage_positions_lower:
            ch_pos = montage_positions_lower[ch_name_lower]
            pos.append(np.hstack([ch_pos, np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])
        else:
            print(f"Warning: Channel {ch_name} not found in montage")
            pos.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])

    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.array(sensor_type).astype(np.int32)
    return pos, sensor_type


def get_sensor_type_mask(sensor_type: np.ndarray):
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]
    meg_mask = mag_mask | grad_mask
    return eeg_mask, mag_mask, grad_mask, meg_mask


def normalize_pos(pos: np.ndarray, eeg_mask, meg_mask):
    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        pos[eeg_mask, :3] /= eeg_scale
    return pos
