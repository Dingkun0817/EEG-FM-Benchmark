import os
import sys
import json
from typing import Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from models.FM.BrainOmni_Tiny.Model_BrainOmni_Tiny import BrainOmni
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import get_pretrained_models_path
import mne

class Loader_BrainOmni_Tiny(ModelLoader):
    """BrainOmni_Tiny multimodal loader: EEG + sensor positions + types."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, from_pretrain (default True), dropout_rate (default 0.1),
                readout ('flatten' or 'pooling').
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        from_pretrain = kwargs.get('from_pretrain', True)
        ckpt_path = get_pretrained_models_path('BrainOmni', 'tiny')
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        super().__init__(eeg_dataset, finetune_strategy)
        
        self.from_pretrain = from_pretrain
        self.ckpt_path = ckpt_path
        self.readout = readout
        self.num_t= eeg_dataset.get_duration()  
        self.num_channels = eeg_dataset.get_channel_count()
        
        self.input_chans = self.num_channels
        self.main_model, self.lm_dim = self.get_brainomni(
            pretrained=self.from_pretrain,
            ckpt_path=self.ckpt_path
        )  
        
        self.nb_classes = eeg_dataset.get_label_count()
        if hasattr(self.main_model, 'head'):
            self.main_model.head = nn.Identity()
        self.ch_names = eeg_dataset.get_channel_names()
        ch_types = ["eeg" for sample in self.ch_names]
        sfreq = 250
        info = mne.create_info(ch_names=self.ch_names, sfreq=sfreq, ch_types=ch_types)
        self.info = info
        self.pos, self.sensor_type = extract_pos_sensor_type(self.info)
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
            out_tem = self.main_model(data_tem,pos, sensor_type)
        if self.readout == 'pooling':
            out_tem = out_tem.mean(dim=(1, 2))
            self.feature_dim = out_tem.shape[1]
        else:
            self.feature_dim = out_tem.shape[1] * out_tem.shape[2] * out_tem.shape[3]
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
        pos = self.pos.unsqueeze(0).expand(b, -1, -1)
        sensor_type = self.sensor_type.unsqueeze(0).expand(b, -1)
        output = self.main_model(x, pos, sensor_type)
        if self.readout == 'pooling':
            output = output.mean(dim=(1, 2))
        output = self.task_head(output)
        return output

    def get_brainomni(self, pretrained: bool = True, ckpt_path: Optional[str] = None):
        """
        Args:
            pretrained: Load BrainOmni.pt weights if True.
            ckpt_path: Directory containing model_cfg.json and BrainOmni.pt.

        Returns:
            (model, lm_dim).
        """
        if ckpt_path is None:
            ckpt_path = get_pretrained_models_path('BrainOmni')
        
        model_config_path = os.path.join(ckpt_path, "model_cfg.json")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")
        
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        model_config.update({
            'num_channels': self.num_channels,
            'num_time_points': self.num_t
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
    
    


class EEGPreprocessor_BrainOmni_Tiny(Preprocessor):
    """Preprocessor for BrainOmni_Tiny: default target_fs=256, l_freq=0.1, h_freq=96, normalize_method='z_score'."""
    def __init__(self, target_fs=256, l_freq=0.1, h_freq=96.0, notch_freq=50.0, normalize_method="z_score", time_length=4, apply_EA=True):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)

LOW = 0.1
HIGH = 96
SAMPLE_RATE = 256
SEED = 42
TOKENIZER_SEGMENT_TIME = 2.0
PRETRAIN_DTYPE = torch.bfloat16
DOWNSTREAM_DTYPE = torch.bfloat16
STANDARD_1020 = [
    "Fp1","Fpz","Fp2","AF9","AF7","AF5","AF3","AF1","AFz","AF2","AF4","AF6","AF8","AF10","F9","F7","F5",
    "F3","F1","Fz","F2","F4","F6","F8","F10","FT9","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8",
    "FT10","T9","T7","C5","C3","C1","Cz","C2","C4","C6","T8","T10","TP9","TP7","CP5","CP3","CP1","CPz",
    "CP2","CP4","CP6","TP8","TP10","P9","P7","P5","P3","P1","Pz","P2","P4","P6","P8","P10","PO9","PO7",
    "PO5","PO3","PO1","POz","PO2","PO4","PO6","PO8","PO10","O1","Oz","O2","O9","CB1","CB2","Iz","O10","T3",
    "T5","T4","T6","M1","M2","A1","A2","CFC1","CFC2","CFC3","CFC4","CFC5","CFC6","CFC7","CFC8","CCP1",
    "CCP2","CCP3","CCP4","CCP5","CCP6","CCP7","CCP8","T1","T2","FTT9h","TTP7h","TPP9h","FTT10h","TPP8h",
    "TPP10h","Fp1-F7","F7-T7","T7-P7","P7-O1","Fp2-F8","F8-T8","T8-P8","P8-O2","Fp1-F3","F3-C3","C3-P3",
    "P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2",
]
# eeg:5697       meg: 5261
NEW_DEVICE_DATASET_LIST = ["ds005261-1.0.0", "ds005697-1.0.2"]

PROJECT_ROOT_PATH = project_root  # must end with /
DATA_ROOT_PATH = "./data/"  # must end with /

RAW_PATH = os.path.join(DATA_ROOT_PATH, "raw") + "/"
PROCESSED_PRETRAIN_PATH=os.path.join(DATA_ROOT_PATH, "processed_pretrain") + "/"
EVALUATE_PATH = os.path.join(DATA_ROOT_PATH, "evaluate") + "/"
PROCESSED_EVALUATE_PATH=os.path.join(DATA_ROOT_PATH, "processed_evaluate") + "/"

PRETRAIN_METADATA_PATH = os.path.join(PROJECT_ROOT_PATH, "share", "metadata",'pretrain')
EVALUATE_METADATA_PATH=os.path.join(PROJECT_ROOT_PATH,'share',"metadata",'evaluate')
CUSTOM_MONTAGE_PATH = os.path.join(current_dir, 'custom_montages')

MONTAGE_DICT = {
    "ds004902-1.0.5": "brainproducts-RNP-BA-128",
    "ds002778-1.0.5": "biosemi32",
    "ds003775-1.2.1": "biosemi64",
    "ds002721-1.0.3": "standard_1020",
    "ds005420-1.0.0": "standard_1020",
    "ds005620-1.0.0": "standard_1020",
    "ds003555-1.0.1": "standard_1020",
}
CUSTOM_MONTAGE_DICT = {
    "ds003478-1.1.0": os.path.join(CUSTOM_MONTAGE_PATH, "ds003478-1.1.0.tsv"),
    "ds004395-2.0.0": os.path.join(CUSTOM_MONTAGE_PATH, "ds004395-2.0.0.tsv"),
    "ds005697-1.0.2": os.path.join(CUSTOM_MONTAGE_PATH, "ds005697-1.0.2.tsv"),
    "stroke": os.path.join(CUSTOM_MONTAGE_PATH, "stroke.tsv"),
    "ds004148": os.path.join(CUSTOM_MONTAGE_PATH, "ds004148.tsv"),
}
SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}
# raw.rename_channels
RENAME_DICT = {
    "ds002721-1.0.3": {"FP1": "Fp1", "FP2": "Fp2"},
    "ds005420-1.0.0": {
        "EEG Fp1-A1A2": "Fp1",
        "EEG Fp2-A1A2": "Fp2",
        "EEG Fz-A1A2": "Fz",
        "EEG F3-A1A2": "F3",
        "EEG F4-A1A2": "F4",
        "EEG F7-A1A2": "F7",
        "EEG F8-A1A2": "F8",
        "EEG Cz-A1A2": "Cz",
        "EEG C3-A1A2": "C3",
        "EEG C4-A1A2": "C4",
        "EEG T3-A1A2": "T3",
        "EEG T4-A1A2": "T4",
        "EEG Pz-A1A2": "Pz",
        "EEG P3-A1A2": "P3",
        "EEG P4-A1A2": "P4",
        "EEG T5-A1A2": "T5",
        "EEG T6-A1A2": "T6",
        "EEG O1-A1A2": "O1",
        "EEG O2-A1A2": "O2",
    },
}


EXCLUDE_DICT = {
    "MEG-Narrative-Dataset": [
        "EEG057-4302",
        "EEG058-4302",
        "EEG059-4302",
        "EEG060-4302",
        "EEG061-4302",
        "EEG062-4302",
        "EEG063-4302",
        "EEG064-4302",
    ],
    "ds000117-1.0.6": ["Cz2", "Cpz"],
    "ds000247-1.0.2": ["ECG", "VEOG", "HEOG"],
    "ds002778-1.0.5": ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"],
    "ds003478-1.1.0": ["CB1", "CB2", "HEOG", "VEOG"],
    "ds004148": ["Cpz"],
    "ds004186-2.0.0": ["Cz"],
    "ds004395-2.0.0": ["E8", "E25", "E126", "E127", "E129"],
    "ds004998-1.2.2": [
        "EEG002",
        "EEG003",
        "EEG004",
        "EEG005",
        "EEG006",
        "EEG007",
        "EEG008",
    ],
    "ds005420-1.0.0": ["EEG LOC-ROC"],
    "ds005505-1.0.0": ["Cz"],
    "ds005506-1.0.0": ["Cz"],
    "ds005507-1.0.0": ["Cz"],
    "ds005508-1.0.0": ["Cz"],
    "ds005509-1.0.0": ["Cz"],
    "ds005510-1.0.0": ["Cz"],
    "ds005511-1.0.0": ["Cz"],
    "ds005512-1.0.0": ["Cz"],
    "ds005620-1.0.0": ["EMG", "HEOG", "VEOG"],
    "ds005697-1.0.2": ["CB1", "CB2", "Trigger"],
    "ds003555-1.0.1": ["T1", "T2"],
}

HPI_LIST = [
    "camcan1630",
    "ds000117-1.0.6",
    "ds004330-1.0.0",
]

montage = [
    "standard_1005",
    "standard_1020",
    "standard_alphabetic",
    "standard_postfixed",
    "standard_prefixed",
    "standard_primed",
    "biosemi16",
    "biosemi32",
    "biosemi64",
    "biosemi128",
    "biosemi160",
    "biosemi256",
    "easycap-M1",
    "easycap-M10",
    "easycap-M43",
    "EGI_256",
    "GSN-HydroCel-32",
    "GSN-HydroCel-64_1.0",
    "GSN-HydroCel-65_1.0",
    "GSN-HydroCel-128",
    "GSN-HydroCel-129",
    "GSN-HydroCel-256",
    "GSN-HydroCel-257",
    "mgh60",
    "mgh70",
    "artinis-octamon",
    "artinis-brite23",
    "brainproducts-RNP-BA-128",
]



def extract_pos_sensor_type(info):
    pos = []
    sensor_type = []

    montage = mne.channels.make_standard_montage("standard_1005")
    montage_positions = montage.get_positions()['ch_pos']
    montage_positions_lower = {k.lower(): v for k, v in montage_positions.items()}

    for i in info["chs"]:
        kind = 2

        if kind == 2:
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


