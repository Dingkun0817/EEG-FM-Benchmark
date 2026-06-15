import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from models.FM.NeuroGPT.Model_NeuroGPT import EEGConformer
from utils.EEGDataLoader import EEGData
from utils.utils import Conv1dWithConstraint
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_NeuroGPT(ModelLoader):
    """NeuroGPT-style loader using EEGConformer as the encoder + task head."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, model_config, ft_only_encoder (default True), dropout_rate, readout ('flatten' or 'pooling').
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        model_config = kwargs.get('model_config', None)
        ft_only_encoder = kwargs.get('ft_only_encoder', True)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        if not isinstance(eeg_dataset, EEGData):
            raise TypeError("eeg_dataset must be an EEGData instance")

        self.num_t = eeg_dataset.get_duration()
        n_chans = 22

        self.nb_classes = eeg_dataset.get_label_count()

        if model_config is None:
            model_config = {
                "num_decoding_classes": self.nb_classes,
                "chunk_len": self.num_t,
                "ft_only_encoder": ft_only_encoder
            }
        self.chan_conv = Conv1dWithConstraint(len(self.ch_names), 22, 1, max_norm=1)

        self.main_model = EEGConformer(
            n_outputs=model_config["num_decoding_classes"],
            n_chans=n_chans,
            n_times=model_config['chunk_len'],
            is_decoding_mode=model_config["ft_only_encoder"]
        )
        from utils.utils import get_pretrained_models_path
        load_path = get_pretrained_models_path('neurogpt.bin')
        if load_path and os.path.exists(load_path):
            try:
                pretrain_ckpt = torch.load(load_path, weights_only=False)

                if 'state_dict' in pretrain_ckpt:
                    pretrain_state_dict = pretrain_ckpt['state_dict']
                elif 'model_state_dict' in pretrain_ckpt:
                    pretrain_state_dict = pretrain_ckpt['model_state_dict']
                else:
                    pretrain_state_dict = pretrain_ckpt

                filtered_state_dict = {}
                encoder_prefix = "encoder."
                for key, value in pretrain_state_dict.items():
                    if key.startswith(encoder_prefix):
                        new_key = key[len(encoder_prefix):]
                        filtered_state_dict[new_key] = value

                if filtered_state_dict:
                    self.main_model.load_state_dict(filtered_state_dict, strict=False)
                    print(f"Loaded encoder weights from {load_path} ({len(filtered_state_dict)} tensors)")
                else:
                    print("Warning: no 'encoder.*' keys in checkpoint; using random init")

            except Exception as e:
                print(f"Warning: failed to load pretrained weights ({e}); using random init")
                if 'missing keys' in str(e) or 'unexpected keys' in str(e):
                    print(e)
        else:
            print(f"Warning: pretrained file not found at {load_path}; using random init")

        self.sampling_rate = eeg_dataset.sampling_rate
        self.model_config = model_config
        self.readout = readout

        with torch.no_grad():
            x0 = torch.randn(1, len(self.ch_names), self.num_time_points)
            x0 = self.chan_conv(x0)
            o0 = self.main_model(x0)
            if self.readout == 'flatten':
                self.feature_dim = int(o0.numel() // o0.shape[0])
            else:
                self.feature_dim = int(o0.shape[-1])

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
        output = self.main_model(x)
        if self.readout == 'pooling':
            output = output.mean(dim=1)
        output = self.task_head(output)
        return output


class EEGPreprocessor_NeuroGPT(Preprocessor):
    """Preprocessor for NeuroGPT: default target_fs=250, l_freq=0.5, h_freq=100, normalize_method='z_score'."""

    def __init__(self, target_fs=250, l_freq=0.5, h_freq=100.0, notch_freq=60.0, normalize_method='z_score', time_length=4, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
