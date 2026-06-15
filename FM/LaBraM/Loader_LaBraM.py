import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from timm.models import create_model
from models.FM.LaBraM import Model_LaBraM
from utils.ModelLoader import ModelLoader
from utils.utils import get_input_chans, LinearLayers, RegressionLayers
from utils.preprocessing import Preprocessor



class Loader_LaBraM(ModelLoader):
    """
    LaBraM model loader for EEGData inputs.
    """
    def __init__(self, eeg_dataset, **kwargs):
        """
        Initialize the Loader_LaBraM model.
        
        Args:
            eeg_dataset: An EEGData object containing EEG samples, channel names,
                and all required dataset metadata.
            **kwargs: Additional optional parameters.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'flatten')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)

        if eeg_dataset.dataset_name == 'CHB_MIT':
           self.num_channels = self.num_channels - 16

        self.readout = readout
        if self.readout == 'flatten':
            # (1 + num_channels * num_segments) tokens * 200-dim
            self.feature_dim = self.num_channels * self.num_time_points + 200
        else:
            # mean pooled representation is 200-dim
            self.feature_dim = 200

        if eeg_dataset.dataset_name == 'CHB_MIT':
            self.input_chans = get_input_chans(self.ch_names[:-16])
        else:
            self.input_chans = get_input_chans(self.ch_names)
        

            
        # Load LaBraM model
        model = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=self.nb_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=True,
            num_t=self.num_time_points
        )
        
     
        self.main_model = model
        
        from utils.utils import get_pretrained_models_path
        pretrained_path = get_pretrained_models_path('labram-base.pth')
        self.load_pretrained_weights(pretrained_path, strict=True)
        
        self.main_model.head = nn.Identity()

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)
        

        self.apply_finetune_strategy()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input EEG tensor with shape (batch_size, num_channels, num_time_points).
        
        Returns:
            Model output features.
        """
        b, n, t = x.shape

        segment_size = 200
        num_segments = t // segment_size
        
        # Reshape into segment blocks expected by LaBraM
        x = x.reshape(b, n, num_segments, segment_size)
        
        # Use precomputed input channel indices directly
        input_chans = self.input_chans
        
        # Forward pass through backbone model
        output = self.main_model(x, input_chans, return_all_tokens=(self.readout == 'flatten'))

        output = self.task_head(output)
        
        return output


    def load_pretrained_weights(self, pretrained_path, strict=True):
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        checkpoint_model = None
        MODEL_KEYS = 'model|module'
        for model_key in MODEL_KEYS.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        
        if checkpoint_model is not None:
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = self.main_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        _load_state_dict(self.main_model, checkpoint_model)
         


class EEGPreprocessor_LaBraM(Preprocessor):
    """LaBraM preprocessor with defaults: target_fs=200, l_freq=0.1, h_freq=75, notch_freq=50."""
    def __init__(self, target_fs=200, l_freq=0.1, h_freq=75, notch_freq=50, normalize_method="0.1mv", time_length=4, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)

    def preprocess(self, eeg_data, task_mode='Cross', train_percentage=0.3, **normalize_kwargs):
        data = super().preprocess(eeg_data, task_mode, train_percentage, **normalize_kwargs)
        if eeg_data.dataset_name == 'CHB_MIT':
            data = data[:, :data.shape[1]-16, :]
        return data



def _load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    """Load state dict into model, ignoring missing keys that match ignore_missing (e.g. relative_position_index)."""
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
