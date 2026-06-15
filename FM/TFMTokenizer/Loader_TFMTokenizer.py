import os
import sys
from einops import rearrange

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
from models.FM.TFMTokenizer.Model_TFMTokenizer import TFM_TOKEN_Classifier, get_stft_torch, get_tfm_tokenizer_2x2x8
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor
from utils.utils import get_pretrained_models_path, Conv1dWithConstraint


class Loader_TFMTokenizer(ModelLoader):
    """TFM-Tokenizer downstream classifier with frozen VQ-VAE tokenizer."""

    def __init__(self, eeg_dataset, **kwargs):
        """
        Args:
            eeg_dataset: EEGData instance.
            **kwargs: finetune_strategy, dropout_rate; readout ('flatten' or 'pooling', default 'pooling');
                optional model hyperparameters
                (emb_size, code_book_size, num_heads, depth, max_seq_len) are fixed in code.
        """
        finetune_strategy = kwargs.get('finetune_strategy', None)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        readout = kwargs.get('readout', 'pooling')
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))
        super().__init__(eeg_dataset, finetune_strategy)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.readout = readout

        emb_size = 64
        code_book_size = 8192

        self.vqvae = get_tfm_tokenizer_2x2x8(code_book_size=code_book_size, emb_size=emb_size)
        tfm_tokenizer_pretrained_path = get_pretrained_models_path('tfm_tokenizer_last.pth')
        self.vqvae.load_state_dict(torch.load(tfm_tokenizer_pretrained_path, map_location=self.device, weights_only=False))
        self.vqvae.to(self.device)
        self.vqvae.eval()

        self.main_model = TFM_TOKEN_Classifier(
            emb_size=emb_size,
            code_book_size=code_book_size,
            num_heads=8,
            depth=4,
            max_seq_len=2048,
            n_classes=self.nb_classes)

        tfm_tokenizer_pretrained_path = get_pretrained_models_path('tfm_encoder_mtp_last.pth')
        checkpoint = torch.load(tfm_tokenizer_pretrained_path, map_location=self.device, weights_only=False)
        filtered_checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if "classification_head" not in key
        }
        self.main_model.load_state_dict(filtered_checkpoint, strict=False)
        self.main_model.to(self.device)

        self.main_model.classification_head = nn.Identity()

        self.chan_conv = Conv1dWithConstraint(self.num_channels, 16, 1, max_norm=1).to(self.device)

        with torch.no_grad():
            x = torch.randn(1, self.num_channels, self.num_time_points, device=self.device)
            x = self.chan_conv(x)
            x_temporal = x
            B, C, T = x_temporal.shape
            x_stft = get_stft_torch(x_temporal, resampling_rate=200)
            x_stft = rearrange(x_stft, 'B C F T -> (B C) F T').to(x_temporal.device)
            x_temporal_flat = rearrange(x_temporal, 'B C T -> (B C) T')
            _, x_tokens, _ = self.vqvae.tokenize(x_stft, x_temporal_flat)
            x_tokens = rearrange(x_tokens, '(B C) T -> B C T', C=C)
            o0 = self.main_model(x_tokens, num_ch=C, readout=self.readout)
            self.feature_dim = int(o0.numel() // o0.shape[0])

        self.set_task_head(self.dataset_type, self.nb_classes, dropout_rate=dropout_rate)

        self.apply_finetune_strategy()

    def forward(self, x):
        x = self.chan_conv(x)
        x_temporal = x
        B, C, T = x_temporal.shape
        x = get_stft_torch(x_temporal, resampling_rate=200)
        x = rearrange(x, 'B C F T -> (B C) F T').to(x_temporal.device)
        x_temporal = rearrange(x_temporal, 'B C T -> (B C) T')

        with torch.no_grad():
            _, x_tokens, _ = self.vqvae.tokenize(x, x_temporal)
        x_tokens = rearrange(x_tokens, '(B C) T -> B C T', C=C)

        out = self.main_model(x_tokens, num_ch=C, readout=self.readout)

        pred = self.task_head(out)
        return pred


class EEGPreprocessor_TFMTokenizer(Preprocessor):
    """Preprocessor for TFM-Tokenizer: default target_fs=200, l_freq=0.1, h_freq=75, notch_freq=50."""

    def __init__(self, target_fs=200, l_freq=0.1, h_freq=75, notch_freq=50, normalize_method=None, time_length=None, apply_EA=False):
        super().__init__(target_fs=target_fs, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq, normalize_method=normalize_method, time_length=time_length, apply_EA=apply_EA)
