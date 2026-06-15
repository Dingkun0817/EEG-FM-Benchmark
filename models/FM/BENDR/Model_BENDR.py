import copy

import numpy as np
import torch

from torch import nn
from math import ceil


class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class BENDR(torch.nn.Module):

    def __init__(self, channels, encoder_h=512, contextualizer_hidden=3076, projection_head=False,
                  dropout=0., layer_drop=0,
                 mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1):
        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        super().__init__()

        encoder = ConvEncoderBENDR(channels, encoder_h=encoder_h, dropout=dropout, projection_head=projection_head)

        contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=contextualizer_hidden, finetuning=True,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, layer_drop=layer_drop,
                                                  mask_c_span=mask_c_span, dropout=dropout,
                                                  mask_t_span=mask_t_span)

        self.encoder = encoder
        self.contextualizer = contextualizer


    def forward(self, x):
        encoded = self.encoder(x)
        context = self.contextualizer(encoded)
        return context[:, :, -1]

    def load_pretrained_modules(self, encoder_file, contextualizer_file,strict=False):
        
        self.encoder.load(encoder_file, strict=strict)
        self.contextualizer.load(contextualizer_file, strict=strict)

class ConvEncoderBENDR(nn.Module):
    def __init__(self, in_features, encoder_h=512, enc_width=(3, 2, 2, 2, 2, 2),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features
        
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        enc_width = [e if e % 2 else e+1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                nn.Dropout1d(dropout),
                nn.GroupNorm(encoder_h // 2, encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h

    def forward(self, x):
        return self.encoder(x)

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples
    
    def load(self, filename, strict=True):
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict, strict=strict)

class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BENDRContextualizer(nn.Module):

    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8, dropout=0.15, activation='gelu',
                 position_encoder=25, layer_drop=0.0, mask_p_t=0.1, mask_p_c=0.004, mask_t_span=6, mask_c_span=64,
                 start_token=-5, finetuning=False):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3

        encoder = nn.TransformerEncoderLayer(d_model=in_features * 3, nhead=heads, dim_feedforward=hidden_feedforward,
                                             dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()

        self.norm = nn.LayerNorm(self._transformer_dim)

        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        self.mask_replacement = torch.nn.Parameter(
            torch.normal(0, in_features ** (-0.5), size=(in_features,)),
            requires_grad=True,
        )

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape
        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(x.device).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))
    
    def load(self, filename, strict=True):
        state_dict = torch.load(filename, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict, strict=strict)

