#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Berkay Döner                                                      *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import os
import sys

# Get the project root before importing custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(project_root)
import math
from functools import partial
from typing import List
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import Mlp

from einops import rearrange
from .modules.rope_transformer_encoder_block import RotaryTransformerBlock
from .modules.frequency_embedder import FrequencyFeatureEmbedder
from .modules.channel_embeddings import ChannelEmbeddings

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    """
    coords: (N, C, 3)
    Returns: (N, C, embed_size)
    """
    N, C, dim = coords.shape
    device = coords.device
    freqs = embed_size // (2 * dim)
    leftover = embed_size - freqs * 2 * dim
    freq_bands = 2.0 ** torch.arange(freqs, device=device).float()
    scaled_coords = coords.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1) # (N, C, dim, freqs)
    sin_enc = torch.sin(scaled_coords) # (N, C, dim, freqs)
    cos_enc = torch.cos(scaled_coords) # (N, C, dim, freqs)
    encoded = torch.stack([sin_enc, cos_enc], dim=-1).permute(0, 1, 3, 2, 4).reshape(N, C, freqs * dim * 2)
    if leftover > 0:
        pad = torch.zeros(N, C, leftover, device=device, dtype=coords.dtype)
        encoded = torch.cat([encoded, pad], dim=-1)
    return encoded

class PatchReconstructionHeadWithQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_queries: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim      
        self.reconstruction_shape = self.input_dim
        self.num_queries = num_queries
        # Projection from embed space to pixel space, according to type of input
        self.decoder_pred = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, dropout=0.0, batch_first=True, activation='gelu', dim_feedforward=int(embed_dim*4), norm_first=True),
            num_layers=1
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_linear = Mlp(embed_dim, int(embed_dim*4), input_dim, act_layer=nn.GELU, drop=0.0) #nn.Linear(embed_dim, input_dim, bias=True)
    
    def forward(self, enc, decoder_queries):
        """
        enc: [B, num_patches, embed_dim], embed_dim = Q*D
        decoder_queries: [B*num_patches, num_channels, embed_dim]
        """
        
        B, num_patches, embed_dim = enc.shape
        enc = rearrange(enc, 'B t (Q D) -> (B t) Q D', Q=self.num_queries)
        out = self.decoder_pred(decoder_queries, enc)  # (B*t, C, D)
        out = self.norm(out)
        out = self.decoder_linear(out) # (B*t, C, patch_size)
        out = rearrange(out, '(B t) C P -> B C (t P)', B=B)
        return out

class ClassificationHeadWithQueries(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 768,
        num_queries: int = 8,
        num_heads: int = 8,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(embed_dim*num_queries)  
        self.reconstruction_shape = self.input_dim
        self.decoder_attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=0.15)
        self.decoder_ffn = Mlp(in_features=self.embed_dim, hidden_features=int(self.embed_dim*4), out_features=self.embed_dim, act_layer=nn.GELU, drop=0.15)
    
        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.embed_dim), requires_grad=True)
    
    def forward(self, x):
        """
        Output shape:
            [B, num_tokens, in_chans, input_dim]
        Args:
            x: [B, num_tokens+1, embed_dim]
            channel_embeddings: [B, in_chans, embed_dim]
        """
        B, num_patches, embed_dim = x.shape
        decoder_queries = self.learned_agg.repeat(x.shape[0], 1, 1)

        x = self.decoder_attn(query=decoder_queries, key=x, value=x)[0]
        x = x[:,0,:]
        x = self.decoder_ffn(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, num_queries, input_embed_dim, output_embed_dim, num_heads, dropout_p=0.1, ff_dim=2048, pre_norm=True):
        super(CrossAttentionBlock, self).__init__()
        self.num_queries = num_queries
        self.dropout_p = dropout_p
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, input_embed_dim), requires_grad=True)  # Learnable queries
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_embed_dim, num_heads=num_heads, dropout=dropout_p,batch_first=True)
        self.temparature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.ffn = Mlp(input_embed_dim, ff_dim, output_embed_dim, act_layer=nn.GELU, drop=dropout_p, norm_layer=nn.LayerNorm)
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.query_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_embed_dim, nhead=num_heads, activation='gelu', dim_feedforward=ff_dim, batch_first=True, norm_first=True), num_layers=3)

    def initialize_weights(self):
        torch.nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
       
    def forward(self, x):
        # x is the input with shape (batch_size*num_patches, num_channels, embed_dim)
        batch_size, num_channels, _ = x.size()
        queries = self.query_embed.repeat(batch_size,1,1)
        queries = self.queries_norm(queries)
        keys = self.keys_norm(x)
        values = self.values_norm(x)

        attention_out, attention_scores = self.cross_attention(query=queries,key=keys,value=values) # Shape: (batch_size*num_patches, num_queries, embed_dim)
        attention_out = self.ffn(attention_out) + attention_out
   
        attention_out = self.query_self_attn(attention_out)
        return attention_out, attention_scores  # Shape: (batch_size*num_patches, num_queries, embed_dim)

class PatchEmbedNetwork(nn.Module):
    def __init__(self, embed_dim=64, patch_size=40):
        super(PatchEmbedNetwork, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = 1
        self.out_channels = int(embed_dim//4)
        self.groups = 4
        self.kernel_size = int(patch_size//2)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, self.kernel_size-1), stride=(1, self.kernel_size//2), padding=(0, self.kernel_size//2-1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),

            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(self.groups, self.out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        """
            x: (B, C, T)
            output: (B, C*S, D) where S = T//patch_size, D = embed_dim
        """
        x = rearrange(x, 'B C (S P) -> B (C S) P', P=self.patch_size)
        x = x.unsqueeze(1)
        x = self.proj_in(x)
        x = rearrange(x, 'B E CS D -> B CS (D E)')
        return x

class LUNA(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=40, num_queries=8,
                 embed_dim=128, depth=24, num_heads=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                drop_path=0.0, num_classes=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.depth = depth
        self.patch_embed = PatchEmbedNetwork(embed_dim=self.embed_dim, patch_size=patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(embed_dim=self.embed_dim, patch_size=patch_size)
        self.channel_location_embedder = nn.Sequential(
            Mlp(in_features=int(self.patch_embed_size), out_features=int(self.patch_embed_size), hidden_features=int(self.patch_embed_size*2), act_layer=nn.GELU, drop=0.0, norm_layer=nn.LayerNorm),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cross_attn = CrossAttentionBlock(num_queries=num_queries, input_embed_dim=self.embed_dim, output_embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=int(mlp_ratio*self.embed_dim), pre_norm=True)
        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(dim=int(self.embed_dim*self.num_queries), num_heads=int(self.num_heads*self.num_queries), mlp_ratio=mlp_ratio, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(int(self.embed_dim*self.num_queries))
        if num_classes==0: # reconstruction (pre-training)
            self.decoder_head = PatchReconstructionHeadWithQueries(input_dim=patch_size, embed_dim=self.embed_dim, num_heads=self.num_heads, num_queries=num_queries)
            self.channel_emb = ChannelEmbeddings(self.embed_dim)
        else: # classification
            self.classifier = ClassificationHeadWithQueries(input_dim=patch_size, num_queries=num_queries, embed_dim=self.embed_dim, num_classes=num_classes, num_heads=self.num_heads)
            self.mask_token.requires_grad = False # no use of mask token for classification
        self.initialize_weights()

    def initialize_weights(self):
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def prepare_tokens(self, x_signal, channel_locations, mask=None):
        num_channels = channel_locations.shape[1]
        num_patches_per_channel = x_signal.shape[-1] // self.patch_size
        x_patched = self.patch_embed(x_signal) 
        freq_embed = self.freq_embed(x_signal)
        x_patched = x_patched + freq_embed
        x_masked = x_patched.clone() # (B, N, D), N = C * num_patches_per_channel
        if mask is not None:
            mask_tokens = self.mask_token.repeat(x_masked.shape[0], x_masked.shape[1], 1) # (B, N, D) N = C * num_patches_per_channel
            mask = rearrange(mask, 'B C (S P) -> B (C S) P', P=self.patch_size) # (B, C, T) -> (B, N, P)
            mask = (mask.sum(dim=-1) > 0).unsqueeze(-1).float() # (B, N, 1), since a patch is either fully masked or not
            x_masked = torch.where(mask.bool(), mask_tokens, x_masked)
        channel_min = torch.min(channel_locations, dim=1, keepdim=True)[0]
        channel_max = torch.max(channel_locations, dim=1, keepdim=True)[0]
        channel_locations = (channel_locations - channel_min) / (channel_max - channel_min + 1e-8)
        if mask is not None:
            channel_locations = channel_locations + torch.randn_like(channel_locations) * 0.02
        channel_locations = nerf_positional_encoding(channel_locations, self.patch_embed_size)
        channel_locations_emb = self.channel_location_embedder(channel_locations)

        x_tokenized = rearrange(x_masked, 'B (C t) D -> (B t) C D', C=num_channels)
        channel_locations_emb = channel_locations_emb.repeat(num_patches_per_channel, 1, 1)
        x_tokenized = x_tokenized + channel_locations_emb
        
        return x_tokenized, channel_locations_emb

    def forward(self, x_signal, mask, channel_locations, channel_names=None):
        x_original = x_signal
        B, C, T = x_signal.shape
        if not isinstance(channel_locations, torch.Tensor):
            channel_locations = torch.tensor(channel_locations, device=x_signal.device)
        else:
            channel_locations = channel_locations.to(x_signal.device)

        # Expand dimensions: (22, 3) -> (1, 22, 3) -> (B, 22, 3)
        # Only expand 2D inputs to avoid repeated expansion errors
        if channel_locations.dim() == 2:
            channel_locations = channel_locations.unsqueeze(0).expand(B, -1, -1)
        x, channel_locations_emb = self.prepare_tokens(x_signal, channel_locations, mask=mask)        
        
        x, attention_scores = self.cross_attn(x) # (B*num_patches, Q, D)
        x = rearrange(x, '(B t) Q D -> B t (Q D)', B=B) # (B, num_patches, Q*D), Q*D is the new embed_dim
        num_patches = x.shape[1]
        for blk in self.blocks:
            x = blk(x) # (B, N, D)
        x_latent = self.norm(x) # (B, N, D)

        if self.num_classes > 0:
            x_classified = self.classifier(x_latent)
            return x_classified
        else:
            channel_emb = self.channel_emb(channel_names)
            channel_emb = channel_emb.repeat(num_patches, 1, 1)
            decoder_queries = channel_locations_emb + channel_emb # (B*N, C, D)
            x_reconstructed = self.decoder_head(x_latent, decoder_queries) 
            return x_reconstructed, x_original, attention_scores
