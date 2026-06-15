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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
from timm.models.layers import Mlp

class FrequencyFeatureEmbedder(nn.Module):
    """
    This class takes data that is of the form (B, C, T) and patches it 
    along the time dimension (T) into patches of size P (patch_size).
    The output is of the form (B, C, S, P) where S = T // P.
    """
    def __init__(self, patch_size, embed_dim):
        super(FrequencyFeatureEmbedder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        in_features = 2*(patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(in_features=in_features, hidden_features=int(4*in_features), out_features=embed_dim)

    def forward(self, x):
        B, C, T = x.size()
        S = T // self.patch_size
        # There is a chance that the input tensor is not divisible by the patch size
        # In this case we need to pad the tensor with zeros
        if T % self.patch_size != 0:
            # Pad last dimension with zeros to make it divisible by patch size
            pad_size = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad_size))
            T = x.size(-1)
            S = T // self.patch_size   
        x = x.view(B, C, S, self.patch_size)

        freq_representation = fft.rfft(x, dim=-1)  # (B, C, num_patches, patch_size // 2 + 1)
        magnitude = torch.abs(freq_representation)
        phase = torch.angle(freq_representation)    
        
        # Concatenate magnitude and phase along the frequency axis (last dimension)
        freq_features = torch.cat((magnitude, phase), dim=-1)
        # Map frequency features to embedding dimension
        embedded = self.frequency_to_embed(freq_features)  # (B, C, num_patches, embed_dim)
        embedded = rearrange(embedded, 'B C t D -> B (C t) D')
        return embedded
