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

class ChannelLocationEmbedder(nn.Module):
    def __init__(
        self,
        channel_locations_dim: int = 3,
        in_chans: int = 22,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.channel_locations_dim = channel_locations_dim
        self.embed_dim = embed_dim
        self.channel_embeddings = nn.ModuleList([
                nn.Linear(channel_locations_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            ])
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, channel_locations):
        """
        Output shape:
            [in_chans, embed_dim]
        Args:
            channel_locations: [B, in_chans, channel_locations_dim]
        """
        
        out = channel_locations
        for layer in self.channel_embeddings:
            out = layer(out)
        return out
