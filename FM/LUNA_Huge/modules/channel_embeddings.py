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
from torcheeg.datasets.constants import SEED_CHANNEL_LIST
import mne 

SEED_PRETRAINING_CHANNEL_LIST = SEED_CHANNEL_LIST
TUEG_CHANNEL_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "A1-T3",
    "T4-A2",
]
SIENA_CHANNEL_LIST = [
    "FP1",
    "FP2",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "FC1",
    "FC5",
    "CP1",
    "CP5",
    "F9",
    "FZ",
    "CZ",
    "PZ",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T4",
    "T6",
    "FC2",
    "FC6",
    "CP2",
    "CP6",
    "F10",
]

all_channels = set()
for ds in [
    SEED_PRETRAINING_CHANNEL_LIST,
    TUEG_CHANNEL_LIST,
    SIENA_CHANNEL_LIST,
]:
    for ch in ds:
        all_channels.add(ch)
CHANNEL_NAMES_TO_IDX = {ch: i for i, ch in enumerate(sorted(all_channels))}
CHANNEL_IDX_TO_NAMES = {i: ch for ch, i in CHANNEL_NAMES_TO_IDX.items()}

def get_channel_indices(channel_names):
    indices = []
    for name in channel_names:
        indices.append(CHANNEL_NAMES_TO_IDX[name])
    return indices

def get_channel_names(channel_indices):
    names = []
    for idx in channel_indices:
        names.append(CHANNEL_IDX_TO_NAMES[idx])
    return names

def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list(set([part for ch in channel_names for part in ch.split('-')]))
    else:
        names = channel_names
    ch_types = ['eeg'] * len(names)  # Channel types
    info = mne.create_info(ch_names=names, sfreq=256, ch_types=ch_types)
    info = info.set_montage(mne.channels.make_standard_montage("standard_1005"),match_case=False,match_alias={'cb1': 'POO7', 'cb2': 'POO8'})
    locs = []
    for name in channel_names:
        if name in TUEG_CHANNEL_LIST:
            electrode1, electrode2 = name.split('-')
            loc1 = info.get_montage().get_positions()['ch_pos'][electrode1]
            loc2 = info.get_montage().get_positions()['ch_pos'][electrode2]
            locs.append(((loc1 + loc2) / 2))
        else:
            locs.append(info.get_montage().get_positions()['ch_pos'][name])
    return locs

class ChannelEmbeddings(nn.Module):
    def __init__(self, embed_dim):
        super(ChannelEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(len(CHANNEL_NAMES_TO_IDX), embed_dim)

    def forward(self, indices):
        return self.embeddings(indices)
    
    def initialize_weights(self):
        torch.init.normal_(self.embeddings.weight, std=2.0)

