import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import numpy as np
from models.DL.TSception.Model_TSception import TSception
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


def generate_TS_channel_order(original_order: list):
    """
    Channel order for TSception asymmetric spatial paths (from TSception reference utils).
    Only names with a trailing numeric suffix participate.
    """
    chan_name, chan_num, chan_final = [], [], []
    for channel in original_order:
        match = re.match(r"^(.*?)(\d+)$", channel)
        if match is None:
            continue
        chan_name.append(match.group(1))
        chan_num.append(int(match.group(2)))
        chan_final.append(channel)
    chan_pair = []
    for ch, id in enumerate(chan_num):
        if id % 2 == 0:
            chan_pair.append(chan_name[ch] + str(id - 1))
        else:
            chan_pair.append(chan_name[ch] + str(id + 1))
    chan_no_duplicate = []
    [
        chan_no_duplicate.extend([f, chan_pair[i]])
        for i, f in enumerate(chan_final)
        if f not in chan_no_duplicate
    ]
    return chan_no_duplicate[0::2] + chan_no_duplicate[1::2]


def _tsception_reorder_indices(original_order: list):
    """
    Map dataset channels to TSception order when possible (subset / missing-name safe).
    If TS order cannot be built or matches no channel, return identity order (default layout).
    """
    original_order = list(original_order)
    if not original_order:
        raise ValueError("TSception: channel_names is empty.")
    identity_idx = list(range(len(original_order)))
    ts_order = generate_TS_channel_order(original_order)
    if not ts_order:
        return identity_idx, list(original_order)

    idx = []
    ordered_names = []
    for chan in ts_order:
        if chan not in original_order:
            continue
        idx.append(original_order.index(chan))
        ordered_names.append(chan)
    if not ordered_names:
        return identity_idx, list(original_order)

    return idx, ordered_names


class Loader_TSception(ModelLoader):
    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get("finetune_strategy", None)
        dropout_rate = kwargs.get("dropout_rate", 0.5)
        num_T = kwargs.get("num_T", 15)
        num_S = kwargs.get("num_S", None)
        hidden = kwargs.get("hidden", 32)
        if num_S is None:
            num_S = num_T
        super().__init__(eeg_dataset, finetune_strategy)
        if self.task_type == "matching":
            self.nb_classes = eeg_dataset.img_feature.shape[1]
        input_size = (1, self.num_channels, self.num_time_points)
        self.main_model = TSception(
            num_classes=self.nb_classes,
            input_size=input_size,
            sampling_rate=float(self.sampling_rate),
            num_T=num_T,
            num_S=num_S,
            hidden=hidden,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.main_model(x)


class EEGPreprocessor_TSception(Preprocessor):
    """
    Try TSception channel reorder/subset first when naming allows; otherwise keep original
    channel order. Then the standard Preprocessor pipeline (resample, trim/pad, etc.).
    Defaults: target_fs=128 Hz, time_length=4 s (per TSception README example).
    """

    def __init__(
        self,
        target_fs=128,
        l_freq=None,
        h_freq=None,
        notch_freq=None,
        normalize_method=None,
        time_length=4,
        apply_EA=False,
    ):
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA,
        )

    def preprocess(
        self, eeg_data, task_mode="Cross", train_percentage=0.3, **kwargs
    ):
        idx, ordered_names = _tsception_reorder_indices(list(eeg_data.channel_names))
        data = np.asarray(eeg_data.eeg_data)
        if data.ndim != 3:
            raise ValueError(f"TSception: expected eeg_data (N,C,T), got shape {data.shape}")
        if data.shape[1] != len(eeg_data.channel_names):
            raise ValueError(
                "TSception: eeg_data channel dimension does not match len(channel_names)."
            )
        eeg_data.eeg_data = data[:, idx, :]
        eeg_data.channel_names = ordered_names
        return super().preprocess(
            eeg_data,
            task_mode=task_mode,
            train_percentage=train_percentage,
            **kwargs,
        )
