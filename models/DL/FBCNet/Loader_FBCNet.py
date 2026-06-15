import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
from models.DL.FBCNet.Model_FBCNet import FBCNet_2
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_FBCNet(ModelLoader):
    """FBCNet loader for classification/regression/matching tasks."""

    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get("finetune_strategy", None)
        super().__init__(eeg_dataset, finetune_strategy)
        if self.task_type == "matching":
            self.nb_classes = eeg_dataset.img_feature.shape[1]

        m = kwargs.get("m", 32)
        temporal_stride = kwargs.get("temporal_stride", 4)
        weight_init_method = kwargs.get("weight_init_method", None)

        self.main_model = FBCNet_2(
            n_classes=self.nb_classes,
            input_shape=(1, 1, self.num_channels, self.num_time_points),
            m=m,
            temporal_stride=temporal_stride,
            weight_init_method=weight_init_method,
        )

    def forward(self, x):
        # Input: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        return self.main_model(x)


class EEGPreprocessor_FBCNet(Preprocessor):
    """Preprocessor for FBCNet: default target_fs=250, l_freq=8, h_freq=32."""

    def __init__(
        self,
        target_fs=250,
        l_freq=8.0,
        h_freq=32.0,
        notch_freq=None,
        normalize_method="car",
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
