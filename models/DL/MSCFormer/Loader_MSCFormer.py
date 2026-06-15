import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from models.DL.MSCFormer.Model_MSCFormer import MSCFormer, Parameters
from utils.ModelLoader import ModelLoader
from utils.preprocessing import Preprocessor


class Loader_MSCFormer(ModelLoader):
    """MSCFormer loader."""

    def __init__(self, eeg_dataset, **kwargs):
        finetune_strategy = kwargs.get("finetune_strategy", None)
        dropout_rate = kwargs.get("dropout_rate", 0.5)
        super().__init__(eeg_dataset, finetune_strategy)
        if self.task_type == "matching":
            self.nb_classes = eeg_dataset.img_feature.shape[1]

        params = Parameters(dropout_rate=dropout_rate)
        self.main_model = MSCFormer(
            parameters=params,
            class_num=self.nb_classes,
            chn=self.num_channels,
        )

    def forward(self, x):
        # Input: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        return self.main_model(x)


class EEGPreprocessor_MSCFormer(Preprocessor):
    """Preprocessor for MSCFormer: default target_fs=250, l_freq=8, h_freq=32."""

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
