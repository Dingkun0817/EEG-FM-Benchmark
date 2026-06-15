import os
import torch
import torch.nn as nn
from utils.EEGDataLoader import EEGData
from utils.utils import LinearLayers, RegressionLayers

class ModelLoader(nn.Module):
    """Base class for model loaders: channel/time config, task head, and optional pretrained weights."""
    def __init__(self, eeg_dataset, finetune_strategy=None):
        super().__init__()
        if not isinstance(eeg_dataset, EEGData):
            raise TypeError("eeg_dataset must be an EEGData instance")
        self.ch_names = eeg_dataset.channel_names
        self.num_channels = len(self.ch_names)
        self.num_time_points = eeg_dataset.get_time_point_count()
        self.num_time = eeg_dataset.get_duration()
        self.sampling_rate = eeg_dataset.sampling_rate
        self.dataset_name = getattr(eeg_dataset, 'dataset_name', 'unknown')
        self.dataset_type = getattr(eeg_dataset, 'dataset_type', 'classification')
        if self.dataset_type == 'regression':
            self.nb_classes = 1
            self.task_type = 'regression'
        elif self.dataset_type == 'classification':
            self.nb_classes = eeg_dataset.get_label_count() if hasattr(eeg_dataset, 'labels') and eeg_dataset.labels is not None else 1
            self.task_type = 'classification'
        elif self.dataset_type == 'matching':
            self.nb_classes = eeg_dataset.get_label_count() if hasattr(eeg_dataset, 'labels') and eeg_dataset.labels is not None else 1
            self.img_feature_dim = eeg_dataset.img_feature.shape[1]
            self.task_type = 'matching'
        self.finetune_strategy = finetune_strategy
        self.main_model = None
        self.task_head = nn.Identity()
        self.feature_dim = None
    
    def set_task_head(self, task_type, num_classes=None, dropout_rate=0.1):
        """Set task head for regression, classification, or matching; requires feature_dim to be set. Returns self."""
        if not hasattr(self, 'feature_dim') or self.feature_dim is None:
            raise AttributeError("feature_dim must be set before calling set_task_head")
        if task_type == 'regression':
            self.task_head = RegressionLayers(self.feature_dim, 512, 1, flatten=1, dropout_rate=dropout_rate)
        elif task_type == 'classification':
            if num_classes is None:
                raise ValueError(f"{task_type} task requires num_classes")
            self.task_head = LinearLayers(self.feature_dim, num_classes, flatten=1, dropout_rate=dropout_rate)
        elif task_type == 'matching':
            self.task_head = LinearLayers(self.feature_dim, self.img_feature_dim, flatten=1, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression', 'classification', or 'matching'")
        return self

    def apply_finetune_strategy(self, strategy=None):
        """Set parameter trainability by strategy: 'full' or 'head_only'. Returns self."""
        strategy = strategy or self.finetune_strategy
        trainable_dtypes = {torch.float, torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128}
        
        def set_params_requires_grad(params, requires_grad):
            for param in params:
                if param.dtype in trainable_dtypes:
                    param.requires_grad = requires_grad
        
        if strategy == 'full' or strategy is None:
            set_params_requires_grad(self.parameters(), True)
        elif strategy == 'head_only':
            if self.main_model is not None:
                set_params_requires_grad(self.main_model.parameters(), False)
            if self.task_head is not None:
                set_params_requires_grad(self.task_head.parameters(), True)
            if hasattr(self, 'chan_conv'):
                set_params_requires_grad(self.chan_conv.parameters(), True)
        else:
            raise ValueError(f"Unsupported finetune strategy: {strategy}. Use 'full' or 'head_only'")
        return self

    def load_pretrained_weights(self, pretrained_path, strict=True):
        """Load pretrained weights into main_model; strip 'module.' prefix if present. Returns True on success."""
        if not os.path.exists(pretrained_path):
            print(f"Warning: pretrained weights file not found: {pretrained_path}")
            return False
        print(f"Loading pretrained weights: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
        if 'module.' in next(iter(state_dict.keys())):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.main_model.load_state_dict(state_dict, strict=strict)
        return True
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")
    def get_model_info(self):
        return {
            'model_name': self.__class__.__name__,
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'num_channels': self.num_channels,
            'num_time_points': self.num_time_points,
            'sampling_rate': self.sampling_rate,
            'nb_classes': self.nb_classes,
            'finetune_strategy': self.finetune_strategy
        }