"""
Compatibility layer: re-exports from split submodules so existing
`from utils.utils import ...` imports continue to work unchanged.
"""
import os

from .dataset_split import split_dataset_cross, split_dataset_fewshot, create_dataloaders
from .channels import standard_1020, get_input_chans
from .model_layers import LinearLayers, RegressionLayers, Conv1dWithConstraint


def get_pretrained_models_path(*path_parts):
    """Return path under models/pretrained_models/. path_parts e.g. ('encoder.pt') or ('BrainOmni', 'tiny')."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    return os.path.join(project_root, 'models', 'pretrained_models', *path_parts)


__all__ = [
    'split_dataset_cross',
    'split_dataset_fewshot',
    'create_dataloaders',
    'standard_1020',
    'get_input_chans',
    'LinearLayers',
    'RegressionLayers',
    'Conv1dWithConstraint',
    'get_pretrained_models_path',
]
