"""Shared model layer components (classification/regression heads, constrained conv)."""
import torch
import torch.nn as nn


class LinearLayers(nn.Sequential):
    def __init__(self, input_dim, output_dim, flatten=0, dropout_rate=0.1):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim),
        )
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        return self.clshead(x)


class RegressionLayers(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, flatten=0, patch_mean=False, remove_cls=False, dropout_rate=0.1):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.remove_cls = remove_cls

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        return self.clshead(x)


class Conv1dWithConstraint(nn.Conv1d):
    """Conv1d with optional weight renormalization (max norm). From EEGNet."""
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv1dWithConstraint, self).forward(x)
