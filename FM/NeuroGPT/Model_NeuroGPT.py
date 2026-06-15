# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
import warnings

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

from collections import OrderedDict

import numpy as np
import torch
from torchinfo import ModelStatistics, summary


def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f'{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead.'
            )
            if new_val is not None:
                raise ValueError(
                    f'{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified.'
                )
            out_args.append(old_val)
    return out_args


class EEGModuleMixin():
    """
    Mixin class for all EEG models in braindecode.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes
        in the case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about each individual EEG channel. This should be filled with
        ``info["chs"]``. Refer to :class:`mne.Info` for more details.
    n_times : int
        Number of time samples of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.
    add_log_softmax: bool
        Whether to use log-softmax non-linearity as the output function.
        LogSoftmax final layer will be removed in the future.
        Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!
        Check the documentation of the torch.nn loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions.

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    FutureWarning: If add_log_softmax is True, since LogSoftmax final layer
                   will be removed in the future.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

    def __init__(
            self,
            n_outputs: Optional[int] = None,
            n_chans: Optional[int] = None,
            chs_info: Optional[List[Dict]] = None,
            n_times: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None,
            add_log_softmax: Optional[bool] = False,
    ):
        if (
                n_chans is not None and
                chs_info is not None and
                len(chs_info) != n_chans
        ):
            raise ValueError(f'{n_chans} different from {chs_info} length')
        if (
                n_times is not None and
                input_window_seconds is not None and
                sfreq is not None and
                n_times != int(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f'{n_times} different from '
                f'{input_window_seconds} * {sfreq}'
            )
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._chs_info = chs_info
        self._n_times = n_times
        self._input_window_seconds = input_window_seconds
        self._sfreq = sfreq
        self._add_log_softmax = add_log_softmax
        super().__init__()

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            raise ValueError('n_outputs not specified.')
        return self._n_outputs

    @property
    def n_chans(self):
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError(
                'n_chans could not be inferred. Either specify n_chans or chs_info.'
            )
        return self._n_chans

    @property
    def chs_info(self):
        if self._chs_info is None:
            raise ValueError('chs_info not specified.')
        return self._chs_info

    @property
    def n_times(self):
        if (
                self._n_times is None and
                self._input_window_seconds is not None and
                self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._n_times is None:
            raise ValueError(
                'n_times could not be inferred. '
                'Either specify n_times or input_window_seconds and sfreq.'
            )
        return self._n_times

    @property
    def input_window_seconds(self):
        if (
                self._input_window_seconds is None and
                self._n_times is not None and
                self._sfreq is not None
        ):
            return self._n_times / self._sfreq
        elif self._input_window_seconds is None:
            raise ValueError(
                'input_window_seconds could not be inferred. '
                'Either specify input_window_seconds or n_times and sfreq.'
            )
        return self._input_window_seconds

    @property
    def sfreq(self):
        if (
                self._sfreq is None and
                self._input_window_seconds is not None and
                self._n_times is not None
        ):
            return self._n_times / self._input_window_seconds
        elif self._sfreq is None:
            raise ValueError(
                'sfreq could not be inferred. '
                'Either specify sfreq or input_window_seconds and n_times.'
            )
        return self._sfreq

    @property
    def add_log_softmax(self):
        if self._add_log_softmax:
            warnings.warn("LogSoftmax final layer will be removed! " +
                          "Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!")
        return self._add_log_softmax

    @property
    def input_shape(self) -> Tuple[int]:
        """Input data shape."""
        return (1, self.n_chans, self.n_times)

    def get_output_shape(self) -> Tuple[int]:
        """Returns shape of neural network output for batch size equal 1.

        Returns
        -------
        output_shape: Tuple[int]
            shape of the network output for `batch_size==1` (1, ...)
    """
        with torch.inference_mode():
            try:
                return tuple(self.forward(
                    torch.zeros(
                        self.input_shape,
                        dtype=next(self.parameters()).dtype,
                        device=next(self.parameters()).device
                    )).shape)
            except RuntimeError as exc:
                if str(exc).endswith(
                        ("Output size is too small",
                         "Kernel size can't be greater than actual input size")
                ):
                    msg = (
                        "During model prediction RuntimeError was thrown showing that at some "
                        f"layer `{str(exc).split('.')[-1]}` (see above in the stacktrace). This "
                        "could be caused by providing too small `n_times`/`input_window_seconds`. "
                        "Model may require longer chunks of signal in the input than "
                        f"{self.input_shape}."
                    )
                    raise ValueError(msg) from exc
                raise exc

    mapping = None

    def load_state_dict(self, state_dict, *args, **kwargs):

        mapping = self.mapping if self.mapping else {}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in mapping:
                new_state_dict[mapping[k]] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)

    def to_dense_prediction_model(self, axis: Tuple[int] = (2, 3)) -> None:
        """
        Transform a sequential model with strides to a model that outputs
        dense predictions by removing the strides and instead inserting dilations.
        Modifies model in-place.

        Parameters
        ----------
        axis: int or (int,int)
            Axis to transform (in terms of intermediate output axes)
            can either be 2, 3, or (2,3).

        Notes
        -----
        Does not yet work correctly for average pooling.
        Prior to version 0.1.7, there had been a bug that could move strides
        backwards one layer.

        """
        if not hasattr(axis, "__len__"):
            axis = [axis]
        assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
        axis = np.array(axis) - 2
        stride_so_far = np.array([1, 1])
        for module in self.modules():
            if hasattr(module, "dilation"):
                assert module.dilation == 1 or (module.dilation == (1, 1)), (
                    "Dilation should equal 1 before conversion, maybe the model is "
                    "already converted?"
                )
                new_dilation = [1, 1]
                for ax in axis:
                    new_dilation[ax] = int(stride_so_far[ax])
                module.dilation = tuple(new_dilation)
            if hasattr(module, "stride"):
                if not hasattr(module.stride, "__len__"):
                    module.stride = (module.stride, module.stride)
                stride_so_far *= np.array(module.stride)
                new_stride = list(module.stride)
                for ax in axis:
                    new_stride[ax] = 1
                module.stride = tuple(new_stride)

    def get_torchinfo_statistics(
            self,
            col_names: Optional[Iterable[str]] = (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
            ),
            row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        return summary(
            self,
            input_size=(1, self.n_chans, self.n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())

class EEGConformer(EEGModuleMixin, nn.Module):
    """EEG Conformer.

    This neural network architecture recieves a traditional braindecode input.
    The input shape should be three-dimensional matrix representing the EEG
    signals.

         `(batch_size, n_channels, n_timesteps)`.

    The EEG Conformer architecture is composed of three modules:
        - PatchEmbedding
        - TransformerEncoder
        - ClassificationHead
    Notes
    -----
    The authors recommend using data augmentation before using Conformer,
    e.g. sementation and recombination,
    Please refer to the original paper and code for more details.

    The model was initially tuned on 4 seconds of 250 Hz data.
    Please adjust the scale of the temporal convolutional layer,
    and the pooling layer for better performance.

    .. versionadded:: 0.8

    We aggregate the parameters based on the parts of the models, or
    when the parameters were used first, e.g. n_filters_time.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    n_classes :
        Alias for n_outputs.
    n_channels :
        Alias for n_chans.
    input_window_samples :
        Alias for n_times.
    References
    ---------------
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
            self,
            n_outputs=4,
            n_chans=None,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=0.5,
            att_depth=6,
            att_heads=10,
            att_drop_prob=0.5,
            final_fc_length="auto",
            return_features=False,
            n_times=None,
            chs_info=None,
            input_window_seconds=None,
            sfreq=None,
            n_classes=None,
            n_channels=None,
            input_window_samples=None,
            add_log_softmax=True,
            ch_pos=None,
            is_decoding_mode=False,
    ):
        n_outputs, n_chans, n_times = deprecated_args(
            self,
            ('n_classes', 'n_outputs', n_classes, n_outputs),
            ('n_channels', 'n_chans', n_channels, n_chans),
            ('input_window_samples', 'n_times', input_window_samples, n_times)
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        self.mapping = {
            'classification_head.fc.6.weight': 'final_layer.final_layer.0.weight',
            'classification_head.fc.6.bias': 'final_layer.final_layer.0.bias'
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes, n_channels, input_window_samples
        if not (self.n_chans <= 64):
            warnings.warn("This model has only been tested on no more " +
                          "than 64 channels. no guarantee to work with " +
                          "more channels.", UserWarning)

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob)

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob)
            
        self.ch_pos = ch_pos
        self.is_decoding_mode = is_decoding_mode


    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        batch, chunks, chann, time = x.size()
        x = x.contiguous().view(batch*chunks, chann, time)
        # x = x.permute(0, 2, 1, 3).contiguous().view(batch, chann, -1)

        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return x

    def get_fc_size(self):

        out = self.patch_embedding(torch.ones((1, 1,
                                               self.n_chans,
                                               self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2


class _PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution module to capture local features,
    instead of position embedding.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    n_channels: int
        Number of channels to be used as number of spatial filters.
    pool_time_length: int
        Length of temporal pooling filter.
    stride_avg_pool: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.

    Returns
    -------
    x: torch.Tensor
        The output tensor of the patch embedding layer.
    """

    def __init__(
            self,
            n_filters_time,
            filter_time_length,
            n_channels,
            pool_time_length,
            stride_avg_pool,
            drop_prob,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time,
                      (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time,
                      (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b d_model 1 seq -> b seq d_model"), # no need, because it will be flattened
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x) 
        
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size, expansion=forward_expansion,
                        drop_p=att_drop
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.

    Parameters
    ----------
    att_depth : int
        Number of transformer encoder blocks.
    emb_size : int
        Embedding size of the transformer encoder.
    att_heads : int
        Number of attention heads.
    att_drop : float
        Dropout probability for the attention layers.

    """

    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(
            *[
                _TransformerEncoderBlock(emb_size, att_heads, att_drop)
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(self, final_fc_length,
                 drop_prob_1=0.5, drop_prob_2=0.3, out_channels=256,
                 hidden_channels=32):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax: bool
            Whether to add LogSoftmax non-linearity as the final layer.
        """

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            # nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class _FinalLayer(nn.Module):
    def __init__(self, n_classes, hidden_channels=32, return_features=False, add_log_softmax=False):
        """Classification head for the transformer encoder.

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax : bool
            Adding LogSoftmax or not.
        """

        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channels, n_classes),
        )
        self.return_features = return_features
        if add_log_softmax:
            classification = nn.LogSoftmax(dim=1)
        else:
            classification = nn.Identity()
        if not self.return_features:
            self.final_layer.add_module("classification", classification)

    def forward(self, x):
        if self.return_features:
            out = self.final_layer(x)
            return out, x
        else:
            out = self.final_layer(x)
            return out