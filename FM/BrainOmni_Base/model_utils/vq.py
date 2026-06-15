from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
try:
    import deepspeed.comm as dist
except Exception:
    import torch.distributed as dist
from typing import List
from math import ceil
from einx import get_at
from einops import rearrange, pack, unpack
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to


def first(it):
    return it[0]


def exists(v):
    return v is not None


def identity(t):
    return t


def default(v, d):
    return v if exists(v) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def get_maybe_sync_seed(device, max_size=10_00):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        (out,) = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


def world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_distributed():
    return world_size() > 1


def broadcast_tensors(tensors: List[torch.Tensor], src_rank=0):
    if not is_distributed():
        return
    for tensor in tensors:
        dist.broadcast(tensor, src=src_rank)


def all_reduce_tensors(tensors: List[torch.Tensor], op):
    if not is_distributed():
        return
    for tensor in tensors:
        dist.all_reduce(tensor, op=op)


def ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, epsilon: float = 1e-6):
    return (x + epsilon) / (x.sum() + epsilon * len(x))


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


@torch.no_grad()
def kmeans(samples: torch.Tensor, nums_clusters: int, kmeans_iters: int):
    samples = rearrange(samples, "... d -> (...) d")
    dim, dtype = samples.shape[1], samples.dtype
    if samples.shape[0] < nums_clusters:
        random_noise = torch.randn(
            size=(nums_clusters - samples.shape[0], dim),
            device=samples.device,
            dtype=dtype,
        )
        samples = torch.cat([samples, random_noise], dim=0)
    centers = sample_vectors(samples, nums_clusters)
    for i in range(kmeans_iters):
        diffs = ((samples.unsqueeze(1) - centers) ** 2).sum(dim=-1)
        buckets = diffs.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=nums_clusters)
        zero_mask = bins == 0
        bins[zero_mask] = 1

        new_centers = centers.new_zeros(nums_clusters, dim, dtype=dtype)
        new_centers.scatter_add_(0, buckets.unsqueeze(-1).repeat([1, dim]), samples)
        new_centers = new_centers / bins[..., None]
        centers = torch.where(zero_mask[..., None], centers, new_centers)

    return centers, bins


class SimVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_dim: int,  # frozen codebook dim could have different dimensions than projection
        codebook_size: int,
        rotation_trick: bool,  # works even better with rotation trick turned on, with no straight through and the commit loss from input to quantize
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        codebook = torch.randn(codebook_size, codebook_dim) * (codebook_dim**-0.5)

        self.code_transform_linear = nn.Linear(codebook_dim, dim)
        self.code_transform_residual = nn.Sequential(nn.SELU(), nn.Linear(dim, dim))

        self.register_buffer("frozen_codebook", codebook)

        # make sure the codebook in every gpu is the same
        broadcast_tensors([self.frozen_codebook], src_rank=0)

        # whether to use rotation trick from Fifty et al.
        # https://arxiv.org/abs/2410.06424

        self.rotation_trick = rotation_trick

    @property
    def codebook(self):
        return self._forward_code_transform(self.frozen_codebook)

    def _forward_code_transform(self, x: torch.Tensor):
        """
        x B T C
        """
        x = self.code_transform_linear(x)
        return x + self.code_transform_residual(x)

    def forward(self, x: torch.Tensor):
        """
        x B ... D
        """
        input_dtype = x.dtype
        x, inverse_pack = pack_one(x, "b * d")

        implicit_codebook = self.codebook

        with torch.no_grad():
            dist = torch.cdist(x.float(), implicit_codebook.float())
            indices = dist.argmin(dim=-1)  # B *
        # select codes
        quantized = get_at("[c] d, b n -> b n d", implicit_codebook, indices)

        # commit loss and straight through, as was done in the paper
        commit_loss = (
            F.mse_loss(
                x.detach().float(), quantized.float()
            )  # ask learned implicit codebook to be close to the input
            + F.mse_loss(
                x.float(), quantized.detach().float()
            )  # ask input to cluster together
            * 0.25
        )

        if self.rotation_trick:
            # rotation trick from @cfifty
            quantized = rotate_to(x, quantized).to(input_dtype)
        else:
            quantized = (quantized - x).detach() + x

        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, "b *")

        return quantized, indices, commit_loss

    def encode(self, x: torch.Tensor):
        """
        x B ... D
        """

        x, inverse_pack = pack_one(x, "b * d")

        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x.float(), implicit_codebook.float())
            indices = dist.argmin(dim=-1)

        indices = inverse_pack(indices, "b *")
        return indices

    def decode(self, indices: torch.Tensor):
        """
        indices B ... codebook_size
        """
        frozen_codes = get_at("[c] d, b ... -> b ... d", self.frozen_codebook, indices)
        quantized = self.code_transform(frozen_codes)
        return quantized


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        embed, cluster_size = kmeans(
            sample_vectors(data, 4096),
            self.codebook_size,
            self.kmeans_iters,
        )
        # embed, cluster_size = embed.to(data.device), cluster_size.to(data.device)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    # flatten in
    @torch.no_grad()
    def quantize(self, x):
        x = x.float()
        embed = self.embed.t().float()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.argmin(dim=-1)
        return embed_ind  # n

    # any in any out
    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    # any in any out
    def encode(self, x):
        shape = x.shape
        # pre-process
        x = rearrange(x, "... d -> (...) d")
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # equals dequantize
    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... d -> (...) d")
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind).type(dtype)

        if self.training:
            self.expire_codes_(x)
            # Count how often each code is used before normalization
            one_hot_sum = embed_onehot.sum(0)
            all_reduce_tensors([one_hot_sum], op=dist.ReduceOp.SUM)
            ema_inplace(self.cluster_size, one_hot_sum, self.decay)
            # Sum embeddings assigned to each code before normalization
            embed_sum = embed_onehot.t() @ x
            embed_sum = embed_sum.to(torch.float32)
            all_reduce_tensors([embed_sum], op=dist.ReduceOp.SUM)
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            # Apply smoothing
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.epsilon)
                * self.cluster_size.sum()
            )
            # Replace with the updated embeddings
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int = None,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        rotation_trick: bool = True,
    ):
        super().__init__()
        self.rotation_trick = rotation_trick
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    # any in any out
    def encode(self, x):
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    # any in any out
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        return quantize

    def forward(self, x):
        input_dtype = x.dtype
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            if self.rotation_trick:
                quantize = rotate_to(x, quantize).to(input_dtype)
            else:
                quantize = x + (quantize - x).detach()

        loss = F.mse_loss(x.float(), quantize.detach().float()) * 0.25
        if not self.training:
            loss = loss.detach()

        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class RVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        rotation_trick=True,  # rotation trick from @cfifty
        quantize_optimize_method="ema",
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([])
        for _ in range(num_quantizers):
            self.layers.append(
                VectorQuantization(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    rotation_trick=rotation_trick,
                )
                if quantize_optimize_method == "ema"
                else SimVQ(
                    dim=dim,
                    codebook_dim=codebook_dim,
                    codebook_size=codebook_size,
                    rotation_trick=rotation_trick,
                )
            )
        # quantize dropout
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebook_size(self):
        return first(self.layers).codebook_size

    @property
    def codebook_dim(self):
        return first(self.layers).codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks)
        return codebooks

    def forward(self, x: torch.Tensor):
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

        # save all inputs across layers, for use during expiration at end under shared codebook setting

        # go through the layers
        for quantizer_index, vq in enumerate(self.layers):

            if (
                should_quantize_dropout
                and quantizer_index > rand_quantize_dropout_index
            ):
                continue

            # sim vq forward

            quantized, indices, loss = vq(residual)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_indices.append(indices)

        # stack all losses and indices
        all_losses = torch.stack(all_losses, dim=-1)
        all_indices = torch.stack(all_indices, dim=-1)

        return quantized_out, all_indices, all_losses.mean()

    def encode(self, x: torch.Tensor):
        """
        x: B W D
        """
        quantized_out = 0.0
        residual = x
        all_indices = []
        for quantizer_index, vq in enumerate(self.layers):
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        all_indices = torch.stack(all_indices, dim=-1)
        return all_indices
