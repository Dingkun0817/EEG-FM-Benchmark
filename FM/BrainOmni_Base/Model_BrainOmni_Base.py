import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .model_utils.attn import SelfAttnBlock, RMSNorm, SpatialTemporalAttentionBlock
from .model_utils.loss import get_time_loss, get_pcc, get_frequency_domain_loss
from .model_utils.module import (
    BrainSensorModule,
    BrainTokenizerEncoder,
    BrainQuantizer,
    BrainTokenizerDecoder,
)



class BrainTokenizer(nn.Module):
    def __init__(
        self,
        window_length,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        n_dim,
        n_neuro,
        n_head,
        dropout,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        **kwargs,
    ):
        super().__init__()
        self.window_length = window_length
        self.n_dim = n_dim
        self.sensor_embed = BrainSensorModule(n_dim)
        self.mask_ratio = 0.25  # hard coded

        self.encoder = BrainTokenizerEncoder(
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=n_dim,
            n_neuro=n_neuro,
            n_head=n_head,
            dropout=dropout,
        )
        self.quantizer = BrainQuantizer(
            n_dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        self.decoder = BrainTokenizerDecoder(
            n_dim=n_dim,
            n_head=n_head,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=dropout,
        )
        # --------------------------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    @torch.jit.ignore
    def get_parameters_groups(self, lr: float, codebook_lr: float, weight_decay: float):
        normal_params = []
        no_decay_params = []
        codebook_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if "norm" in n or n in [
                    "sensor_embed.sensor_embedding_layer.weight",
                ]:
                    no_decay_params.append(p)
                elif "quantizer" in n:
                    codebook_params.append(p)
                else:
                    normal_params.append(p)
        return [
            {"params": normal_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
            {"params": codebook_params, "lr": codebook_lr, "weight_decay": 0.0},
        ]

    def unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))
        if overlap_ratio > 0.0:
            stride = int(self.window_length * (1 - overlap_ratio))
            right_remain = (x.shape[-1] - self.window_length) % stride
            if right_remain > 0:
                x = F.pad(x, pad=(0, stride - right_remain))
        return x.unfold(
            dimension=-1,
            size=self.window_length,
            step=int(self.window_length * (1 - overlap_ratio)),
        )

    def norm_target(self, x: torch.Tensor):
        """
        x: B C N L
        """
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1,keepdim=True)+1e-6)
        return x

    def add_noise(self, x: torch.Tensor):
        return x + torch.randn_like(x) * 0.1

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        """
        x: B C (N L)
        pos: B C 6
        sensor_type: B C
        """
        x = self.unfold(x)

        sensor_embedding = self.sensor_embed(pos, sensor_type)
        random_index = torch.randperm(x.shape[1], device=x.device)
        x = x.index_select(dim=1, index=random_index)
        sensor_embedding = sensor_embedding.index_select(dim=1, index=random_index)
        n_mask_channel = max(int(x.shape[1] * self.mask_ratio), 1)
        feature = self.encoder(
            self.add_noise(x[:, n_mask_channel:]),
            sensor_embedding[:, n_mask_channel:],
        )

        feature, indices, commitment_loss = self.quantizer(feature)

        x_rec = self.decoder(feature, sensor_embedding)

        x_rec = x_rec.float()
        x = self.norm_target(x)

        time_loss = get_time_loss(x_rec, x)
        pcc = get_pcc(x_rec, x)
        amp_loss, phase_loss = get_frequency_domain_loss(x_rec, x)
        return {
            "loss": time_loss
            + torch.exp(-pcc)
            + commitment_loss
            + amp_loss
            + 0.5 * phase_loss,
            "time_loss": time_loss.detach(),
            "pcc": pcc.detach(),
            "amp_loss": amp_loss.detach(),
            "phase_loss": phase_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "judge_loss": (
                time_loss
                + torch.exp(-pcc)
                + commitment_loss
                + amp_loss
                + 0.5 * phase_loss
            ).detach(),
        }, indices

    @torch.no_grad()
    def visualize(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor, **kwargs
    ):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C
        """
        x = self.unfold(x)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        x_rec = self.decoder(feature, sensor_embedding)
        return {
            "x": self.norm_target(x),
            "x_rec": x_rec.float(),
            "sensor_type": sensor_type,
        }

    @torch.no_grad()
    def tokenize(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        overlap_ratio: float,
        **kwargs,
    ):
        """
        x: B C T
        pos: B C 6
        sensor_type: B C
        """
        self.eval()
        x = self.unfold(x, overlap_ratio=overlap_ratio)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, commitment_loss = self.quantizer(feature)
        feature = rearrange(feature, "B C N T D->B C (N T) D")
        indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
        return feature, indices

    def get_finetune_parameter_groups(self, weight_decay, layer_decay):
        del self.decoder
        del self.quantizer
        parameter_groups = {}

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            this_weight_decay = weight_decay
            group_name = "decay"

            # Create group if it doesn't exist
            if group_name not in parameter_groups:
                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": layer_decay,
                }

            parameter_groups[group_name]["params"].append(p)

        return list(parameter_groups.values())

class BrainOmni(nn.Module):
    def __init__(
        self,
        # tokenizer parameter
        window_length: int,
        n_filters: int,
        ratios: List[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        n_neuro: int,
        dropout: float,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
        # lm model parameter
        overlap_ratio: float,
        lm_dim: int,
        lm_head: int,
        lm_depth: int,
        lm_dropout: float,
        mask_ratio: float,
        num_quantizers_used: int,
        **kwargs,
    ):
        super().__init__()
        self.lm_dim = lm_dim
        self.window_length = window_length
        self.overlap_ratio = overlap_ratio
        self.mask_ratio = mask_ratio
        self.num_quantizers_used = (
            num_quantizers_used if num_quantizers_used != None else num_quantizers
        )
        # B C T -> unfold -> B C T' -> tokenizer -> (B C) W D -> next predict
        self.tokenizer = BrainTokenizer(
            window_length,
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            n_dim,
            n_neuro,
            n_head,
            dropout,
            codebook_dim,
            codebook_size,
            num_quantizers,
            rotation_trick,
            quantize_optimize_method,
        )
        self.mask_token = nn.Parameter(torch.randn(n_dim))
        self.projection = nn.Linear(n_dim, lm_dim) if n_dim != lm_dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(lm_dim, lm_head, lm_dropout, causal=False)
                for _ in range(lm_depth)
            ]
        )
        self.predict_head = nn.Linear(lm_dim, num_quantizers_used * codebook_size)
        # --------------------------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, RMSNorm):
            if isinstance(m.weight, nn.Parameter):
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    @torch.jit.ignore
    def load_frozen_tokenizer_ckpt(self, tokenizer_ckpt_path='/data0/ldk_zgc/EEG_Large_Model/cz/EEGFM_Bench11.6/models/pretrained_models/BrainOmni/braintokenizer/BrainTokenizer.pt'):
        self.tokenizer.load_state_dict(
            torch.load(tokenizer_ckpt_path, weights_only=True)
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        return None

    @torch.jit.ignore
    def get_parameters_groups(self, lr: float, weight_decay: float):
        no_decay_params = []
        normal_params = []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if (
                    "norm" in n
                    or "predict_head" in n
                    or n in ["projection.weight", "projection.bias", "mask_token"]
                ):
                    no_decay_params.append(p)
                else:
                    normal_params.append(p)

        return [
            {"params": normal_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        **kwargs,
    ):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C
        """
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, D = x.shape

        mask = (
            torch.rand(size=(B, C, W), device=x.device) > self.mask_ratio
        )  # true in mask will be preserve, false in mask will be masked
        # 20% random select a token from the minibatch
        x = torch.where(
            mask.unsqueeze(-1).repeat(1, 1, 1, D),
            x,
            rearrange(
                x.view(-1, D)[torch.randperm(B * C * W, device=x.device)],
                "(B C W) D -> B C W D",
                B=B,
                C=C,
            ),
        )
        # 80% use mask token
        tmp_mask = (mask.float() + torch.rand(size=(B, C, W), device=x.device)) > 0.8
        tmp_mask = tmp_mask.unsqueeze(-1).type_as(x)
        mask_token = self.mask_token.type_as(x)
        x = x * tmp_mask + mask_token * (1 - tmp_mask)

        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro

        x = self.projection(x)

        for block in self.blocks:
            x = block(x)

        return x

    def encode(self, x: torch.Tensor, pos: torch.Tensor, sensor_type: torch.Tensor):
        """
        x: B C (W T)
        pos: B C 6
        sensor_type: B C

        output: B W D
        """
        
        x, label_indices = self.tokenizer.tokenize(
            x, pos, sensor_type, self.overlap_ratio
        )

        B, C, W, _ = x.shape
        neuro = self.tokenizer.encoder.neuros.type_as(x).detach().view(1, C, 1, -1)
        x = x + neuro
        x = self.projection(x)

        for block in self.blocks[:-1]:
            x = block(x)

        return F.normalize(
            x,
            p=2.0,
            dim=-1,
            eps=1e-6,
        )

    def compute_cross_entropy(
        self, logits: torch.Tensor, label: torch.Tensor, mask: torch.Tensor
    ):
        """
        logits: B C W num_quantizers_used codebook_size
        label:  B C W num_quantizers_used
        mask:   B C W
        """
        B, C, W, N = label.shape
        logits = logits[~mask]
        label = label[~mask]
        #  X is masked num , N is codebook depth, M is codebook size
        logits = rearrange(logits, "X N M -> (X N) M")
        label = label.view(-1)

        loss = F.cross_entropy(logits.float(), label, reduction="mean")

        acc = (
            rearrange((logits.argmax(dim=-1)) == label, "(X N) -> N X", N=N)
            .float()
            .mean(dim=-1)
        )

        return loss, acc