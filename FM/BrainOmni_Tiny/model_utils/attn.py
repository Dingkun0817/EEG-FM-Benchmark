import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


class RMSNorm(torch.nn.Module):
    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim)) if elementwise_affine else 1.0
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, n_dim, init_seq_len, base=10000):
        super().__init__()
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim)),
        )
        self._set_rotate_cache(init_seq_len)

    def _set_rotate_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, device=self.freqs.device).type_as(self.freqs)
        rotate = torch.outer(t, self.freqs).float()
        self.register_buffer("rotate", torch.polar(torch.ones_like(rotate), rotate))

    def reshape_for_broadcast(self, x: torch.Tensor):
        """
        x      Batch seq n_head d_head
        rotate seq dim
        """
        B, T, H, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert H * D == rotate.shape[1]
        return rearrange(rotate, "T (H D)-> T H D", H=H).unsqueeze(0)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(3)
        k_out = torch.view_as_real(k_ * rotate).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(self, n_dim, n_head, dropout, causal):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.time_attn = SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=causal, rope=True
        )
        self.spatial_attn = SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=False, rope=False
        )
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        """
        True element in mask will take part in attention
        """
        x = x + self._attn_operator(self.pre_attn_norm(x))
        x = x + self.ff(self.pre_ff_norm(x))
        return x

    def _attn_operator(self, x):
        B, C, W, D = x.shape
        xs = rearrange(x[:, :, :, D // 2 :], "B C W D -> (B W) C D")
        xt = rearrange(x[:, :, :, : D // 2], "B C W D->(B C) W D")
        xs = self.spatial_attn(xs, None)
        xt = self.time_attn(xt, None)
        xs = rearrange(xs, "(B W) C D -> B C W D", B=B)
        xt = rearrange(xt, "(B C) W D->B C W D", B=B)
        return torch.cat([xs, xt], dim=-1)


# Attention
class SelfAttnBlock(nn.Module):
    def __init__(self, n_dim, n_head, dropout, causal, rope):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.attn = SelfAttention(n_dim, n_head, dropout, causal=causal, rope=rope)
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        """
        True element in mask will take part in attention
        """
        x = x + self.attn(self.pre_attn_norm(x), mask)
        x = x + self.ff(self.pre_ff_norm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, n_dim, n_head, dropout, causal: bool = False, rope: bool = False
    ):
        super().__init__()
        assert n_dim % n_head == 0
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope = rope
        self.rope_embedding_layer = (
            RotaryEmbedding(n_dim=n_dim, init_seq_len=240) if self.rope else Identity()
        )

    def forward(self, x: torch.Tensor, mask=None):
        """
        True element in mask will take part in attention
        """
        B, T, C = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)

        # RoPE changes expected layout; branch on use_rope
        if self.rope:
            q = q.view(B, T, self.n_head, -1)
            k = k.view(B, T, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
            k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)

        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)

        # add head_dim
        if mask != None:
            mask = mask.unsqueeze(1)

        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, T, -1)
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, n_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_dim, int(4 * n_dim)),
            nn.SELU(),
            nn.Linear(int(4 * n_dim), n_dim),
            nn.Dropout(dropout) if dropout != 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)
