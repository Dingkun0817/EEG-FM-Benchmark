import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .attn import RMSNorm, FeedForward, SelfAttnBlock
from .seanet import SEANetEncoder, SEANetDecoder
from .vq import RVQ


# pos+sensor_type -> sensor_embedding
class BrainSensorModule(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        self.aggregate_mlp = FeedForward(n_dim, 0.0)
        self.norm = RMSNorm(n_dim)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor):
        """
        pos,dir         B C 6
        sensor_type     B C
        """
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class ForwardSolution(nn.Module):
    def __init__(self, n_dim, n_head, dropout):
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, sensor_embedding: torch.Tensor, neurons: torch.Tensor):
        B, C, _ = sensor_embedding.shape
        kv = self.kv(neurons)
        k, v = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        q = rearrange(sensor_embedding, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)
        output = (
            F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.dropout, is_causal=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, C, -1)
        return self.proj(output)


class BackWardSolution(nn.Module):
    def __init__(self, n_dim, n_head, dropout):
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, neuros: torch.Tensor, k: torch.Tensor, x: torch.Tensor):
        B, N_q, _ = neuros.shape
        q = rearrange(neuros, "B T (H D) -> B H T D", H=self.n_head)
        k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)
        v = rearrange(self.v(x), "B T (H D) -> B H T D", H=self.n_head)
        output = (
            F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.dropout, is_causal=False
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, N_q, -1)
        return self.proj(output)


# recieve B C (W T) sensor_embedding  -> conv encode -> spatial encode -> time encode -> squeeze to B W (Q D)
class BrainTokenizerEncoder(nn.Module):
    def __init__(
        self,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ):
        super().__init__()
        self.seanet_encoder = SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = BackWardSolution(
            n_dim=n_dim, n_head=n_head, dropout=dropout
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor = None, **kwargs):
        """
        x:                 B C N L(unfolded)    batch channel N_split length
        sensor_embedding   B C D                batch channel neuro_dim
        mask_ratio         0.0  means no mask
        """
        B, C, N, L = x.shape
        x = rearrange(x, "B C N L -> (B C N) 1 L")
        x = self.seanet_encoder(x)
        x = rearrange(x, "(B C N) D T -> B C (N T) D", B=B, C=C, N=N)
        B, C, W, _ = x.shape
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, W, 1), "B C W D -> (B W) C D"
        )
        x = rearrange(x, "B C W D -> (B W) C D")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        x = rearrange(x, "(B N T) C D -> B C (N T) D", B=B, N=N)
        return rearrange(x, "B C (N T) D -> B C N T D", N=N)


# recieve B W (Q D) sensor_embedding -> B W (Q D) quantized feature and  num_quantizers B W label
class BrainQuantizer(nn.Module):
    def __init__(
        self,
        n_dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
    ):
        super().__init__()
        self.rvq = RVQ(
            dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )

    @torch.no_grad()
    def encode(self, x):
        indices = self.rvq.encode(F.normalize(x, p=2.0, dim=-1))
        return indices

    def forward(self, x):
        """
        x: B W D
        x_q: B W D
        indices: B W num quantizer
        loss: num_quantizer
        """
        x_q, indices, loss = self.rvq(F.normalize(x, p=2.0, dim=-1))
        return x_q, indices, loss


# recieve B W (Q D) sensor_embedding -> B C W D reconstruct
class BrainTokenizerDecoder(nn.Module):
    def __init__(
        self, n_dim, n_head, n_filters, ratios, kernel_size, last_kernel_size, dropout
    ):
        super().__init__()
        self.forwardsolution = ForwardSolution(n_dim, n_head, dropout)

        self.seanet_decoder = SEANetDecoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor = None):
        """
        neuros              B N T (Q D)
        sensor_feature      B C D
        """
        B, C, N, T, D = x.shape
        x = rearrange(x, "B C N T D -> (B N T) C D")
        sensor_embedding = rearrange(
            sensor_embedding.view(B, -1, 1, 1, D).repeat(1, 1, N, T, 1),
            "B C N T D -> (B N T) C D",
        )
        x = self.forwardsolution(sensor_embedding, x)
        x = rearrange(x, "(B N T) C D -> (B C N) D T", B=B, N=N, T=T)
        x = self.seanet_decoder(x)
        return rearrange(x, "(B C N) 1 L -> B C N L", B=B, N=N)
