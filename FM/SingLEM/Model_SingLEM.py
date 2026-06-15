import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F

class Config:
    # (1) Transformer encoder hyperparameters
    token_len: int      = 128   # 1 s @128 Hz
    n_heads: int        = 8
    d_model: int        = 128
    n_layers: int       = 12
    max_seq_len: int    = 16    # (maximum number of tokens)

    # (2) CNN feature encoder hyperparameters
    filter_num: int     = 32    # number of channels in each band-specific conv
    filter1_size: int   = 3     # kernel size of the first conv layer
    filter2_size: int   = 61    # kernel size of the second conv layer
    cnn_dropout         = 0.3   # dropout
    cnn_features: int   = 128  

    # (3) Token‐embedding hyperparameters
    emb_n_heads: int    = 4
    emb_n_layers: int   = 4
    emb_seq_len: int    = 5    # size of sliding window
    emb_dim: int        = 32 
    emb_d_model: int    = 128

    # (4) Final representation dimension
    rep_dim: int        = 16

    # (5) Masking probability (for pretraining). Set this to 0.0 at downstream applications.
    mask_prob: float    = 0

config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

class BandFeat(nn.Module):
    """
    Extract per-token features from raw 1D EEG using band-specific Conv1d layers.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.filter_num = config.filter_num
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=config.filter_num, kernel_size=config.filter1_size, padding=config.filter1_size//2, bias=True)
        self.conv2 = nn.Conv1d(in_channels=config.filter_num, out_channels=config.filter_num, kernel_size=config.filter2_size, padding=config.filter2_size//2, bias=True)
        self.conv3 = nn.Conv1d(in_channels=config.filter_num, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.LN1 = nn.LayerNorm(config.filter_num)
        self.LN2 = nn.LayerNorm(config.filter_num)

        self.dropout = nn.Dropout(p=config.cnn_dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x = self.LN1(x)
        x = x.permute(0,2,1)
        x = self.dropout(x)
        x = self.elu(x)
        
        x = self.conv2(x)
        x = x.permute(0,2,1)
        x = self.LN2(x)
        x = x.permute(0,2,1)
        x = self.dropout(x)
        x = self.elu(x)

        x = self.conv3(x)
        x = self.elu(x)

        return x

class SelfAttention(nn.Module):
    """ multiple heads self-attention """
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.d_model % config.n_heads == 0
        self.concat_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.concat_proj = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_model = config.d_model

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        returns: same shape
        """
        B_s, S_l, d_m = x.size() 
        qkv = self.concat_attn(x) 
        q, k, v = qkv.split(self.d_model, dim=2)

        k = k.view(B_s, S_l, self.n_heads, d_m // self.n_heads).transpose(1, 2)
        q = q.view(B_s, S_l, self.n_heads, d_m // self.n_heads).transpose(1, 2)
        v = v.view(B_s, S_l, self.n_heads, d_m // self.n_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B_s, S_l, d_m) 
        # output projection
        hidden_states = self.concat_proj(y)
        return hidden_states

class MLP(nn.Module):
    """ FFN inside each Transformer layer """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W_1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.W_2 = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, hidden_states):
        hidden_states = self.W_1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.W_2(hidden_states)
        
        return hidden_states

class TransformerLayer(nn.Module):
    """ Single transformer layer """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.self_att_heads = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
            
    def forward(self, x):
        x = x + self.self_att_heads(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class TransformerEncoder(nn.Module):
    """ Stacked transformer layers """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.Transformer_layers = nn.Sequential(*[TransformerLayer(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)

    def forward(self, masked_emb):
        batch_size, seq_len, d_model = masked_emb.size()
        pos_idx = torch.arange(0, seq_len, dtype=torch.long, device=masked_emb.device)
        pos_emb = self.pos_embedding(pos_idx)
        input = masked_emb + pos_emb
        hidden_states = self.Transformer_layers(input)
        normalized_hidden_states = self.ln_final(hidden_states)
        
        return normalized_hidden_states

class Embedding(nn.Module):
    """
    Given CNN-features for each overlapping window, append a emb token,
    run a small transformer to learn local context,
    then project the emb-token output → d_model. 
    """
    def __init__(self, config):
        super().__init__()
        emb_config = deepcopy(config)
        emb_config.n_heads = emb_config.emb_n_heads
        emb_config.d_model = emb_config.emb_d_model
        emb_config.n_layers = emb_config.emb_n_layers

        self.emb_token = nn.Parameter(torch.randn(1, 1, emb_config.d_model))
        self.pos_embedding = nn.Embedding(emb_config.emb_seq_len + 1, emb_config.d_model)  # emb token is added, so +1
        self.linear_proj = nn.Linear(emb_config.cnn_features, emb_config.d_model)
        self.Emb_Transformer_layers = nn.Sequential(*[TransformerLayer(emb_config) for _ in range(emb_config.n_layers)])
        self.ln = nn.LayerNorm(emb_config.emb_d_model)
        self.proj = nn.Sequential(
            nn.Linear(emb_config.emb_d_model, emb_config.emb_d_model // 2),
            nn.GELU(),
            nn.Linear(emb_config.emb_d_model // 2, emb_config.emb_dim),
            nn.GELU(),
            nn.Linear(emb_config.emb_dim, emb_config.d_model)
        )

    def forward(self, x):
        """
        x: (batch_size * current_seq_len, window_len, cnn_features)
        returns: (batch_size * current_seq_len, d_model)
        """
        emb_batch = x.size(0)
        window_len = x.size(1)

        x = self.linear_proj(x)
        emb_tokens = self.emb_token.expand(emb_batch, -1, -1)
        x = torch.cat([emb_tokens, x], dim=1)

        total_len = window_len + 1
        pos_idx = torch.arange(total_len, dtype=torch.long, device=x.device)
        pos_emb = self.pos_embedding(pos_idx)
        pos_emb = pos_emb.unsqueeze(0).expand(emb_batch, -1, -1) 
        
        x = x + pos_emb
        x = self.Emb_Transformer_layers(x)
        x = x[:, 0, :]  # take emb token
        x = self.proj(self.ln(x))
        
        return x

class MaskedZero(nn.Module):
    """
    Randomly zero out a fraction of the token embeddings.
    If mask_prob=0.0 → identity.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.mask_prob = config.mask_prob

    def forward(self, emb, current_seq_len):
        batch_size, seq_len, d_model = emb.shape
        num_to_mask = int(current_seq_len * self.mask_prob)
        if num_to_mask == 0:
            empty = emb.new_empty((batch_size, 0), dtype=torch.long)
            return emb, empty
        mask_indices = torch.rand(batch_size, current_seq_len, device=device).argsort(dim=-1)[:, :num_to_mask]
        # Create a mask tensor
        mask = torch.zeros(batch_size, current_seq_len, dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)
        masked_emb = torch.where(mask.unsqueeze(-1), torch.zeros_like(emb).to(device), emb)
        
        return masked_emb, mask_indices

class SlidingWindow(nn.Module):
    """Creates overlapping windows from CNN features"""
    def __init__(self, config):
        super().__init__()
        self.window_len = config.emb_seq_len
    
    def forward(self, features, batch_size, current_seq_len):
        """
        features: (batch_size * current_seq_len, feature_dim)
        returns: (batch_size * current_seq_len, window_len, feature_dim)
        """
        total_pad = self.window_len - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        batch_features = features.view(batch_size, current_seq_len, -1)
        features_padded = nn.functional.pad(batch_features, (0, 0, pad_left, pad_right), mode='replicate') 
        extended_features = features_padded.unfold(1, self.window_len, 1)  
        extended_features_permuted = extended_features.permute(0, 1, 3, 2) 
        S_W_output = extended_features_permuted.contiguous().view(batch_size * current_seq_len, self.window_len, features.size(-1)) 
        
        return S_W_output

class Sequencing(nn.Module):
    """
    Take (batch_size * seq_len, d_model)
    and reshape → (batch_size, seq_len, d_model).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
    
    def forward(self, embs, current_seq_len):
        total_tokens = embs.shape[0]
        batch_size = total_tokens // current_seq_len
        embs = embs.view(batch_size, current_seq_len, self.d_model)
        return embs

class EEGEncoder(nn.Module):
    """SingLEM for EEG representation learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cnn = BandFeat(config)
        self.SlidingWindow = SlidingWindow(config)
        self.Embedding = Embedding(config)
        self.Sequencing = Sequencing(config)
        self.Masking = MaskedZero(config)
        self.TransformerEncoder = TransformerEncoder(config)
        # final projection from d_model -> rep_dim
        self.DimRedLayer = nn.Linear(config.d_model, config.rep_dim)
        
        # initialize all weights
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        x: (batch_size, current_seq_len, token_len)
        returns:
          - representations: (batch_size * current_seq_len, rep_dim)
          - mask_idx: (batch_size, num_masked)
          - current_seq_len: int
        """
        batch_size, current_seq_len, token_len = x.shape
        #tokens = x.view(batch_size * current_seq_len, 1, -1)
        tokens = x
        features = self.cnn(tokens).squeeze(1)
        features_extended = self.SlidingWindow(features, batch_size, current_seq_len)
        embeddings = self.Embedding(features_extended)
        seq_embs = self.Sequencing(embeddings, current_seq_len)
        masked, mask_idx = self.Masking(seq_embs, current_seq_len)
        features = self.TransformerEncoder(masked)
        representations = self.DimRedLayer(features)
        
        return representations, mask_idx, current_seq_len

##### EDModel (PRE-TRAINING ONLY)
class EDModel(nn.Module):
    """
    Full pre-training model (encoder + decoder)
    Not used for downstream inference—only for masked-autoencoder pretraining.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = EEGEncoder(config)
        # Decoder: rep_dim -> token_len (reconstruct raw tokens)
        self.decoder = nn.Linear(config.rep_dim, config.token_len)
        self.mask_prob = config.mask_prob
    
    def forward(self, tokens):
        """
        tokens: (batch_size, seq_len, token_len)
        returns:
          - reconstructed tokens: (batch_size, seq_len, token_len)
          - mask_idx: (batch_size, num_masked)
          - current_seq_len
        """
        reps, mask_idx, current_seq_len = self.encoder(tokens)
        recons = self.decoder(reps)
        
        return recons, mask_idx, current_seq_len