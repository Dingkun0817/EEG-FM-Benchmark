import re
import torch
import torch.nn as nn

from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F
from einops import rearrange
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 emb_size = 64,
                 num_heads = 8,
                 depth = 4,
                 max_seq_len = 1024,   
                 ):
        super().__init__()
        
        self.transformer = LinearAttentionTransformer(
            dim = emb_size,
            heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )  
        
    def forward(self, x):
        x = self.transformer(x)
        return x

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, emb_size, code_book_size, decay=0.99, eps=1e-5):
        """
        emb_size: Dimensionality of embeddings
        num_embeddings: Number of codebook entries
        decay: Exponential moving average decay factor
        eps: Small constant for numerical stability
        """
        super().__init__()
        self.emb_size = emb_size
        self.code_book_size = code_book_size
        self.decay = decay
        self.eps = eps

        # Initialize the embeddings
        self.embedding = nn.Embedding(code_book_size, emb_size)
        self.embedding.weight.data.uniform_(-1/code_book_size, 1/code_book_size)

        # Buffers for EMA updates
        self.register_buffer('cluster_size', torch.zeros(code_book_size))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(self, x):
        # x: (B, T, emb_size) or (any_batch_dim, emb_size)
        # Flatten input to (batch_size * time_steps, emb_size)
        flat_x = x.reshape(-1, self.emb_size)

        # Compute distances to each embedding
        # dist: (batch*steps, code_book_size)
        dist = (flat_x.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat_x @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t())

        # Find nearest codebook entries
        encoding_indices = torch.argmin(dist, dim=1)
        quantized = self.embedding(encoding_indices).view_as(x)

        if self.training:
            # Get one-hot encodings: (batch_size * T, code_book_size)
            encodings_one_hot = F.one_hot(encoding_indices, self.code_book_size).type_as(flat_x)

            # Update cluster size with EMA
            new_cluster_size = encodings_one_hot.sum(dim=0)
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

            # Compute new cluster weights
            dw = encodings_one_hot.t() @ flat_x
            self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            # Normalize to get the updated embeddings
            # Normalized cluster sizes
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.code_book_size * self.eps) * n)

            # Avoid division by zero
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # quantized: reconstructed quantized vectors (B, T, emb_size)
        # encoding_indices: indices of selected codebook entries (B*T)
        encoding_indices = encoding_indices.reshape(x.size(0), x.size(1))
        return quantized, encoding_indices


        
def freq_bin_temporal_masking(
    X,
    freq_mask_ratio=0.5,
    freq_bin_size=5,
    time_mask_ratio=0.5,
    time_bin_size=10
):

    B, F, T = X.shape

    # -------------------------------------------------
    # 1) FREQUENCY-BIN MASKING
    # -------------------------------------------------
    # How many freq bins can we form?
    num_freq_bins = F // freq_bin_size

    # Reshape to chunk freq dimension => (B, num_freq_bins, freq_bin_size, T)
    X_freq_binned = X.view(B, num_freq_bins, freq_bin_size, T)

    # Create a frequency mask (1=keep, 0=mask)
    freq_mask = torch.ones_like(X_freq_binned)

    # Number of freq bins to mask
    num_freq_bins_to_mask = int(num_freq_bins * freq_mask_ratio)
    freq_bins_to_mask = torch.randperm(num_freq_bins)[:num_freq_bins_to_mask]

    # Apply mask to chosen freq bins
    freq_mask[:, freq_bins_to_mask, ...] = 0

    # Also reshape freq_mask for combining later
    freq_mask = freq_mask.view(B, F, T)       # (1=keep, 0=mask)


    # -------------------------------------------------
    # 2) TEMPORAL MASKING (on freq-masked result)
    # -------------------------------------------------
    # How many time bins?
    num_time_bins = T // time_bin_size

    # Reshape to chunk time dimension => (B, F, num_time_bins, time_bin_size)
    X_time_binned = X.view(B, F, num_time_bins, time_bin_size)

    # Create a time mask
    time_mask = torch.ones_like(X_time_binned)

    # Number of time bins to mask
    num_time_bins_to_mask = int(num_time_bins * time_mask_ratio)
    time_bins_to_mask = torch.randperm(num_time_bins)[:num_time_bins_to_mask]

    # Mask those time bins
    time_mask[:, :, time_bins_to_mask, :] = 0

    
    # Also reshape time_mask for combining
    time_mask = time_mask.view(B, F, T)

    # -------------------------------------------------
    # 3) COMBINE FREQUENCY + TIME MASKS
    # -------------------------------------------------
    # The final “keep” region is where freq_mask == 1 AND time_mask == 1
    # (i.e., the intersection).
    full_mask = freq_mask * time_mask       # shape (B, F, T)
    full_mask_sym = 1 - full_mask           # inverse
    full_mask = full_mask.to(torch.bool)
    full_mask_sym = full_mask_sym.to(torch.bool)
    # Final masked outputs after freq & time
    X_masked = X * full_mask
    X_masked_sym = X * full_mask_sym


    return X_masked, X_masked_sym, full_mask, full_mask_sym
        
        

################
# This is the downstream model for classification. 
################
class TFM_TOKEN_Classifier(nn.Module):
    def __init__(
        self, 
        emb_size = 256,
        code_book_size = 8192,
        num_heads = 8,
        depth = 12,
        max_seq_len = 61,
        n_classes = 5):            
        super().__init__()

        self.eeg_token_embedding = nn.Embedding(code_book_size+1, emb_size)

        # channel embedding
        self.channel_embed = nn.Embedding(16, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(16)), requires_grad=False
        )
        # temporal position embedding
        self.temporal_pos_embed = PositionalEncoding(emb_size)
        self.pos_drop = nn.Dropout(p=0.1)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        
        self.LAT = LinearAttentionTransformer(
            dim = emb_size,
            heads = num_heads,
            depth = depth,
            max_seq_len = max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
            )    

        self.classification_head = nn.Linear(emb_size, n_classes)
        
    def forward(self, x, num_ch=16, readout='pooling'):
        if readout not in ('flatten', 'pooling'):
            raise ValueError("readout must be 'flatten' or 'pooling', got {!r}".format(readout))

        x = self.eeg_token_embedding(x)

        
        for i in range(x.shape[1]):
            used_channel_embed = self.channel_embed(self.index[i]).unsqueeze(0).unsqueeze(0).expand(x.size(0),-1,-1)
            x[:,i] = self.temporal_pos_embed(x[:,i]+used_channel_embed)
        
        x = rearrange(x, 'B C T E -> B (C T) E')

        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.LAT(x)
        if readout == 'pooling':
            feat = x[:, 1:].mean(dim=1)
        else:
            feat = x
        pred = self.classification_head(feat)
        return pred
    
    def masked_prediction(self, x,num_ch = 16):
        x = self.eeg_token_embedding(x)
        
        for i in range(x.shape[1]):
            used_channel_embed = self.channel_embed(self.index[i]).unsqueeze(0).unsqueeze(0).expand(x.size(0),-1,-1)
            x[:,i] = self.temporal_pos_embed(x[:,i]+used_channel_embed)
        
        x = rearrange(x, 'B C T E -> B (C T) E')
        
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.LAT(x)
        pred = self.classification_head(x[:, 1:])
        return pred
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temporal_pos_embed','cls_token'}

# best model configuration
def get_tfm_token_classifier_64x4(n_classes = 5,code_book_size = 8192, emb_size = 64):
    classifier = TFM_TOKEN_Classifier(
        emb_size = emb_size,
        code_book_size = code_book_size,
        num_heads = 8,
        depth = 4,
        max_seq_len = 2048,
        n_classes = n_classes)
    return classifier
 


################
# TFM-Tokenizer Module with raw EEG and STFT as input
# kindly ignore the naming convention as this is maintained to be consistent with the weights names
################

class TFM_VQVAE2_deep(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 n_freq = 100,
                 n_freq_patch = 5,
                 emb_size = 64,
                 code_book_size = 8192,
                 trans_freq_encoder_depth = 4, 
                 trans_temporal_encoder_depth = 4, 
                 trans_decoder_depth = 4, 
                 beta = 1.0,
                 ):
        super().__init__()
        self.n_freq_patch = n_freq_patch
        
        # bin wise frequency embedding
        self.freq_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size = n_freq_patch, stride = n_freq_patch),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size),
            )
        
        
        # Freq Encoder
        self.trans_freq_encoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = trans_freq_encoder_depth,
                                                max_seq_len = n_freq//n_freq_patch)
        
        # Temporal embedding
        self.temporal_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size = 200, stride = 100),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size),
            nn.Conv1d(emb_size, emb_size//2, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.GroupNorm(emb_size//4,emb_size//2),
            )
        
        
        
        # attention based aggregation
        global_freq_divider = n_freq//(n_freq_patch*n_freq_patch)
        self.freq_patch_embedding_2_atten = nn.Sequential(
            nn.Conv1d(emb_size, emb_size//(global_freq_divider*2), kernel_size = n_freq_patch, stride = n_freq_patch),
            nn.Sigmoid()
              )
        self.freq_patch_embedding_2 = nn.Sequential(
            nn.Conv1d(emb_size, emb_size//(global_freq_divider*2), kernel_size = n_freq_patch, stride = n_freq_patch),
              )
        
        
        
        # Temporal Encoder
        self.trans_temporal_encoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = trans_temporal_encoder_depth)
        
        
        # Vector quantization bottleneck
        self.quantizer =EMAVectorQuantizer(emb_size, code_book_size)
        self.beta = beta
        
        
        # Decoder
        self.trans_decoder = TransformerEncoder(emb_size = emb_size,
                                                num_heads = 8,
                                                depth = trans_decoder_depth)
        
        
        # self.decoder = nn.Linear(emb_size, n_freq)
        self.decoder = nn.Sequential(
                    nn.Linear(emb_size, emb_size),
                    nn.Tanh(),
                    nn.Linear(emb_size, n_freq,)
                    )
        
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantizer.embedding.weight'}
        
    def tokenize(self, x, x_temporal):
        B,F,T = x.shape
        x = x.permute(0, 2, 1).reshape(-1,1,F)
        x = self.freq_patch_embedding(x)
        x = x.permute(0, 2, 1)
        
        
        # learn frequency information
        x = self.trans_freq_encoder(x)
        
        ## Freq info aggregation
        x = x.permute(0, 2, 1)
        atten = self.freq_patch_embedding_2_atten(x)
        x = self.freq_patch_embedding_2(x)*atten
        x = x.reshape(-1,x.size(1)*x.size(2))
        
        x = rearrange(x, '(B T) E -> B T E', T=T)
        
        
        ## temporal patch embedding
        x_temporal = x_temporal.unsqueeze(1)
        x_temporal = self.temporal_patch_embedding(x_temporal)
        x_temporal = rearrange(x_temporal, 'B E T -> B T E')
        
        # concatenate the freq and temporal embeddings
        x = torch.cat((x,x_temporal),dim=-1)
        
        
        # learn temporal information
        x = self.trans_temporal_encoder(x)
        
        quant_in = l2norm(x)
        # Vector quantization
        quant_out, indices = self.quantizer(quant_in)
        
        
        return quant_out, indices, quant_in
        
    def forward(self, x, x_temporal):
        
        quant_out, indices, quant_in = self.tokenize(x, x_temporal)
        
        # Straight through estimator
        quant_out = quant_in + (quant_out - quant_in).detach()
        
        
        x = self.trans_decoder(quant_out)
        
        x = self.decoder(x)
        
        x = x.permute(0, 2, 1)
        
        return x, indices,quant_out,quant_in
        
    def vec_quantizer_loss(self, quant_in, quant_out):
        # compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_in) ** 2)
        code_book_loss = torch.mean((quant_out - quant_in.detach()) ** 2)
        
        loss = code_book_loss + self.beta * commitment_loss
        
        return loss, code_book_loss, commitment_loss
    
    @torch.no_grad()
    def forward_ana(self, x, x_temporal):
        """
        Forward pass with intermediate outputs.
        
        Returns:
        - reconstructed output (x_dec)
        - quantizer indices (indices)
        - quantized representation (quant_out)
        - input to quantizer (quant_in)
        - frequency encoder tokens (freq_encoded)
        - temporal encoder tokens (temporal_encoded)
        """
        B, F, T = x.shape
     
        # Frequency branch
        x_freq = x.permute(0, 2, 1).reshape(-1, 1, F)  # (B*T, 1, F)
        x_freq = self.freq_patch_embedding(x_freq)       # Frequency patch embedding
        x_freq = x_freq.permute(0, 2, 1)                  # (B*T, seq_len, emb_size)
        
        # Get frequency transformer encoder output
        freq_encoded = self.trans_freq_encoder(x_freq)    # (B*T, seq_len, emb_size)

        # Frequency aggregation
        x_freq_agg = freq_encoded.permute(0, 2, 1)          # (B*T, emb_size, seq_len)
        atten = self.freq_patch_embedding_2_atten(x_freq_agg)
        x_freq_agg = self.freq_patch_embedding_2(x_freq_agg) * atten
        x_freq_agg = x_freq_agg.reshape(-1, x_freq_agg.size(1) * x_freq_agg.size(2))
        x_freq_agg = rearrange(x_freq_agg, '(B T) E -> B T E', T=T)

        # Temporal branch
        x_temporal_branch = x_temporal.unsqueeze(1)
        x_temporal_branch = self.temporal_patch_embedding(x_temporal_branch)
        x_temporal_branch = rearrange(x_temporal_branch, 'B E T -> B T E')

        # Concatenate frequency and temporal embeddings
        x_combined = torch.cat((x_freq_agg, x_temporal_branch), dim=-1)
        
        # Get temporal transformer encoder output
        temporal_encoded = self.trans_temporal_encoder(x_combined)

        # Vector quantization
        quant_in = l2norm(temporal_encoded)
        quant_out, indices = self.quantizer(quant_in)
        quant_out = quant_in + (quant_out - quant_in).detach()

        
        # Decoder branch
        x_dec = self.trans_decoder(quant_out)
        x_dec = self.decoder(x_dec)
        x_dec = x_dec.permute(0, 2, 1)

        return x_dec, indices, quant_out, quant_in, freq_encoded, temporal_encoded

        
def get_tfm_tokenizer_2x2x8(code_book_size = 8192,emb_size = 64):
    vqvae = TFM_VQVAE2_deep(in_channels = 1,
                 n_freq = 100,
                 n_freq_patch = 5,
                 emb_size = emb_size,
                 code_book_size = code_book_size,
                 trans_freq_encoder_depth = 2, #8
                 trans_temporal_encoder_depth = 2, #2
                 trans_decoder_depth = 8, # 3,
                 beta = 1.0,)
    return vqvae




def load_embedding_weights(source_model, target_model):
    """
    Load nn.Embedding weights from `tfm_token2_deep` into `TFM_classifier`
    while handling size mismatches.
    
    Args:
        source_model (torch.nn.Module): The model `tfm_token2_deep` where weights are loaded from.
        target_model (torch.nn.Module): The model `TFM_classifier` where weights are assigned.
    """
    # Extract embedding weights
    source_weights = source_model.quantizer.embedding.weight.data
    target_weights = target_model.eeg_token_embedding.weight.data

    src_vocab_size, src_emb_dim = source_weights.shape
    tgt_vocab_size, tgt_emb_dim = target_weights.shape


    # Ensure embedding dimensions match (they should ideally be the same)
    if src_emb_dim != tgt_emb_dim:
        raise ValueError(f"Embedding size mismatch: {src_emb_dim} (source) vs {tgt_emb_dim} (target)")

    # Adjust the vocabulary size
    if src_vocab_size > tgt_vocab_size:
        # Trim extra embeddings
        adapted_weights = source_weights[:tgt_vocab_size, :]
    elif src_vocab_size < tgt_vocab_size:
        # Pad embeddings with zeros for missing entries
        adapted_weights = torch.zeros((tgt_vocab_size, tgt_emb_dim), dtype=source_weights.dtype)
        adapted_weights[:src_vocab_size, :] = source_weights
    else:
        # No need for adjustments
        adapted_weights = source_weights

    # Load the adapted weights into the target model
    target_model.eeg_token_embedding.weight.data.copy_(adapted_weights)

def get_stft_torch(X, resampling_rate = 200):
    B,C,T = X.shape
    x_temp = rearrange(X, 'B C T -> (B C) T')
    window = torch.hann_window(resampling_rate).to(x_temp.device)
    x_stft_temp = torch.abs(torch.stft(x_temp, n_fft=resampling_rate, hop_length=resampling_rate//2, 
                          onesided = True,
                          return_complex=True, center = False,#normalized = True,
                          window = window)[:,:resampling_rate//2,:])
    
    x_stft_temp = rearrange(x_stft_temp, '(B C) F T -> B C F T', B=B)
    
    return x_stft_temp