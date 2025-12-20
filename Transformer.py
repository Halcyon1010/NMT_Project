import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Normalization Modules (Ablation)
# ==========================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g


# ==========================================
# 2. Positional Encodings (Ablation)
# ==========================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


# ==========================================
# 3. Transformer Layers (Fixed for Compatibility)
# ==========================================
class TransformerEncoderLayer(nn.Module):
    """
    Custom Encoder Layer compatible with PyTorch's native naming.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        # Dynamic Norm
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # 🔴 Fix: Added **kwargs to catch 'is_causal' passed by nn.TransformerEncoder
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # 1. Self-Attention
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(
            src_norm, src_norm, src_norm, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        # 2. Feed Forward
        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src2)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Custom Decoder Layer compatible with PyTorch's native naming.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    # 🔴 Fix: Added **kwargs to catch arguments passed by nn.TransformerDecoder
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        
        # 1. Self-Attention
        tgt_norm = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, 
            attn_mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # 2. Cross-Attention
        tgt_norm = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt_norm, memory, memory, 
            attn_mask=memory_mask, 
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)

        # 3. Feed Forward
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


# ==========================================
# 4. Main Transformer Model
# ==========================================
class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 d_model=512, nhead=8, num_layers=6, 
                 pos_embed_type='sinusoidal', norm_type='layernorm'):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # 2. Position Encoding
        if pos_embed_type == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model)
            self.pos_decoder = LearnablePositionalEncoding(d_model)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
            self.pos_decoder = SinusoidalPositionalEncoding(d_model)

        # 3. Encoder & Decoder (Using PyTorch Containers for weight compatibility)
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, norm_type=norm_type)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, norm_type=norm_type)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        
        # 4. Output Projection
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def encode(self, src, src_mask):
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        # nn.TransformerEncoder will auto-loop and pass kwargs if needed
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        return memory

    def decode(self, trg, memory, trg_mask):
        trg_emb = self.pos_decoder(self.trg_embedding(trg) * math.sqrt(self.d_model))
        return self.decoder(trg_emb, memory, tgt_mask=trg_mask)

    def forward(self, src, trg, trg_mask=None, src_key_padding_mask=None, trg_key_padding_mask=None):
        # 1. Embeddings
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_decoder(self.trg_embedding(trg) * math.sqrt(self.d_model))
        
        # 2. Encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            
        # 3. Decoder
        output = self.decoder(
            tgt=trg_emb, 
            memory=memory, 
            tgt_mask=trg_mask, 
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.fc_out(output)