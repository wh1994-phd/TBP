import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .fsin_block import FreqSparseInteractionBlock
except ImportError:
    from fsin_block import FreqSparseInteractionBlock


class PGTSINet(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, d_ff, dropout, d_mem=32):
        super(PGTSINet, self).__init__()
        self.d_model = d_model
        chan_attn_layer_p = nn.TransformerEncoderLayer(
            d_model=self.d_v, nhead=max(1, n_heads // 2), dim_feedforward=self.d_v * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.channel_attention_p = nn.TransformerEncoder(chan_attn_layer_p, num_layers=1)
        chan_attn_layer_d = nn.TransformerEncoderLayer(
            d_model=self.d_v, nhead=max(1, n_heads // 2), dim_feedforward=self.d_v * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.channel_attention_d = nn.TransformerEncoder(chan_attn_layer_d, num_layers=1)
        self.fsin_blocks = nn.ModuleList([
            FreqSparseInteractionBlock(d_model=self.d_v, seq_len=seq_len, d_mem=d_mem)
            for _ in range(4)
        ])


        self.cross_embed_dim = 2 * self.d_v
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.cross_embed_dim, # <--- 修正点: 从 d_model 改为 self.cross_embed_dim
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm_cross1 = nn.LayerNorm(d_model)
        self.norm_cross2 = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

    def forward(self, F_mixed, return_attention_weights=False, return_sparse_attention=False):
        B, L, _ = F_mixed.shape
        x_recovered = self.recover_layer(F_mixed).view(B, L, 4, self.d_v)
        F_in_p = x_recovered[:, :, 0:2, :]  
        F_in_d = x_recovered[:, :, 2:4, :]  
        F_in_p_flat = F_in_p.reshape(B * L, 2, self.d_v)
        F_in_d_flat = F_in_d.reshape(B * L, 2, self.d_v)

        attention_weights = {} if return_attention_weights else None
        if return_attention_weights:
            with torch.no_grad():
                _, p_attn_weights = self.channel_attention_p.layers[0].self_attn(
                    F_in_p_flat, F_in_p_flat, F_in_p_flat
                )
                attention_weights['channel_attention_p'] = p_attn_weights.reshape(B, L, 2, 2).detach().cpu()
            with torch.no_grad():
                _, d_attn_weights = self.channel_attention_d.layers[0].self_attn(
                    F_in_d_flat, F_in_d_flat, F_in_d_flat
                )
                attention_weights['channel_attention_d'] = d_attn_weights.reshape(B, L, 2, 2).detach().cpu()

            F_p_prime_flat = self.channel_attention_p(F_in_p_flat)
            F_d_prime_flat = self.channel_attention_d(F_in_d_flat)
        else:
            F_p_prime_flat = self.channel_attention_p(F_in_p_flat)
            F_d_prime_flat = self.channel_attention_d(F_in_d_flat)

        # Reshape back: [B*L, 2, D_v] -> [B, L, 2, D_v]
        F_p_prime = F_p_prime_flat.reshape(B, L, 2, self.d_v)
        F_d_prime = F_d_prime_flat.reshape(B, L, 2, self.d_v)
        p_connects = [] 
        sparse_attention_weights = [] 
        if return_sparse_attention:
            out_p0, p0, attn0 = self.fsin_blocks[0](F_p_prime[:, :, 0, :], return_attention_weights=True)
            out_p1, p1, attn1 = self.fsin_blocks[1](F_p_prime[:, :, 1, :], return_attention_weights=True)
            sparse_attention_weights.extend([attn0, attn1])
        else:
            out_p0, p0 = self.fsin_blocks[0](F_p_prime[:, :, 0, :])
            out_p1, p1 = self.fsin_blocks[1](F_p_prime[:, :, 1, :])
        p_connects.extend([p0, p1])

        if return_sparse_attention:
            out_d0, p2, attn2 = self.fsin_blocks[2](F_d_prime[:, :, 0, :], return_attention_weights=True)
            out_d1, p3, attn3 = self.fsin_blocks[3](F_d_prime[:, :, 1, :], return_attention_weights=True)
            sparse_attention_weights.extend([attn2, attn3])
        else:
            out_d0, p2 = self.fsin_blocks[2](F_d_prime[:, :, 0, :])
            out_d1, p3 = self.fsin_blocks[3](F_d_prime[:, :, 1, :])
        p_connects.extend([p2, p3])


        F_p_double_prime = torch.stack([out_p0, out_p1], dim=2)  # Shape: [B, L, 2, D_v]
        F_d_double_prime = torch.stack([out_d0, out_d1], dim=2)  # Shape: [B, L, 2, D_v]
        F_p_out = F_p_double_prime.reshape(B, L, 2 * self.d_v)
        F_d_out = F_d_double_prime.reshape(B, L, 2 * self.d_v)
        x_for_cross = torch.cat([F_p_out, F_d_out], dim=-1)  # Shape: [B, L, 4*D_v=D_m]
        residual = x_for_cross
        x = self.norm_cross1(x_for_cross)
        attn_out, cross_attn_weights = self.cross_attention(
            query=F_p_out, key=F_d_out, value=F_d_out
        )
        
        if return_attention_weights:
            attention_weights['cross_attention'] = cross_attn_weights.detach().cpu()
      
        cross_fused = torch.cat([attn_out, torch.zeros_like(F_d_out)], dim=-1)
        x = residual + self.dropout_cross(cross_fused)
        residual = x
        x = self.norm_cross2(x)
        x = self.cross_ffn(x)
        F_fin = residual + self.dropout_cross(x)
        if return_attention_weights and return_sparse_attention:
            return F_fin, p_connects, attention_weights, sparse_attention_weights
        elif return_attention_weights:
            return F_fin, p_connects, attention_weights
        elif return_sparse_attention:
            return F_fin, p_connects, sparse_attention_weights
        else:
            return F_fin, p_connects


