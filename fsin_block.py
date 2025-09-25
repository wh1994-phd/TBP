

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FreqSparseInteractionBlock(nn.Module):


    def __init__(self, d_model, seq_len, d_mem, top_k_factor=25):
        super(FreqSparseInteractionBlock, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.freq_len = seq_len // 2 + 1

        self.top_k = int(top_k_factor * math.log(self.freq_len))

        self.top_k = max(1, min(self.top_k, self.freq_len))


        self.M_frq = nn.Parameter(torch.randn(self.freq_len, d_mem))
        nn.init.xavier_uniform_(self.M_frq)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_mem, 
            nhead=8,  
            dim_feedforward=d_mem * 4, 
            dropout=0.1,
            activation='gelu',  
            batch_first=True  
        )
        self.connection_predictor = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x_ts, return_attention_weights=False):
       
        B, L, _ = x_ts.shape
        x_freq = torch.fft.rfft(x_ts, n=L, dim=1)
        M_frq_ctx = self.connection_predictor(self.M_frq.unsqueeze(0)).squeeze(0) # 输出 squeeze(0) -> [F, D_mem]
        logits = M_frq_ctx @ M_frq_ctx.T / (self.M_frq.shape[-1] ** 0.5)
        p_connect = torch.sigmoid(logits)  # 形状: [F, F]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, -float('inf'))
        sparse_logits.scatter_(-1, top_k_indices, top_k_logits)
        attention_weights = F.softmax(sparse_logits, dim=-1)  # 形状: [F, F]
        C_frq_complex = attention_weights.to(x_freq.dtype)
        x_freq_interacted = torch.einsum('bfd,gf->bgd', x_freq, C_frq_complex)  # 形状: (B, F, D_v)
        x_ts_interacted = torch.fft.irfft(x_freq_interacted, n=L, dim=1)  # 形状: (B, L, D_v)
        norm_layer = nn.LayerNorm(self.d_model, device=x_ts.device)
        Z_out = norm_layer(x_ts + x_ts_interacted)
        if return_attention_weights:
            return Z_out, p_connect, attention_weights
        else:
            return Z_out, p_connect
