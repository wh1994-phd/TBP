

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from torch.utils.data import Dataset, DataLoader

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class AstroEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu"):
        super(AstroEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class AstroForecaster(nn.Module):
    def __init__(self, d_p, c_out, d_model, n_heads, n_ast_layers, d_ff, dropout, seq_len=96, pred_len=30):
        super(AstroForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_len = seq_len + pred_len  # 输出历史+未来长度
        
        self.embedding = DataEmbedding(c_in=d_p, d_model=d_model, dropout=dropout)
        self.encoder_stack = nn.ModuleList(
            [AstroEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_ast_layers)])
        
        # 修改投影层：从历史长度映射到历史+未来长度
        self.projection = nn.Linear(d_model, c_out)
        self.output_projection = nn.Linear(seq_len * c_out, self.output_len * c_out)

    def forward(self, x_per):
        """
        前向传播：从历史时间特征预测历史+未来的天文基线
        
        Args:
            x_per: 历史时间特征 [B, seq_len, d_p]
            
        Returns:
            天文基线预测 [B, seq_len + pred_len, c_out]
        """
        B, L, _ = x_per.shape
        
        # Transformer编码
        enc_out = self.embedding(x_per)  # [B, L, d_model]
        for encoder_layer in self.encoder_stack:
            enc_out = encoder_layer(enc_out)
        
        # 投影到输出特征维度
        projected = self.projection(enc_out)  # [B, L, c_out]
        
        # 扩展到历史+未来长度
        flattened = projected.reshape(B, -1)  # [B, L * c_out]
        output = self.output_projection(flattened)  # [B, (L+H) * c_out]
        
        return output.reshape(B, self.output_len, -1)  # [B, L+H, c_out]
