
import torch
import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, seq_len, d_model, pred_len, n_channels, **kwargs):
        super(PredictionHead, self).__init__()
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.feature_projection = nn.Linear(d_model, n_channels)

    def forward(self, x):
        y_pred = self.feature_projection(x_temp.transpose(1, 2)) # [B, H, C]
        return y_pred


class BackcastHead(nn.Module):
    def __init__(self, d_model, n_channels, hidden_dim_factor=2):

        super(BackcastHead, self).__init__()

        hidden_dim = d_model * hidden_dim_factor

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_channels)
        )

    def forward(self, x):
        y_backcast = self.mlp(x)
        return y_backcast
