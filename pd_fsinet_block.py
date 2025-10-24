import torch
import torch.nn as nn


try:
    from .pimdm import PIMDM
    from .pgtsi_net import PGTSINet
    from .heads import PredictionHead, BackcastHead
except ImportError:
    from pimdm import PIMDM
    from pgtsi_net import PGTSINet
    from heads import PredictionHead, BackcastHead

class PD_FSINet_Block(nn.Module):

    def __init__(self, seq_len, pred_len, in_channels, d_model, n_heads, d_ff,
                 dropout, num_bands, d_mem):

        self.pimdm = PIMDM(
            seq_len=seq_len,
            in_channels=in_channels,
            d_model=d_model,
            num_bands=num_bands
        )

        self.pgtsi_net = PGTSINet(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            d_mem=d_mem
        )

        self.prediction_head = PredictionHead(
            seq_len=seq_len,
            d_model=d_model,
            pred_len=pred_len,
            n_channels=in_channels
        )

        self.backcast_head = BackcastHead(
            d_model=d_model,
            n_channels=in_channels
        )

    def forward(self, x_in, B_ast_hist, is_first_block: bool):
        F_mixed = self.pimdm(x_in, B_ast_hist, is_first_block)
        F_fin, p_connects = self.pgtsi_net(F_mixed)
        Y_comp = self.prediction_head(F_fin)
        X_rec = self.backcast_head(F_fin)
        X_res = x_in - X_rec
        return Y_comp, X_res, p_connects
