import torch
import torch.nn as nn

try:
    from .pd_fsinet_block import PD_FSINet_Block
    from .astro_forecaster import AstroForecaster
    from utils.timefeatures import FourierFeatureEmbedding
    import torch.nn.functional as F
except ImportError:
    from pd_fsinet_block import PD_FSINet_Block
    from astro_forecaster import AstroForecaster
    import sys, os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from utils.timefeatures import FourierFeatureEmbedding

class PD_FSINet(nn.Module):
    def __init__(self, num_blocks, seq_len, pred_len, in_channels_dyn, in_channels_per,
                 d_model, n_heads, d_ff, dropout, num_bands, d_mem, embed_dim):
        super(PD_FSINet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_p = (in_channels_per * embed_dim) + 1

        self.fourier_embedder = FourierFeatureEmbedding(
            feature_info={'Day_of_Year': 366, 'Lunar_Day': 30, 'Month': 12},
            embed_dim=embed_dim
        )
        self.astro_forecaster = AstroForecaster(
            d_p=self.d_p, c_out=in_channels_dyn, d_model=d_model, n_heads=n_heads,
            n_ast_layers=3,  
            d_ff=d_ff, dropout=dropout, seq_len=seq_len, pred_len=pred_len
        )
        self.blocks = nn.ModuleList([
            PD_FSINet_Block(
                seq_len=seq_len, pred_len=pred_len, in_channels=in_channels_dyn,
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                num_bands=num_bands, d_mem=d_mem
            ) for _ in range(num_blocks)
        ])

    def forward(self, x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year):
        x_hist_per_embedded = self.fourier_embedder(x_hist_per)
        x_per_final_hist = torch.cat([x_hist_per_embedded, x_hist_year], dim=-1)
        astro_full = self.astro_forecaster(x_per_final_hist)  # [B, L+H, C]
        B_ast_hist = astro_full[:, :self.seq_len, :]          
        Y_ast = astro_full[:, self.seq_len:, :]               

        current_residual = x_hist_dyn
        Y_comp_list = []
        p_connects_all_blocks = []

        for i, block in enumerate(self.blocks):
            is_first = (i == 0)

            Y_comp, X_res, p_connects = block(
                x_in=current_residual,
                B_ast_hist=B_ast_hist if is_first else None,
                is_first_block=is_first
            )
            Y_comp_list.append(Y_comp)
            p_connects_all_blocks.extend(p_connects)
            current_residual = X_res
        Y_res_total = torch.stack(Y_comp_list, dim=0).sum(dim=0)
        Y_hat = Y_ast + Y_res_total

        outputs = {
            "prediction": Y_hat,  
            "astro_baseline_hist": B_ast_hist,  
            "prediction_astro": Y_ast,  
            "astro_baseline_full": astro_full,  
            "sparse_probs": p_connects_all_blocks  
        }

        return outputs
