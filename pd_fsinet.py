

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
        """
        Args:
            num_blocks (int):       要堆叠的 PD_FSINet_Block 的数量 (N)。
            seq_len (int):          输入序列长度 (L)。
            pred_len (int):         预测序列长度 (H)。
            in_channels_dyn (int):  动态特征的通道数 (C)。
            in_channels_per (int):  原始周期性特征的通道数 (e.g., 3 for Day, Lunar, Month)。
            d_model (int):          模型隐藏维度 (D_m)。
            n_heads (int):          多头注意力的头数。
            d_ff (int):             前馈网络的隐藏维度。
            dropout (float):        Dropout比率。
            num_bands (int):        PIMDM中频带分解的数量。
            d_mem (int):            FSIN_Block内部频率记忆嵌入的维度。
            embed_dim (int):        每个周期性特征的傅里叶嵌入维度。
        """
        super(PD_FSINet, self).__init__()

        # 计算编码后的周期性特征总维度 (D_p)
        # (傅里叶编码维度 + 年份维度)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_p = (in_channels_per * embed_dim) + 1

        # 1. 实例化共享模块
        self.fourier_embedder = FourierFeatureEmbedding(
            feature_info={'Day_of_Year': 366, 'Lunar_Day': 30, 'Month': 12},
            embed_dim=embed_dim
        )

        self.astro_forecaster = AstroForecaster(
            d_p=self.d_p, c_out=in_channels_dyn, d_model=d_model, n_heads=n_heads,
            n_ast_layers=3,  # 可以设为超参数
            d_ff=d_ff, dropout=dropout, seq_len=seq_len, pred_len=pred_len
        )

        # 2. 实例化 N 个可堆叠的 PD_FSINet_Block
        self.blocks = nn.ModuleList([
            PD_FSINet_Block(
                seq_len=seq_len, pred_len=pred_len, in_channels=in_channels_dyn,
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                num_bands=num_bands, d_mem=d_mem
            ) for _ in range(num_blocks)
        ])

    def forward(self, x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year):
        """
        PD-FSINet 的完整前向传播流程。

        Args:
            x_hist_dyn (torch.Tensor):  历史动态数据, 形状 [B, L, C]。
            x_hist_per (torch.Tensor):  历史原始周期性数据, 形状 [B, L, 3]。
            x_hist_year (torch.Tensor): 历史归一化年份数据, 形状 [B, L, 1]。
            y_fut_per (torch.Tensor):   未来原始周期性数据, 形状 [B, H, 3]。
            y_fut_year (torch.Tensor):  未来归一化年份数据, 形状 [B, H, 1]。

        Returns:
            dict: 一个包含所有必要输出的字典。
        """
        # 1. 编码周期性特征（只处理历史部分）
        x_hist_per_embedded = self.fourier_embedder(x_hist_per)
        x_per_final_hist = torch.cat([x_hist_per_embedded, x_hist_year], dim=-1)

        # 2. 计算天文基线（一次性输出历史+未来）
        astro_full = self.astro_forecaster(x_per_final_hist)  # [B, L+H, C]
        B_ast_hist = astro_full[:, :self.seq_len, :]          # 历史部分 [B, L, C]
        Y_ast = astro_full[:, self.seq_len:, :]               # 未来部分 [B, H, C]

        # 3. 层级化残差学习
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

            # 收集每个块的输出
            Y_comp_list.append(Y_comp)
            p_connects_all_blocks.extend(p_connects)  # 将所有概率矩阵展平到一个列表中

            # 更新残差
            current_residual = X_res

        # 4. 最终预测合成
        Y_res_total = torch.stack(Y_comp_list, dim=0).sum(dim=0)
        Y_hat = Y_ast + Y_res_total

        # 准备输出字典
        outputs = {
            "prediction": Y_hat,  # 最终预测
            "astro_baseline_hist": B_ast_hist,  # 历史天文基线 (向后兼容)
            "prediction_astro": Y_ast,  # 预测天文基线 (向后兼容)
            "astro_baseline_full": astro_full,  # 完整天文基线 (新版本)
            "sparse_probs": p_connects_all_blocks  # 所有连接概率 (用于L_sparse)
        }

        return outputs
