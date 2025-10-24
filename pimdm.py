import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_band_decomposition(x, num_bands):
    B, T, C = x.shape
    x_freq = torch.fft.rfft(x, dim=1)
    freq_len = x_freq.shape[1]
    band_width = (freq_len + num_bands - 1) // num_bands
    bands = []
    for i in range(num_bands):
        mask = torch.zeros_like(x_freq)
        start_idx = i * band_width
        end_idx = min(start_idx + band_width, freq_len)
        mask[:, start_idx:end_idx, :] = 1
        x_band_freq = x_freq * mask
        x_band_time = torch.fft.irfft(x_band_freq, n=T, dim=1)
        bands.append(x_band_time)
    return bands


class ScaleMixingBlock(nn.Module):
    def __init__(self, d_model, seq_len):
        super(ScaleMixingBlock, self).__init__()
        self.temporal_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.GELU(), nn.Linear(seq_len, seq_len)
        )
        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm_temporal_in = nn.LayerNorm(d_model)
        self.norm_feature_in = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, F_mix_prev, Z_current):
        residual = F_mix_prev
        x = self.norm_temporal_in(F_mix_prev)
        x_temp = self.temporal_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x_temp
        residual = x
        x = self.norm_feature_in(x)
        x_feat = self.feature_mixer(x)
        x = residual + x_feat
        output = self.norm_final(x + Z_current)
        return output



class PIMDM(nn.Module):
    def __init__(self, seq_len, in_channels, d_model, num_bands):
        self.mixing_blocks = nn.ModuleList([
            ScaleMixingBlock(d_model, seq_len) for _ in range(num_bands)
        ])
    def forward(self, x_in, B_ast_hist, is_first_block: bool):
        if is_first_block:
            if B_ast_hist is None:
                raise ValueError("")

            perturbation_bands = fourier_band_decomposition(x_pert, self.num_bands)
            all_components = [B_ast_hist] + perturbation_bands
        else:
            residual_bands = fourier_band_decomposition(x_in, self.num_bands)
            all_components = residual_bands
        Z_components = [self.embed_layer(comp) for comp in all_components]
        for i, Z_current in enumerate(Z_components[1:]):
            F_mix = self.mixing_blocks[i](F_mix, Z_current)

        return F_mix

