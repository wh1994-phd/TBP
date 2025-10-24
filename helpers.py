import torch
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
