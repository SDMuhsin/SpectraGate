"""
Compact Spectro-Spatial Forecaster.

Two-layer frequency-domain forecaster that projects multivariate input into a
learned low-rank channel subspace, applies two sequential truncated-frequency
complex linear transforms with GELU nonlinearity between them, and projects back.
The first layer extracts features at the input temporal resolution; the second
maps features to the forecast horizon. Exploits spectral concentration and
channel rank deficiency -- two structural properties of meteorological time series.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        self.rank = getattr(configs, 'rank', 14)
        self.cut_freq = getattr(configs, 'cut_freq', 24)

        # Reversible instance normalization
        self.revin = RevIN(self.channels, affine=True)

        # Channel subspace projections (no bias -- input is normalized)
        self.proj_down = nn.Linear(self.channels, self.rank, bias=False)
        self.proj_up = nn.Linear(self.rank, self.channels, bias=False)

        # Block 1: feature extraction (same-length: K -> K per component)
        self.block1 = nn.ModuleList([
            nn.Linear(self.cut_freq, self.cut_freq).to(torch.cfloat)
            for _ in range(self.rank)
        ])

        # Block 2: forecasting (K -> freq_out per component)
        total_len = self.seq_len + self.pred_len
        self.length_ratio = total_len / self.seq_len
        freq_out = int(self.cut_freq * self.length_ratio)
        self.block2 = nn.ModuleList([
            nn.Linear(self.cut_freq, freq_out).to(torch.cfloat)
            for _ in range(self.rank)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, C]

        # Reversible instance normalization
        x = self.revin(x_enc, 'norm')

        # Project to low-rank channel subspace: [B, T, C] -> [B, T, R]
        x = self.proj_down(x)

        # Block 1: feature extraction (same-length frequency transform)
        x_freq = torch.fft.rfft(x, dim=1)[:, :self.cut_freq, :]  # [B, K, R]
        B = x_freq.shape[0]
        out_freq1 = torch.zeros_like(x_freq)
        for i in range(self.rank):
            out_freq1[:, :, i] = self.block1[i](x_freq[:, :, i])
        x = torch.fft.irfft(out_freq1, n=self.seq_len, dim=1)  # [B, seq_len, R]

        # Nonlinearity between blocks
        x = F.gelu(x)

        # Block 2: forecasting (length-changing frequency transform)
        x_freq2 = torch.fft.rfft(x, dim=1)[:, :self.cut_freq, :]  # [B, K, R]
        freq_out_size = int(self.cut_freq * self.length_ratio)
        out_freq2 = torch.zeros(
            B, freq_out_size, self.rank,
            dtype=x_freq2.dtype, device=x_freq2.device
        )
        for i in range(self.rank):
            out_freq2[:, :, i] = self.block2[i](x_freq2[:, :, i])

        # iRFFT reconstruction
        total_len = self.seq_len + self.pred_len
        out = torch.fft.irfft(out_freq2, n=total_len, dim=1)  # [B, total_len, R]
        out = out * self.length_ratio  # energy scaling

        # Extract forecast window
        out = out[:, -self.pred_len:, :]  # [B, pred_len, R]

        # Project back to original channel space
        out = self.proj_up(out)  # [B, pred_len, C]

        # Denormalize
        out = self.revin(out, 'denorm')

        return out
