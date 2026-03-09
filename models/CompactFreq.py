"""
Compact Spectro-Spatial Forecaster (v8).

Dual-path frequency-domain forecaster that projects multivariate input into a
learned low-rank channel subspace, applies two sequential truncated-frequency
complex linear transforms with cross-component mixing and GELU nonlinearity,
combines with both a time-domain projection shortcut and a spectral residual
that preserves input spectral structure in the forecast. Exploits spectral
concentration and channel rank deficiency in meteorological time series.
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

        self.rank = getattr(configs, 'rank', 18)
        self.cut_freq = getattr(configs, 'cut_freq', 24)
        self.dropout = getattr(configs, 'dropout', 0)

        R = self.rank
        K = self.cut_freq

        # Reversible instance normalization
        self.revin = RevIN(self.channels, affine=True)

        # Channel subspace projections (no bias -- input is normalized)
        self.proj_down = nn.Linear(self.channels, R, bias=False)
        self.proj_up = nn.Linear(R, self.channels, bias=False)

        # Block 1: feature extraction (same-length: K -> K per component)
        self.freq1 = nn.ModuleList([
            nn.Linear(K, K).to(torch.cfloat)
            for _ in range(R)
        ])
        self.mix1 = nn.Linear(R, R).to(torch.cfloat)

        # Block 2: forecasting (K -> freq_out per component)
        total_len = self.seq_len + self.pred_len
        self.length_ratio = total_len / self.seq_len
        freq_out = int(K * self.length_ratio)
        self.freq_out = freq_out
        self.freq2 = nn.ModuleList([
            nn.Linear(K, freq_out).to(torch.cfloat)
            for _ in range(R)
        ])
        self.mix2 = nn.Linear(R, R).to(torch.cfloat)

        # Spectral residual: learnable per-frequency per-component weight
        # Preserves input spectral structure in the forecast output
        self.spec_res = nn.Parameter(torch.zeros(K, R))

        # Time-domain shortcut: bottleneck projection seq_len -> D -> pred_len
        D = getattr(configs, 'time_bottleneck', None) or K
        self.time_down = nn.Linear(self.seq_len, D)
        self.time_up = nn.Linear(D, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, C]
        B = x_enc.shape[0]
        K = self.cut_freq
        R = self.rank

        # Reversible instance normalization
        x = self.revin(x_enc, 'norm')

        # Project to low-rank channel subspace: [B, T, C] -> [B, T, R]
        x = self.proj_down(x)

        # Time-domain shortcut: [B, seq_len, R] -> [B, pred_len, R]
        xt = x.transpose(1, 2)  # [B, R, seq_len]
        time_out = self.time_up(F.gelu(self.time_down(xt))).transpose(1, 2)

        # Block 1: feature extraction (same-length frequency transform)
        x_freq = torch.fft.rfft(x, dim=1)[:, :K, :]  # [B, K, R]
        # Save input spectrum for spectral residual
        raw_freq = x_freq

        out1 = torch.zeros_like(x_freq)
        for i in range(R):
            out1[:, :, i] = self.freq1[i](x_freq[:, :, i])
        out1 = self.mix1(out1)  # [B, K, R] -- mix across components
        x = torch.fft.irfft(out1, n=self.seq_len, dim=1)  # [B, seq_len, R]

        # Nonlinearity and regularization between blocks
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Block 2: forecasting (length-changing frequency transform)
        x_freq2 = torch.fft.rfft(x, dim=1)[:, :K, :]  # [B, K, R]
        out2 = torch.zeros(
            B, self.freq_out, R,
            dtype=x_freq2.dtype, device=x_freq2.device
        )
        for i in range(R):
            out2[:, :, i] = self.freq2[i](x_freq2[:, :, i])
        out2 = self.mix2(out2)  # [B, freq_out, R]

        # Add spectral residual: inject input spectrum into first K output bins
        spec_weight = self.spec_res.to(raw_freq.dtype)  # [K, R] real→complex
        out2[:, :K, :] = out2[:, :K, :] + raw_freq * spec_weight

        # iRFFT reconstruction
        total_len = self.seq_len + self.pred_len
        out = torch.fft.irfft(out2, n=total_len, dim=1)  # [B, total_len, R]
        out = out * self.length_ratio  # energy scaling

        # Extract forecast window and add time shortcut
        out = out[:, -self.pred_len:, :]  # [B, pred_len, R]
        out = out + time_out  # combine with time path

        # Project back to original channel space
        out = self.proj_up(out)  # [B, pred_len, C]

        # Denormalize
        out = self.revin(out, 'denorm')

        return out
