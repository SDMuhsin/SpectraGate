"""
SpectralMixer: Multivariate time series forecaster exploiting joint
spectral-spatial sparsity.

Architecture:
  1. rFFT along time → truncate to K dominant frequency bins
  2. Per-frequency channel mixing: at each spectral position, mix C channels
     through a low-rank bottleneck (C → H → C), with residual + pre-norm
  3. Per-channel MLP: map 2K spectral features → d_model → pred_len

Design grounded in empirical findings:
- Temporal energy concentrated in K << 49 frequency bins (both Weather & ECL)
- At dominant frequencies, cross-channel variance is low-rank (compressible)
- Spectral truncation removes exactly those bins where channels are full-rank
"""
import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.cut_freq = configs.cut_freq
        self.hidden = configs.hidden_size
        self.d_model = configs.d_model
        self.n_layers = getattr(configs, 'e_layers', 1)
        self.drop = configs.dropout

        # Instance normalization
        self.revin = RevIN(self.channels, affine=True, subtract_last=False)

        # Stacked per-frequency channel mixers (shared across spectral positions)
        self.mixers = nn.ModuleList()
        self.mixer_norms = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mixers.append(nn.Sequential(
                nn.Linear(self.channels, self.hidden),
                nn.GELU(),
                nn.Dropout(self.drop),
                nn.Linear(self.hidden, self.channels),
                nn.Dropout(self.drop),
            ))
            self.mixer_norms.append(nn.LayerNorm(self.channels))

        # Per-channel MLP: 2K spectral features → d_model → pred_len
        self.mlp_norm = nn.LayerNorm(2 * self.cut_freq)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.cut_freq, self.d_model),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape

        # 1. Instance normalization
        x = self.revin(x_enc, 'norm')

        # 2. rFFT + spectral truncation
        x_freq = torch.fft.rfft(x, dim=1)[:, :self.cut_freq, :]  # [B, K, C]

        # 3. Stack real/imag as per-frequency channel vectors: [B, 2K, C]
        h = torch.cat([x_freq.real, x_freq.imag], dim=1)  # [B, 2K, C]

        # 4. Per-frequency channel mixing with residual + pre-norm
        for mixer, norm in zip(self.mixers, self.mixer_norms):
            h = h + mixer(norm(h))  # [B, 2K, C]

        # 5. Per-channel MLP: spectral features → forecast
        h = h.permute(0, 2, 1)  # [B, C, 2K]
        h = self.mlp_norm(h)
        out = self.mlp(h)  # [B, C, pred_len]
        out = out.permute(0, 2, 1)  # [B, pred_len, C]

        # 6. Denormalize
        out = self.revin(out, 'denorm')
        return out
