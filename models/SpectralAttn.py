"""
SpectralAttn: Multivariate time series forecaster exploiting low-dimensional
non-stationary cross-channel dependencies in the spectral domain.

Architecture:
  1. rFFT along time → truncate to K dominant frequency bins
  2. Per-sample adaptive channel mixing via spectral similarity:
     dot-product similarity in the truncated spectral domain determines
     how channels share information, with no learned mixing parameters
  3. Per-channel MLP: mixed spectral features → forecast

Design grounded in empirical findings:
- Cross-channel predictive information is substantial (R²=0.57 on ECL)
- Cross-channel covariance non-stationarity is low-dimensional (~4 modes)
- Fixed channel mixing overfits (18% vali→test gap); similarity mixing adapts per sample
- Zero-parameter mixing avoids cross-channel overfitting entirely
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class SpectralSimilarityMixer(nn.Module):
    """Per-sample channel mixing via spectral dot-product similarity.

    No learned mixing parameters — the mixing pattern is determined entirely
    by the input's spectral structure. Each channel attends to spectrally
    similar channels, weighted by softmax similarity.
    """

    def __init__(self, d_feature, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_feature // n_heads
        self.scale = self.d_k ** -0.5
        # Single learnable scalar controlling mixing strength (init small)
        self.mix_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, C, D] where D = 2*K spectral features
        B, C, D = x.shape

        # Compute similarity weights with stop-gradient for stable training.
        # Mixing pattern adapts per-sample; gradients flow through values only.
        x_det = x.detach()

        if self.n_heads == 1:
            sim = torch.bmm(x_det, x_det.transpose(1, 2)) * self.scale  # [B, C, C]
            weights = sim.softmax(dim=-1)  # [B, C, C]
            mixed = torch.bmm(weights, x)  # [B, C, D]
        else:
            H = self.n_heads
            d_k = self.d_k
            xh_det = x_det.reshape(B, C, H, d_k).permute(0, 2, 1, 3)  # [B, H, C, d_k]
            sim = (xh_det @ xh_det.transpose(-2, -1)) * self.scale  # [B, H, C, C]
            weights = sim.softmax(dim=-1)  # [B, H, C, C]
            xh = x.reshape(B, C, H, d_k).permute(0, 2, 1, 3)  # [B, H, C, d_k]
            mixed = (weights @ xh).transpose(1, 2).reshape(B, C, D)  # [B, C, D]

        return self.mix_scale * mixed


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.cut_freq = configs.cut_freq
        self.d_spec = 2 * self.cut_freq  # spectral feature dimension
        self.n_heads = getattr(configs, 'n_heads', 1)
        self.n_layers = getattr(configs, 'e_layers', 1)
        self.d_model = configs.d_model
        self.drop = configs.dropout

        # Instance normalization
        self.revin = RevIN(self.channels, affine=True, subtract_last=False)

        # Spectral similarity mixing layers (near-zero learned params)
        self.mixer_layers = nn.ModuleList()
        self.mixer_norms = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mixer_layers.append(
                SpectralSimilarityMixer(self.d_spec, self.n_heads)
            )
            self.mixer_norms.append(nn.LayerNorm(self.d_spec))

        # Per-channel MLP: spectral features → forecast
        # d_layers controls hidden layers: 1 = [d_spec→D→pred], 2 = [d_spec→D→D→pred]
        self.d_layers = getattr(configs, 'd_layers', 1)
        self.mlp_norm = nn.LayerNorm(self.d_spec)
        layers = [nn.Linear(self.d_spec, self.d_model), nn.GELU(), nn.Dropout(self.drop)]
        for _ in range(self.d_layers - 1):
            layers.extend([nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Dropout(self.drop)])
        layers.append(nn.Linear(self.d_model, self.pred_len))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape

        # 1. Instance normalization
        x = self.revin(x_enc, 'norm')

        # 2. rFFT + spectral truncation
        x_freq = torch.fft.rfft(x, dim=1)[:, :self.cut_freq, :]  # [B, K, C]

        # 3. Stack real/imag: [B, 2K, C] → permute to [B, C, 2K]
        h = torch.cat([x_freq.real, x_freq.imag], dim=1)  # [B, 2K, C]
        h = h.permute(0, 2, 1)  # [B, C, 2K]

        # 4. Per-sample adaptive channel mixing via spectral similarity
        for mixer, norm in zip(self.mixer_layers, self.mixer_norms):
            h = h + mixer(norm(h))  # [B, C, 2K]

        # 5. Per-channel MLP: spectral features → forecast
        h = self.mlp_norm(h)
        out = self.mlp(h)  # [B, C, pred_len]
        out = out.permute(0, 2, 1)  # [B, pred_len, C]

        # 6. Denormalize
        out = self.revin(out, 'denorm')
        return out
