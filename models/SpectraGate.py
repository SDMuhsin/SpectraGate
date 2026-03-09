import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class Model(nn.Module):
    """Per-channel spectral self-gating forecaster.

    Truncates the rFFT to K dominant frequency bins, applies per-channel
    self-gating (each channel's spectral profile generates adaptive frequency
    emphasis via a shared gate network), then maps gated features to
    predictions via a shared MLP. Zero cross-channel interaction.

    MLP configurations (controlled by e_layers and d_layers):
      e_layers >= 2: Original 3-layer MLP with D×D hidden layer
      e_layers == 1, d_layers == 0: 2-layer MLP (no D×D)
      e_layers == 1, d_layers > 0: 2-layer MLP + residual low-rank D×D
        The residual adds: y = x + Linear(r→D)(GELU(Linear(D→r)(x)))
        where r = d_layers. This preserves depth with minimal params.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        K = getattr(configs, 'cut_freq', 24)
        D = getattr(configs, 'd_model', 512)
        D_gate = getattr(configs, 'd_ff', 128)
        drop = configs.dropout
        mlp_layers = getattr(configs, 'e_layers', 2)
        R = getattr(configs, 'd_layers', 0)

        self.K = K
        self.drop = drop
        feat_dim = 2 * K  # real + imag stacked

        self.revin = RevIN(self.channels, affine=True, subtract_last=False)
        self.ln_in = nn.LayerNorm(feat_dim)

        # Self-gate: shared across channels, input-adaptive
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, D_gate),
            nn.GELU(),
            nn.Linear(D_gate, feat_dim),
            nn.Sigmoid(),
        )

        # Shared MLP: processes gated spectral features per-channel
        self.use_residual = (mlp_layers == 1 and R > 0)

        if mlp_layers >= 2:
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, D),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(D, self.pred_len),
            )
        elif self.use_residual:
            # 2-layer MLP + residual low-rank D×D
            self.expand = nn.Linear(feat_dim, D)
            self.res_down = nn.Linear(D, R)
            self.res_up = nn.Linear(R, D)
            self.project = nn.Linear(D, self.pred_len)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, D),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(D, self.pred_len),
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, L, C]
        x = self.revin(x_enc, 'norm')           # [B, L, C]
        x = x.permute(0, 2, 1)                  # [B, C, L]

        # Spectral decomposition: rFFT along time, keep K bins
        xf = torch.fft.rfft(x, dim=2)           # [B, C, L//2+1] complex
        xf = xf[:, :, :self.K]                  # [B, C, K] complex

        # Stack real and imaginary parts
        xr = torch.cat([xf.real, xf.imag], dim=2)  # [B, C, 2K]
        xr = self.ln_in(xr)

        # Per-channel self-gate: adapt frequency emphasis
        g = self.gate(xr)                        # [B, C, 2K]
        xr = xr * g                             # [B, C, 2K]

        # Shared MLP: spectral features → prediction
        if self.use_residual:
            h = F.gelu(self.expand(xr))
            h = F.dropout(h, p=self.drop, training=self.training)
            h = h + self.res_up(F.gelu(self.res_down(h)))  # residual low-rank
            h = F.gelu(h)
            h = F.dropout(h, p=self.drop, training=self.training)
            out = self.project(h)
        else:
            out = self.mlp(xr)                   # [B, C, pred_len]

        out = out.permute(0, 2, 1)               # [B, pred_len, C]
        out = self.revin(out, 'denorm')
        return out
