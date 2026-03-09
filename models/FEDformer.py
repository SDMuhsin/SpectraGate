import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Autoformer_EncDec import series_decomp


class FourierBlock(nn.Module):
    """Frequency Enhanced Block for self-attention (FEDformer).

    Applies a learnable frequency filter by:
    1. FFT the input
    2. Apply complex linear weights to selected modes
    3. IFFT back
    """
    def __init__(self, d_model, n_heads, seq_len, modes=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = min(modes, seq_len // 2 + 1)
        self.scale = 1.0 / (self.d_head ** 0.5)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Complex-valued weights for frequency-domain interaction
        self.weights = nn.Parameter(
            torch.randn(n_heads, self.modes, self.d_head, dtype=torch.cfloat) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in, k_in=None, v_in=None):
        if k_in is None:
            k_in = q_in
        if v_in is None:
            v_in = q_in

        B, L, _ = q_in.shape

        q = self.q_proj(q_in).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(v_in).view(B, v_in.shape[1], self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # [B, H, L, d_head]

        # FFT along time dimension
        v_ft = torch.fft.rfft(v, dim=2)  # [B, H, L//2+1, d_head]

        # Apply learnable frequency filter to selected modes
        out_ft = torch.zeros_like(v_ft)
        modes = min(self.modes, v_ft.shape[2])
        out_ft[:, :, :modes, :] = v_ft[:, :, :modes, :] * self.weights[:, :modes, :]

        # IFFT back
        out = torch.fft.irfft(out_ft, n=L, dim=2)  # [B, H, L, d_head]

        # Reshape back
        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out


class FourierCrossAttention(nn.Module):
    """Frequency Enhanced Cross-Attention for decoder."""
    def __init__(self, d_model, n_heads, seq_len_q, seq_len_kv, modes=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.modes = min(modes, min(seq_len_q, seq_len_kv) // 2 + 1)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.weights = nn.Parameter(
            torch.randn(n_heads, self.modes, self.d_head, dtype=torch.cfloat) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in, k_in, v_in):
        B, L_q, _ = q_in.shape
        _, L_kv, _ = k_in.shape

        v = self.v_proj(v_in).view(B, L_kv, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # FFT of values
        v_ft = torch.fft.rfft(v, dim=2)

        # Apply frequency filter
        out_ft = torch.zeros(B, self.n_heads, L_q // 2 + 1, self.d_head,
                             dtype=torch.cfloat, device=v.device)
        modes = min(self.modes, v_ft.shape[2], out_ft.shape[2])
        out_ft[:, :, :modes, :] = v_ft[:, :, :modes, :] * self.weights[:, :modes, :]

        # IFFT with target length
        out = torch.fft.irfft(out_ft, n=L_q, dim=2)

        out = out.permute(0, 2, 1, 3).reshape(B, L_q, self.d_model)
        out = self.dropout(self.out_proj(out))
        return out


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, modes=32,
                 dropout=0.1, activation='gelu', moving_avg=25):
        super().__init__()
        self.attention = FourierBlock(d_model, n_heads, seq_len, modes, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        x, _ = self.decomp1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, _ = self.decomp2(x + y)
        return x


class FEDformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, c_out, seq_len_q, seq_len_kv,
                 modes=32, dropout=0.1, activation='gelu', moving_avg=25):
        super().__init__()
        self.self_attention = FourierBlock(d_model, n_heads, seq_len_q, modes, dropout)
        self.cross_attention = FourierCrossAttention(
            d_model, n_heads, seq_len_q, seq_len_kv, modes, dropout
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.projection = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode='circular')

    def forward(self, x, cross):
        attn_out = self.self_attention(x)
        x = x + self.dropout(attn_out)
        x, trend1 = self.decomp1(x)
        attn_out = self.cross_attention(x, cross, cross)
        x = x + self.dropout(attn_out)
        x, trend2 = self.decomp2(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        trend = trend1 + trend2 + trend3
        trend = self.projection(trend.permute(0, 2, 1)).transpose(1, 2)
        return x, trend


class Model(nn.Module):
    """FEDformer with Fourier-enhanced attention and series decomposition."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        modes = 32

        self.decomp = series_decomp(configs.moving_avg)

        # Embeddings
        self.enc_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, configs.seq_len, configs.d_model) * 0.02)
        self.dec_embedding = nn.Linear(configs.enc_in, configs.d_model)
        self.dec_pos = nn.Parameter(
            torch.randn(1, configs.label_len + configs.pred_len, configs.d_model) * 0.02
        )

        dec_len = configs.label_len + configs.pred_len

        # Encoder
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.seq_len, modes, configs.dropout,
                configs.activation, configs.moving_avg
            )
            for _ in range(configs.e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(configs.d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            FEDformerDecoderLayer(
                configs.d_model, configs.n_heads, configs.d_ff, configs.enc_in,
                dec_len, configs.seq_len, modes, configs.dropout,
                configs.activation, configs.moving_avg
            )
            for _ in range(configs.d_layers)
        ])
        self.decoder_norm = nn.LayerNorm(configs.d_model)

        self.projection = nn.Linear(configs.d_model, configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mean = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        seasonal, trend = self.decomp(x_enc)

        seasonal_dec = x_dec
        trend_dec = torch.cat([x_enc[:, -self.label_len:, :], mean], dim=1)

        # Encoder
        enc_out = self.enc_embedding(x_enc) + self.enc_pos
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        enc_out = self.encoder_norm(enc_out)

        # Decoder
        dec_out = self.dec_embedding(seasonal_dec) + self.dec_pos
        trend_out = trend_dec
        for layer in self.decoder_layers:
            dec_out, trend_part = layer(dec_out, enc_out)
            trend_out = trend_out + trend_part
        dec_out = self.decoder_norm(dec_out)

        dec_out = self.projection(dec_out) + trend_out

        return dec_out[:, -self.pred_len:, :]
