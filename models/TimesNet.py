import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InceptionBlock(nn.Module):
    """Multi-scale 2D convolution with square kernels (matching TSLib)."""
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        return sum(out) / len(out)


def FFT_for_Period(x, k=2):
    """Batch-averaged FFT period detection (matching TSLib)."""
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # Average amplitudes across batch and channels
    frequency_list = xf.abs().mean(0).mean(-1)  # [T//2+1]
    frequency_list[0] = 0  # remove DC
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, xf.abs().mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k

        self.conv = nn.Sequential(
            InceptionBlock(configs.d_model, configs.d_ff, configs.num_kernels),
            nn.GELU(),
            InceptionBlock(configs.d_ff, configs.d_model, configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # Pad to make length divisible by period
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # Reshape: [B, length//period, period, N] -> [B, N, length//period, period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D inception convolution
            out = self.conv(out)
            # Back to 1D: [B, N, num_seg, period] -> [B, T, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)  # [B, T, N, top_k]
        # Adaptive aggregation with per-sample weights
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)  # [B, T, N]
        # Residual
        res = res + x
        return res


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=1, padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(
            self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu'
        )

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class Model(nn.Module):
    """TimesNet matching TSLib architecture."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers

        # Predict linear: extend sequence
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)

        # Embedding
        self.enc_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # TimesBlocks
        self.blocks = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Output projection
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, C = x_enc.shape

        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        # Embed
        enc_out = self.dropout(self.enc_embedding(x_enc))

        # Extend sequence
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesBlocks (residual is inside TimesBlock now)
        for block in self.blocks:
            enc_out = self.layer_norm(block(enc_out))

        # Project to output channels
        dec_out = self.projection(enc_out)

        # Denormalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)

        return dec_out[:, -self.pred_len:, :]
