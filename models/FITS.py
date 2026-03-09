"""
FITS: Frequency Interpolation Time Series forecasting.
Paper: https://arxiv.org/abs/2307.03756
Official code: https://github.com/VEWOXIC/FITS
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.cut_freq = configs.cut_freq
        self.individual = getattr(configs, 'individual', True)

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        freq_out = int(self.cut_freq * self.length_ratio)

        if self.individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(self.cut_freq, freq_out).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.freq_upsampler = nn.Linear(
                self.cut_freq, freq_out
            ).to(torch.cfloat)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, C]

        # Instance normalization
        x_mean = torch.mean(x_enc, dim=1, keepdim=True)
        x = x_enc - x_mean
        x_var = torch.var(x, dim=1, keepdim=True)
        x_stdev = torch.sqrt(x_var + 1e-5)
        x = x / x_stdev

        # FFT along time dimension (dim=1)
        x_freq = torch.fft.rfft(x, dim=1)  # [B, seq_len//2+1, C]

        # Low-pass: keep first cut_freq components
        x_freq_low = x_freq[:, :self.cut_freq, :]  # [B, cut_freq, C]

        # Complex linear: frequency upsampling
        if self.individual:
            B = x_freq_low.shape[0]
            freq_out_size = int(self.cut_freq * self.length_ratio)
            out_freq = torch.zeros(
                B, freq_out_size, self.channels,
                dtype=x_freq_low.dtype, device=x_freq_low.device
            )
            for i in range(self.channels):
                out_freq[:, :, i] = self.freq_upsampler[i](x_freq_low[:, :, i])
        else:
            out_freq = self.freq_upsampler(
                x_freq_low.permute(0, 2, 1)
            ).permute(0, 2, 1)

        # IFFT to reconstruct signal of length seq_len + pred_len
        out = torch.fft.irfft(out_freq, n=self.seq_len + self.pred_len, dim=1)
        out = out * self.length_ratio  # scale correction

        # Take last pred_len as the forecast
        out = out[:, -self.pred_len:, :]  # [B, pred_len, C]

        # Denormalize
        out = out * x_stdev
        out = out + x_mean

        return out
