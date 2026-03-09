import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    RLinear: Reversible Linear model for time series forecasting.
    Paper: Revisiting Long-term Time Series Forecasting (RTSF)
    Link: https://github.com/plumprc/RTSF

    A single linear layer (seq_len -> pred_len) applied channel-independently,
    optionally wrapped with RevIN (Reversible Instance Normalization).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.rev = configs.rev

        if self.rev:
            self.revin = RevIN(self.channels, affine=True, subtract_last=False)

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, C]
        x = x_enc

        if self.rev:
            x = self.revin(x, 'norm')

        # [B, seq_len, C] -> [B, C, seq_len]
        x = x.permute(0, 2, 1)

        if self.individual:
            out = torch.zeros(x.size(0), self.channels, self.pred_len,
                              dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                out[:, i, :] = self.Linear[i](x[:, i, :])
            x = out
        else:
            x = self.Linear(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        if self.rev:
            x = self.revin(x, 'denorm')

        return x
