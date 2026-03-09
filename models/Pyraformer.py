import torch
import torch.nn as nn
from layers.Pyraformer_EncDec import Encoder


class Model(nn.Module):
    """
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs, window_size=[4, 4], inner_size=5):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        if self.task_name == 'short_term_forecast':
            window_size = [2, 2]
        self.encoder = Encoder(configs, window_size, inner_size)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                (len(window_size) + 1) * self.d_model, self.pred_len * configs.enc_in)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(
            enc_out.size(0), self.pred_len, -1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
