"""
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
(NeurIPS 2023, https://github.com/thuml/iTransformer)

Key idea: treat each variate (channel) as a token and apply attention across
variates rather than across time steps.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        d_model = configs.d_model
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        d_ff = configs.d_ff
        dropout = configs.dropout
        activation = configs.activation

        # Variate embedding: map each variate's time series to d_model
        self.embedding = nn.Linear(self.seq_len, d_model)

        # Transformer encoder (attention is computed across variates)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=e_layers
        )

        # Output projection: map d_model back to pred_len for each variate
        self.projection = nn.Linear(d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Instance normalization (Non-stationary Transformer style)
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / stdev

        # Invert: treat each variate as a token, its time series as the feature
        x = x_enc.permute(0, 2, 1)       # [B, C, seq_len]
        x = self.embedding(x)             # [B, C, d_model]

        # Self-attention across variates
        x = self.encoder(x)               # [B, C, d_model]

        # Project each variate to pred_len
        x = self.projection(x)            # [B, C, pred_len]
        x = x.permute(0, 2, 1)           # [B, pred_len, C]

        # Denormalize
        x = x * stdev + means

        return x
