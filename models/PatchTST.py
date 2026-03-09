import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Calculate number of patches (with padding if needed)
        self.pad_len = 0
        if (self.seq_len - self.patch_len) % self.stride != 0:
            self.pad_len = self.stride - (self.seq_len - self.patch_len) % self.stride
        self.padded_len = self.seq_len + self.pad_len
        self.num_patches = (self.padded_len - self.patch_len) // self.stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(self.patch_len, configs.d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, configs.d_model) * 0.02)

        # Dropout
        self.dropout = nn.Dropout(configs.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model, nhead=configs.n_heads,
            dim_feedforward=configs.d_ff, dropout=configs.dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        self.norm = nn.LayerNorm(configs.d_model)

        # Prediction head: flatten patches -> pred_len
        self.head = nn.Linear(self.num_patches * configs.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, C]
        B, L, C = x_enc.shape

        # Instance normalization
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        # Channel-independent: reshape to [B*C, seq_len]
        x = x.permute(0, 2, 1).reshape(B * C, L)  # [B*C, L]

        # Pad at beginning if needed
        if self.pad_len > 0:
            x = F.pad(x, (self.pad_len, 0), mode='replicate')  # [B*C, padded_len]

        # Create patches: [B*C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*C, num_patches, patch_len]

        # Patch embedding
        x = self.patch_embedding(x)  # [B*C, num_patches, d_model]
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)  # [B*C, num_patches, d_model]
        x = self.norm(x)

        # Flatten and predict
        x = x.reshape(B * C, -1)  # [B*C, num_patches * d_model]
        x = self.head(x)  # [B*C, pred_len]

        # Reshape back to [B, pred_len, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]

        # Denormalize
        x = x * stdev + means

        return x
