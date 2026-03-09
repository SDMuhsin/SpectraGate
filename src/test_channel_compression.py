"""
Phase 3 Experiment: Channel compression via PCA projection.

Tests H2: Can we reduce 21 channels to R < 21 without losing predictive performance?

Method:
- Compute PCA on standardized training data (fixed projection)
- Create a simple frequency-domain model that operates in R-dimensional channel space
- Compare performance at R={8, 11, 14, 21} (21=control, full rank)

Model: Per-channel frequency-domain linear (like FITS logic) but in PCA-reduced space,
plus fixed encode/decode projections.
"""

import os
import sys
import argparse
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(PROJECT_ROOT, '..', 'dataset', 'weather', 'weather.csv')


def compute_pca_basis(n_components=21):
    """Compute PCA basis from training data."""
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    data = df.iloc[:, 1:].values
    num_train = int(len(data) * 0.7)

    scaler = StandardScaler()
    scaler.fit(data[:num_train])
    train_scaled = scaler.transform(data[:num_train])

    # SVD
    U, S, Vt = np.linalg.svd(train_scaled, full_matrices=False)
    # Return top n_components right singular vectors
    return Vt[:n_components, :]  # [R, 21]


class PCAFreqModel(nn.Module):
    """
    Simple model: PCA channel compress → instance norm → frequency-domain linear → decompress.
    Tests whether R < 21 channel dimensions suffice for prediction.
    """
    def __init__(self, seq_len, pred_len, enc_in, pca_rank, cut_freq=24):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.pca_rank = pca_rank
        self.cut_freq = cut_freq

        # PCA projection (frozen)
        self.register_buffer('pca_encode', torch.zeros(enc_in, pca_rank))
        self.register_buffer('pca_decode', torch.zeros(pca_rank, enc_in))

        # Frequency-domain linear per latent channel
        self.length_ratio = (seq_len + pred_len) / seq_len
        freq_out = int(cut_freq * self.length_ratio)

        self.freq_layers = nn.ModuleList([
            nn.Linear(cut_freq, freq_out).to(torch.cfloat)
            for _ in range(pca_rank)
        ])

    def set_pca(self, Vt_R):
        """Set PCA projection matrices. Vt_R: [R, enc_in]"""
        V = torch.tensor(Vt_R.T, dtype=torch.float32)  # [enc_in, R]
        self.pca_encode.copy_(V)
        self.pca_decode.copy_(V.T)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, T, C = x_enc.shape

        # Project to PCA space: [B, T, C] @ [C, R] -> [B, T, R]
        x = x_enc @ self.pca_encode

        # Instance normalization (per latent channel)
        x_mean = x.mean(dim=1, keepdim=True)
        x = x - x_mean
        x_var = x.var(dim=1, keepdim=True)
        x_std = torch.sqrt(x_var + 1e-5)
        x = x / x_std

        # FFT along time
        x_freq = torch.fft.rfft(x, dim=1)  # [B, T//2+1, R]
        x_freq_low = x_freq[:, :self.cut_freq, :]  # [B, cut_freq, R]

        # Per-channel frequency linear
        freq_out_size = int(self.cut_freq * self.length_ratio)
        out_freq = torch.zeros(B, freq_out_size, self.pca_rank,
                               dtype=x_freq_low.dtype, device=x_freq_low.device)
        for i in range(self.pca_rank):
            out_freq[:, :, i] = self.freq_layers[i](x_freq_low[:, :, i])

        # IFFT
        out = torch.fft.irfft(out_freq, n=self.seq_len + self.pred_len, dim=1)
        out = out * self.length_ratio

        # Take prediction window
        out = out[:, -self.pred_len:, :]

        # Denormalize
        out = out * x_std
        out = out + x_mean

        # Project back to original channel space: [B, pred_len, R] @ [R, C] -> [B, pred_len, C]
        out = out @ self.pca_decode

        return out


def run_experiment(pca_rank, gpu=0, seed=2021, cut_freq=24):
    """Run a full train/test cycle for a PCA-compressed frequency model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Build args for data loading
    args = argparse.Namespace(
        task_name='long_term_forecast',
        is_training=1,
        model='PCAFreq',
        model_id=f'Weather_96_96',
        data='custom',
        root_path=os.path.join(PROJECT_ROOT, '..', 'dataset', 'weather'),
        data_path='weather.csv',
        features='M',
        target='OT',
        freq='h',
        embed='timeF',
        seasonal_patterns='Monthly',
        seq_len=96,
        label_len=48,
        pred_len=96,
        inverse=False,
        enc_in=21,
        dec_in=21,
        c_out=21,
        batch_size=32,
        learning_rate=0.001,
        train_epochs=100,
        patience=10,
        dropout=0,
        num_workers=0,
        itr=1,
        des='Exp',
        loss='MSE',
        lradj='type1',
        use_amp=False,
        use_gpu=torch.cuda.is_available(),
        gpu=gpu,
        use_multi_gpu=False,
        devices='0',
        checkpoints=os.path.join(PROJECT_ROOT, 'checkpoints'),
    )

    # Compute PCA basis
    print(f"\n{'='*60}")
    print(f"PCA Rank: {pca_rank} (of 21 channels), cut_freq={cut_freq}")
    print(f"{'='*60}")

    Vt_R = compute_pca_basis(pca_rank)

    # Create model
    model = PCAFreqModel(
        seq_len=96, pred_len=96, enc_in=21,
        pca_rank=pca_rank, cut_freq=cut_freq
    )
    model.set_pca(Vt_R)
    model = model.float().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}, Trainable: {trainable_params}")

    # Data
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Training
    setting = f'PCAFreq_R{pca_rank}_cf{cut_freq}'
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss_avg = np.mean(train_loss)

        # Validate
        model.eval()
        vali_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()

                outputs = model(batch_x)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]

                loss = criterion(outputs.cpu(), batch_y)
                vali_losses.append(loss.item())

        vali_loss = np.mean(vali_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss_avg:.6f}, vali_loss={vali_loss:.6f}")

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), weights_only=True))

    # Test
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]

            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae_val, mse_val, _, _, _ = metric(preds, trues)
    print(f"\nResult: PCA_rank={pca_rank}, cut_freq={cut_freq}")
    print(f"  MSE: {mse_val:.6f}, MAE: {mae_val:.6f}")
    print(f"  Params: {trainable_params}")
    return mse_val, mae_val, trainable_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pca_rank', type=int, required=True)
    parser.add_argument('--cut_freq', type=int, default=24)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    cli = parser.parse_args()

    mse, mae, params = run_experiment(
        pca_rank=cli.pca_rank,
        gpu=cli.gpu,
        seed=cli.seed,
        cut_freq=cli.cut_freq,
    )
    print(f"\nFINAL: rank={cli.pca_rank}, cut_freq={cli.cut_freq}, "
          f"params={params}, MSE={mse:.6f}, MAE={mae:.6f}")
