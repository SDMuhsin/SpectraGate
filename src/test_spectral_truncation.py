"""
Phase 3 Experiment: Spectral truncation.

Tests H1: Can we use fewer frequency components without losing predictive performance?

Uses FITS-style frequency-domain linear model with varying cut_freq values.
Each experiment gets its own checkpoint directory.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class FreqLinearModel(nn.Module):
    """Per-channel frequency-domain linear model. Identical to FITS logic."""
    def __init__(self, seq_len, pred_len, channels, cut_freq):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.cut_freq = cut_freq

        self.length_ratio = (seq_len + pred_len) / seq_len
        freq_out = int(cut_freq * self.length_ratio)

        self.freq_layers = nn.ModuleList([
            nn.Linear(cut_freq, freq_out).to(torch.cfloat)
            for _ in range(channels)
        ])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Instance normalization
        x_mean = torch.mean(x_enc, dim=1, keepdim=True)
        x = x_enc - x_mean
        x_var = torch.var(x, dim=1, keepdim=True)
        x_std = torch.sqrt(x_var + 1e-5)
        x = x / x_std

        # FFT
        x_freq = torch.fft.rfft(x, dim=1)
        x_freq_low = x_freq[:, :self.cut_freq, :]

        # Per-channel frequency linear
        B = x_freq_low.shape[0]
        freq_out_size = int(self.cut_freq * self.length_ratio)
        out_freq = torch.zeros(B, freq_out_size, self.channels,
                               dtype=x_freq_low.dtype, device=x_freq_low.device)
        for i in range(self.channels):
            out_freq[:, :, i] = self.freq_layers[i](x_freq_low[:, :, i])

        # IFFT
        out = torch.fft.irfft(out_freq, n=self.seq_len + self.pred_len, dim=1)
        out = out * self.length_ratio
        out = out[:, -self.pred_len:, :]

        # Denormalize
        out = out * x_std + x_mean
        return out


def run_one(cut_freq, gpu=0, seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    args = argparse.Namespace(
        task_name='long_term_forecast', is_training=1,
        model='FreqLinear', model_id='Weather_96_96',
        data='custom',
        root_path=os.path.join(PROJECT_ROOT, '..', 'dataset', 'weather'),
        data_path='weather.csv', features='M', target='OT',
        freq='h', embed='timeF', seasonal_patterns='Monthly',
        seq_len=96, label_len=48, pred_len=96, inverse=False,
        enc_in=21, dec_in=21, c_out=21,
        batch_size=32, learning_rate=0.001,
        train_epochs=100, patience=10, dropout=0,
        num_workers=0, lradj='type1',
        use_gpu=torch.cuda.is_available(), gpu=gpu,
        use_multi_gpu=False, devices='0',
        checkpoints=os.path.join(PROJECT_ROOT, 'checkpoints'),
    )

    model = FreqLinearModel(96, 96, 21, cut_freq).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\ncut_freq={cut_freq}: params={total_params}")

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Unique checkpoint path
    setting = f'FreqLinear_cf{cut_freq}'
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        for batch_x, batch_y, bxm, bym in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)[:, -96:, :]
            loss = criterion(out, batch_y[:, -96:, :])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        vali_loss_list = []
        with torch.no_grad():
            for batch_x, batch_y, bxm, bym in vali_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()
                out = model(batch_x)[:, -96:, :]
                loss = criterion(out.cpu(), batch_y[:, -96:, :])
                vali_loss_list.append(loss.item())
        vali_loss = np.mean(vali_loss_list)

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print(f"  Early stop at epoch {epoch+1}, best vali={early_stopping.val_loss_min:.6f}")
            break
        adjust_learning_rate(optimizer, epoch + 1, args)

    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), weights_only=True))

    # Test
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, bxm, bym in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)[:, -96:, :]
            preds.append(out.cpu().numpy())
            trues.append(batch_y[:, -96:, :].cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae_v, mse_v, _, _, _ = metric(preds, trues)
    print(f"  Result: MSE={mse_v:.6f}, MAE={mae_v:.6f}, params={total_params}")
    return cut_freq, mse_v, mae_v, total_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    cli = parser.parse_args()

    results = []
    for cf in [3, 6, 10, 16, 24, 36, 48]:
        r = run_one(cf, gpu=cli.gpu)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"SPECTRAL TRUNCATION RESULTS (pred_len=96)")
    print(f"{'='*60}")
    print(f"{'cut_freq':>8} | {'params':>8} | {'MSE':>8} | {'MAE':>8}")
    print('-' * 42)
    for cf, mse, mae, p in results:
        print(f"{cf:8d} | {p:8d} | {mse:.6f} | {mae:.6f}")
