"""Phase 3c Experiment: Per-sample temporal coherence via SVD.

Hypothesis: The temporal patterns in a multivariate time series window
can be captured by R << C per-sample SVD prototypes, even after RevIN
normalization. If true, processing R prototypes instead of C channels
enables massive parameter compression with minimal information loss.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/workspace/FilterNet')
from data_provider.data_factory import data_provider
from argparse import Namespace

def get_data(dataset_name, seq_len=96):
    """Load dataset using the standard data provider."""
    if dataset_name == 'Weather':
        args = Namespace(
            data='custom', root_path='../dataset/weather/',
            data_path='weather.csv', features='M', target='OT',
            seq_len=seq_len, label_len=48, pred_len=96,
            freq='h', embed='timeF', num_workers=0,
            scale=True, timeenc=1, batch_size=32,
            task_name='long_term_forecast', seasonal_patterns=None,
        )
    elif dataset_name == 'ECL':
        args = Namespace(
            data='custom', root_path='../dataset/electricity/',
            data_path='electricity.csv', features='M', target='OT',
            seq_len=seq_len, label_len=48, pred_len=96,
            freq='h', embed='timeF', num_workers=0,
            scale=True, timeenc=1, batch_size=32,
            task_name='long_term_forecast', seasonal_patterns=None,
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    _, loader = data_provider(args, 'test')
    return loader

def revin_normalize(x):
    """Apply RevIN-style normalization: per-channel mean subtraction and std division."""
    # x: [B, L, C]
    mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]
    std = x.std(dim=1, keepdim=True) + 1e-5  # [B, 1, C]
    return (x - mean) / std, mean, std

def analyze_svd(dataset_name, n_samples=2000):
    """Analyze per-sample SVD statistics."""
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*60}")

    loader = get_data(dataset_name)

    all_singular_values = []
    all_variance_captured = {r: [] for r in [2, 4, 8, 16, 32, 64, 128]}
    all_recon_mse = {r: [] for r in [2, 4, 8, 16, 32, 64, 128]}

    # Also analyze in frequency domain
    all_freq_variance_captured = {r: [] for r in [2, 4, 8, 16, 32, 64, 128]}

    count = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        if count >= n_samples:
            break

        # batch_x: [B, L, C]
        B, L, C = batch_x.shape

        # RevIN normalize
        x_norm, mean, std = revin_normalize(batch_x)

        for i in range(min(B, n_samples - count)):
            # Per-sample SVD of [C, L] matrix (channels × time)
            sample = x_norm[i].T  # [C, L]
            U, S, Vh = torch.linalg.svd(sample, full_matrices=False)  # U[C,min], S[min], Vh[min,L]

            total_var = (S ** 2).sum().item()
            all_singular_values.append(S.numpy())

            for r in all_variance_captured.keys():
                if r > min(C, L):
                    continue
                var_r = (S[:r] ** 2).sum().item()
                frac = var_r / total_var if total_var > 0 else 0
                all_variance_captured[r].append(frac)

                # Reconstruction error
                recon = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]  # [C, L]
                mse = ((sample - recon) ** 2).mean().item()
                all_recon_mse[r].append(mse)

            # Frequency domain SVD
            x_freq = torch.fft.rfft(sample, dim=1)  # [C, L//2+1] complex
            x_freq_real = torch.cat([x_freq.real, x_freq.imag], dim=1)  # [C, L+2] approx
            U_f, S_f, Vh_f = torch.linalg.svd(x_freq_real, full_matrices=False)
            total_var_f = (S_f ** 2).sum().item()

            for r in all_freq_variance_captured.keys():
                if r > min(x_freq_real.shape):
                    continue
                var_r = (S_f[:r] ** 2).sum().item()
                frac = var_r / total_var_f if total_var_f > 0 else 0
                all_freq_variance_captured[r].append(frac)

            count += 1

        if count >= n_samples:
            break

    print(f"\nAnalyzed {count} samples, C={C}, L={L}")

    # Report variance captured (time domain)
    print(f"\n--- TIME-DOMAIN SVD: Variance captured by R prototypes ---")
    print(f"{'R':>5} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'P5':>8} | {'P25':>8} | {'Median':>8}")
    print("-" * 65)
    for r in sorted(all_variance_captured.keys()):
        vals = all_variance_captured[r]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        print(f"{r:>5} | {arr.mean():>8.4f} | {arr.std():>8.4f} | {arr.min():>8.4f} | {np.percentile(arr, 5):>8.4f} | {np.percentile(arr, 25):>8.4f} | {np.median(arr):>8.4f}")

    # Report reconstruction MSE (time domain)
    print(f"\n--- TIME-DOMAIN SVD: Reconstruction MSE ---")
    print(f"{'R':>5} | {'Mean MSE':>10} | {'Median MSE':>10}")
    print("-" * 40)
    for r in sorted(all_recon_mse.keys()):
        vals = all_recon_mse[r]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        print(f"{r:>5} | {arr.mean():>10.6f} | {np.median(arr):>10.6f}")

    # Report frequency-domain variance captured
    print(f"\n--- FREQ-DOMAIN SVD: Variance captured by R prototypes ---")
    print(f"{'R':>5} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Median':>8}")
    print("-" * 50)
    for r in sorted(all_freq_variance_captured.keys()):
        vals = all_freq_variance_captured[r]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        print(f"{r:>5} | {arr.mean():>8.4f} | {arr.std():>8.4f} | {arr.min():>8.4f} | {np.median(arr):>8.4f}")

    # Report singular value decay
    print(f"\n--- SINGULAR VALUE DECAY (mean across samples) ---")
    sv_matrix = np.array(all_singular_values)
    mean_sv = sv_matrix.mean(axis=0)
    mean_sv_normalized = mean_sv / mean_sv[0]  # normalize by largest
    for idx in [0, 1, 2, 4, 7, 15, 31, 63, min(95, len(mean_sv)-1)]:
        if idx < len(mean_sv_normalized):
            print(f"  SV[{idx:>3}] = {mean_sv[idx]:>8.4f} (relative: {mean_sv_normalized[idx]:>8.4f})")

    # Key metric: what R gives 95% and 99% variance?
    print(f"\n--- KEY METRICS ---")
    for target_var in [0.90, 0.95, 0.99]:
        for r in sorted(all_variance_captured.keys()):
            vals = all_variance_captured[r]
            if len(vals) == 0:
                continue
            if np.mean(vals) >= target_var:
                print(f"  {target_var*100:.0f}% variance: R = {r} (mean {np.mean(vals):.4f}, min {np.min(vals):.4f})")
                break

    return all_variance_captured, all_recon_mse


def analyze_output_consistency(dataset_name, n_samples=500):
    """Check if SVD prototypes from input window also span the output window.

    Key question: do the temporal prototypes from x[0:96] also capture
    the future pattern y[96:192]? If U is stable, reconstruction works.
    """
    print(f"\n{'='*60}")
    print(f"OUTPUT CONSISTENCY: {dataset_name}")
    print(f"{'='*60}")

    loader = get_data(dataset_name)

    all_output_var = {r: [] for r in [2, 4, 8, 16, 32, 64]}

    count = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        if count >= n_samples:
            break

        B, L, C = batch_x.shape
        _, P, _ = batch_y.shape  # P = label_len + pred_len

        # RevIN normalize using INPUT statistics
        mean = batch_x.mean(dim=1, keepdim=True)
        std = batch_x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (batch_x - mean) / std
        y_norm = (batch_y - mean) / std  # normalize output with input stats

        for i in range(min(B, n_samples - count)):
            x_sample = x_norm[i].T  # [C, L]
            y_sample = y_norm[i, -96:, :].T  # [C, 96] — last 96 steps (pred_len)

            # SVD of input
            U, S, Vh = torch.linalg.svd(x_sample, full_matrices=False)

            # Project output onto input's U subspace
            total_var_y = (y_sample ** 2).sum().item()

            for r in all_output_var.keys():
                if r > min(C, L):
                    continue
                U_r = U[:, :r]  # [C, r]
                y_proj = U_r @ (U_r.T @ y_sample)  # project y onto R-dim subspace from x
                var_captured = (y_proj ** 2).sum().item()
                frac = var_captured / total_var_y if total_var_y > 0 else 0
                all_output_var[r].append(frac)

            count += 1
        if count >= n_samples:
            break

    print(f"\nAnalyzed {count} samples")
    print(f"\n--- OUTPUT variance captured by INPUT's R-subspace ---")
    print(f"{'R':>5} | {'Mean':>8} | {'Std':>8} | {'P5':>8} | {'Median':>8}")
    print("-" * 50)
    for r in sorted(all_output_var.keys()):
        vals = all_output_var[r]
        if len(vals) == 0:
            continue
        arr = np.array(vals)
        print(f"{r:>5} | {arr.mean():>8.4f} | {arr.std():>8.4f} | {np.percentile(arr, 5):>8.4f} | {np.median(arr):>8.4f}")


if __name__ == '__main__':
    # Phase 3c: Per-sample SVD analysis on both datasets
    for ds in ['Weather', 'ECL']:
        analyze_svd(ds, n_samples=2000)
        analyze_output_consistency(ds, n_samples=500)
