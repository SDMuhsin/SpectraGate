"""
Phase 3 analysis: Cross-channel covariance non-stationarity.

Tests whether:
1. Cross-channel covariance varies significantly across time windows (non-stationary)
2. The variation is low-dimensional (captured by few covariance modes)
3. Per-sample channel attention could capture the variation

Also tests: prediction mapping rank (is 96→pred_len mapping low-rank across channels?)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data_provider.data_factory import data_provider
import argparse


def load_data(dataset='ECL', split='train'):
    """Load dataset using existing data pipeline."""
    if dataset == 'ECL':
        args = argparse.Namespace(
            task_name='long_term_forecast', data='custom',
            root_path='/workspace/dataset/electricity/',
            data_path='electricity.csv', features='M', target='OT',
            freq='h', embed='timeF', seq_len=96, label_len=48, pred_len=96,
            seasonal_patterns='Monthly', enc_in=321, dec_in=321, c_out=321,
        )
    elif dataset == 'Weather':
        args = argparse.Namespace(
            task_name='long_term_forecast', data='custom',
            root_path='/workspace/dataset/weather/',
            data_path='weather.csv', features='M', target='OT',
            freq='h', embed='timeF', seq_len=96, label_len=48, pred_len=96,
            seasonal_patterns='Monthly', enc_in=21, dec_in=21, c_out=21,
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    args.batch_size = 1  # Load one sample at a time for covariance analysis
    args.num_workers = 0
    data, loader = data_provider(args, split)
    return data, loader, args


def analyze_covariance_nonstationarity(dataset='ECL', n_windows=2000, seed=42):
    """Analyze how cross-channel covariance varies across time windows."""
    print(f"\n{'='*60}")
    print(f"COVARIANCE NON-STATIONARITY ANALYSIS: {dataset}")
    print(f"{'='*60}")

    np.random.seed(seed)
    data, loader, args = load_data(dataset, 'train')
    C = args.enc_in

    # Collect samples
    print(f"Collecting {n_windows} windows...")
    windows = []
    for i, (batch_x, batch_y, _, _) in enumerate(loader):
        if i >= n_windows:
            break
        windows.append(batch_x.numpy()[0])  # [96, C]

    windows = np.array(windows)  # [N, 96, C]
    N = len(windows)
    print(f"Collected {N} windows, shape: {windows.shape}")

    # Compute per-window channel covariance matrices
    print("Computing per-window channel covariances...")
    cov_matrices = np.zeros((N, C, C))
    for i in range(N):
        # Standardize per window (remove mean, unit variance per channel)
        w = windows[i]  # [96, C]
        w_centered = w - w.mean(axis=0, keepdims=True)
        std = w.std(axis=0, keepdims=True) + 1e-8
        w_normed = w_centered / std
        cov_matrices[i] = w_normed.T @ w_normed / 96  # [C, C]

    # Flatten upper triangle of covariance matrices
    triu_indices = np.triu_indices(C)
    n_features = len(triu_indices[0])
    cov_flat = np.zeros((N, n_features))
    for i in range(N):
        cov_flat[i] = cov_matrices[i][triu_indices]

    print(f"Covariance feature dimension: {n_features}")

    # Center the covariance vectors
    cov_mean = cov_flat.mean(axis=0)
    cov_centered = cov_flat - cov_mean

    # SVD of covariance variations
    print("Computing SVD of covariance variations...")
    # Use truncated SVD for efficiency
    if n_features > 1000:
        from numpy.linalg import svd
        # Compute on the smaller matrix (N × N instead of n_features × n_features)
        gram = cov_centered @ cov_centered.T  # [N, N]
        eigenvalues = np.linalg.eigvalsh(gram)[::-1]  # Sort descending
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    else:
        _, s, _ = np.linalg.svd(cov_centered, full_matrices=False)
        eigenvalues = s ** 2

    # Compute explained variance
    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var

    print(f"\nCross-channel covariance variation spectrum:")
    print(f"Total variance: {total_var:.4f}")
    for r in [1, 2, 5, 10, 20, 50, 100]:
        if r <= len(cumvar):
            print(f"  Top {r:3d} modes: {cumvar[r-1]*100:.1f}% of variation")

    # Find 90%, 95%, 99% thresholds
    for thresh in [0.50, 0.80, 0.90, 0.95, 0.99]:
        r = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh*100:.0f}% variation captured by: {r} modes (out of {len(cumvar)})")

    # Analyze: how much does covariance change between consecutive windows?
    print(f"\nConsecutive window covariance change:")
    diffs = np.linalg.norm(cov_flat[1:] - cov_flat[:-1], axis=1)
    norms = np.linalg.norm(cov_flat, axis=1)
    rel_change = diffs / (norms[:-1] + 1e-8)
    print(f"  Mean relative change: {rel_change.mean():.4f}")
    print(f"  Std relative change: {rel_change.std():.4f}")
    print(f"  Max relative change: {rel_change.max():.4f}")

    # Split into first half and second half (proxy for train/test split)
    mid = N // 2
    first_half_mean = cov_flat[:mid].mean(axis=0)
    second_half_mean = cov_flat[mid:].mean(axis=0)
    shift = np.linalg.norm(second_half_mean - first_half_mean) / np.linalg.norm(cov_mean)
    print(f"\n  Covariance shift (first half vs second half): {shift:.4f} (relative norm)")


def analyze_prediction_mapping_rank(dataset='ECL', n_samples=2000, seed=42):
    """Analyze the rank of the input→output mapping across channels."""
    print(f"\n{'='*60}")
    print(f"PREDICTION MAPPING RANK ANALYSIS: {dataset}")
    print(f"{'='*60}")

    np.random.seed(seed)
    data, loader, args = load_data(dataset, 'train')
    C = args.enc_in

    # Collect input-output pairs
    print(f"Collecting {n_samples} input-output pairs...")
    X_all = []
    Y_all = []
    for i, (batch_x, batch_y, _, _) in enumerate(loader):
        if i >= n_samples:
            break
        X_all.append(batch_x.numpy()[0])  # [96, C]
        Y_all.append(batch_y.numpy()[0, -96:, :])  # [96, C] (last 96 of output)

    X = np.array(X_all)  # [N, 96, C]
    Y = np.array(Y_all)  # [N, 96, C]
    N = len(X)
    print(f"Collected {N} pairs")

    # For each channel, compute the effective rank of the X→Y mapping
    # If Y[:,c] ≈ X[:,c] @ W_c for some W_c, and the W_c are similar across channels,
    # then the mapping is low-rank across channels.

    # Compute per-channel regression: Y_c = X_c @ W_c
    print("Computing per-channel linear prediction quality...")
    per_channel_r2 = []
    W_list = []
    for c in range(C):
        X_c = X[:, :, c]  # [N, 96]
        Y_c = Y[:, :, c]  # [N, 96]
        # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
        lam = 0.01 * N
        W = np.linalg.solve(X_c.T @ X_c + lam * np.eye(96), X_c.T @ Y_c)  # [96, 96]
        Y_pred = X_c @ W  # [N, 96]
        ss_res = np.sum((Y_c - Y_pred) ** 2)
        ss_tot = np.sum((Y_c - Y_c.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        per_channel_r2.append(r2)
        W_list.append(W.flatten())

    r2 = np.array(per_channel_r2)
    print(f"Per-channel linear prediction R²:")
    print(f"  Mean: {r2.mean():.4f}, Median: {np.median(r2):.4f}")
    print(f"  Min: {r2.min():.4f}, Max: {r2.max():.4f}")
    print(f"  Channels with R²>0.5: {(r2 > 0.5).sum()}/{C}")
    print(f"  Channels with R²>0.8: {(r2 > 0.8).sum()}/{C}")

    # Stack weight matrices and compute their rank
    W_matrix = np.array(W_list)  # [C, 96*96]
    print(f"\nPer-channel weight matrix shape: {W_matrix.shape}")

    # SVD of stacked weight matrices
    W_centered = W_matrix - W_matrix.mean(axis=0)
    _, s, _ = np.linalg.svd(W_centered, full_matrices=False)
    eigenvalues = s ** 2
    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var

    print(f"\nCross-channel weight matrix variation:")
    for r in [1, 2, 5, 10, 20]:
        if r <= len(cumvar):
            print(f"  Top {r:3d} modes: {cumvar[r-1]*100:.1f}%")

    for thresh in [0.50, 0.80, 0.90, 0.95]:
        r = np.searchsorted(cumvar, thresh) + 1
        print(f"  {thresh*100:.0f}% captured by: {r} modes (out of {C})")


def analyze_spectral_prediction_rank(dataset='ECL', n_samples=2000, seed=42):
    """Analyze whether spectral features have low-rank prediction mapping."""
    print(f"\n{'='*60}")
    print(f"SPECTRAL PREDICTION MAPPING ANALYSIS: {dataset}")
    print(f"{'='*60}")

    np.random.seed(seed)
    data, loader, args = load_data(dataset, 'train')
    C = args.enc_in

    # Collect samples
    X_all, Y_all = [], []
    for i, (batch_x, batch_y, _, _) in enumerate(loader):
        if i >= n_samples:
            break
        X_all.append(batch_x.numpy()[0])  # [96, C]
        Y_all.append(batch_y.numpy()[0, -96:, :])  # [96, C]

    X = np.array(X_all)  # [N, 96, C]
    Y = np.array(Y_all)  # [N, 96, C]
    N = len(X)

    # Compute spectral features
    X_freq = np.fft.rfft(X, axis=1)  # [N, 49, C] complex
    # Truncate to K=24
    K = 24
    X_trunc = X_freq[:, :K, :]  # [N, 24, C] complex
    X_spec = np.concatenate([X_trunc.real, X_trunc.imag], axis=1)  # [N, 48, C]

    print(f"Input spectral features: [N={N}, 2K={2*K}, C={C}]")

    # For each channel, how well do 48 spectral features predict 96 output values?
    print("Per-channel spectral→temporal prediction quality:")
    r2_list = []
    for c in range(C):
        X_c = X_spec[:, :, c]  # [N, 48]
        Y_c = Y[:, :, c]  # [N, 96]
        lam = 0.01 * N
        W = np.linalg.solve(X_c.T @ X_c + lam * np.eye(2*K), X_c.T @ Y_c)  # [48, 96]
        Y_pred = X_c @ W
        ss_res = np.sum((Y_c - Y_pred) ** 2)
        ss_tot = np.sum((Y_c - Y_c.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_list.append(r2)

    r2 = np.array(r2_list)
    print(f"  Mean R²: {r2.mean():.4f}, Median: {np.median(r2):.4f}")
    print(f"  Channels with R²>0.5: {(r2 > 0.5).sum()}/{C}")
    print(f"  Channels with R²>0.8: {(r2 > 0.8).sum()}/{C}")

    # Now test: how well can ONE channel's spectral features predict ANOTHER channel's output?
    # This measures cross-channel predictive information in spectral domain
    print(f"\nCross-channel spectral prediction (random pairs):")
    np.random.seed(42)
    n_pairs = min(200, C * (C - 1))
    cross_r2 = []
    for _ in range(n_pairs):
        c1, c2 = np.random.choice(C, 2, replace=False)
        X_c = X_spec[:, :, c1]  # [N, 48] — input channel c1
        Y_c = Y[:, :, c2]  # [N, 96] — output channel c2
        lam = 0.01 * N
        W = np.linalg.solve(X_c.T @ X_c + lam * np.eye(2*K), X_c.T @ Y_c)
        Y_pred = X_c @ W
        ss_res = np.sum((Y_c - Y_pred) ** 2)
        ss_tot = np.sum((Y_c - Y_c.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        cross_r2.append(r2)

    cross_r2 = np.array(cross_r2)
    print(f"  Mean cross-channel R²: {cross_r2.mean():.4f}")
    print(f"  Pairs with R²>0.1: {(cross_r2 > 0.1).sum()}/{n_pairs}")
    print(f"  Pairs with R²>0.3: {(cross_r2 > 0.3).sum()}/{n_pairs}")


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('--dataset', default='ECL', choices=['Weather', 'ECL'])
    parser.add_argument('--n_windows', type=int, default=2000)
    args = parser.parse_args()

    analyze_covariance_nonstationarity(args.dataset, args.n_windows)
    analyze_prediction_mapping_rank(args.dataset, min(args.n_windows, 2000))
    analyze_spectral_prediction_rank(args.dataset, min(args.n_windows, 2000))
