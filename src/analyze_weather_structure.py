"""
Phase 3: Weather Data Structural Analysis
==========================================
Analyzes intrinsic dimensionality of Weather data to identify compression opportunities.

Hypotheses tested:
  H1: Spectral concentration — predictive signal in K << 48 frequency bins
  H2: Channel rank deficiency — 21 channels span a low-rank subspace
  H3: Temporal basis compression — patterns captured by M << 96 basis functions

All analysis uses standardized training data (70/10/20 split, StandardScaler on train).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, '..', 'dataset', 'weather', 'weather.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'phase3_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 96
N_CHANNELS = 21


def load_weather_train():
    """Load and standardize Weather training data (70/10/20 split)."""
    df = pd.read_csv(DATA_PATH)
    data = df.iloc[:, 1:].values  # drop date column, 21 channels
    assert data.shape[1] == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {data.shape[1]}"

    num_train = int(len(data) * 0.7)
    train_data = data[:num_train]

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)

    print(f"Weather data: {data.shape[0]} total, {num_train} train samples, {N_CHANNELS} channels")
    return train_scaled, scaler


def extract_windows(data, seq_len=SEQ_LEN, stride=1, max_windows=None):
    """Extract sliding windows from time series data."""
    n_samples = len(data) - seq_len + 1
    if max_windows is not None and n_samples > max_windows:
        # Subsample uniformly
        indices = np.linspace(0, n_samples - 1, max_windows, dtype=int)
    else:
        indices = np.arange(0, n_samples, stride)

    windows = np.array([data[i:i+seq_len] for i in indices])
    return windows  # shape: [N_windows, seq_len, channels]


# =============================================================================
# HYPOTHESIS 1: Spectral Concentration
# =============================================================================
def analyze_spectral_concentration(data):
    """
    Test: Is the signal energy in Weather data concentrated in few frequency bins?

    Method: Compute rFFT of each 96-step window per channel, measure cumulative
    energy as a function of frequency count.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: SPECTRAL CONCENTRATION")
    print("="*70)

    # Extract windows (use stride=96 for non-overlapping, plus denser sample)
    windows = extract_windows(data, SEQ_LEN, stride=48, max_windows=5000)
    N, T, C = windows.shape
    print(f"Analyzing {N} windows of length {T} across {C} channels")

    # Compute rFFT per window per channel
    # rFFT of length-96 signal gives 49 frequency bins (0 to Nyquist)
    spectra = np.fft.rfft(windows, axis=1)  # [N, 49, C]
    n_freq = spectra.shape[1]
    print(f"Number of frequency bins (rFFT): {n_freq}")

    # Power spectrum: |F(k)|^2
    power = np.abs(spectra) ** 2  # [N, 49, C]

    # Average power across all windows and channels
    avg_power = power.mean(axis=(0, 2))  # [49]
    total_power = avg_power.sum()

    # Sort frequencies by power (descending) — not by index
    sorted_idx = np.argsort(-avg_power)
    sorted_power = avg_power[sorted_idx]
    cumulative = np.cumsum(sorted_power) / total_power

    print(f"\nTotal average power: {total_power:.4f}")
    print(f"\nCumulative energy by top-K frequencies (sorted by power):")
    for k in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 48]:
        if k <= n_freq:
            pct = cumulative[k-1] * 100
            print(f"  Top {k:3d} / {n_freq} freqs: {pct:6.2f}% energy")

    # Also look at per-channel variation
    print(f"\nPer-channel spectral concentration (top-K for 90%/95%/99% energy):")
    for threshold in [0.90, 0.95, 0.99]:
        k_per_channel = []
        for c in range(C):
            ch_power = power[:, :, c].mean(axis=0)  # [49]
            ch_sorted = np.sort(ch_power)[::-1]
            ch_cum = np.cumsum(ch_sorted) / ch_sorted.sum()
            k_needed = np.searchsorted(ch_cum, threshold) + 1
            k_per_channel.append(k_needed)
        k_arr = np.array(k_per_channel)
        print(f"  {threshold*100:.0f}% energy: min={k_arr.min()}, median={np.median(k_arr):.0f}, "
              f"max={k_arr.max()}, mean={k_arr.mean():.1f}")

    # Look at ORDERED (by frequency index) cumulative energy
    ordered_cum = np.cumsum(avg_power) / total_power
    print(f"\nCumulative energy by frequency INDEX (low to high):")
    for k in [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 48]:
        if k <= n_freq:
            pct = ordered_cum[k-1] * 100
            print(f"  Freqs 0..{k-1:2d} / {n_freq-1}: {pct:6.2f}% energy")

    # Power spectrum profile
    print(f"\nPower spectrum (frequency index : average power):")
    for i in range(n_freq):
        bar = '#' * int(avg_power[i] / avg_power.max() * 50)
        print(f"  freq[{i:2d}]: {avg_power[i]:10.4f} {bar}")

    return {
        'avg_power': avg_power,
        'sorted_idx': sorted_idx,
        'cumulative_by_power': cumulative,
        'cumulative_by_index': ordered_cum,
        'n_freq': n_freq,
    }


# =============================================================================
# HYPOTHESIS 2: Channel Rank Deficiency
# =============================================================================
def analyze_channel_rank(data):
    """
    Test: Do the 21 Weather channels span a low-rank subspace?

    Method: Compute SVD of the data matrix [time, channels]. Check singular
    value decay and effective rank.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: CHANNEL RANK DEFICIENCY")
    print("="*70)

    # SVD on the full training data matrix [T, 21]
    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    total_var = (S ** 2).sum()
    cum_var = np.cumsum(S ** 2) / total_var

    print(f"\nSingular values of training data [{data.shape[0]} x {data.shape[1]}]:")
    for i in range(N_CHANNELS):
        pct = (S[i] ** 2) / total_var * 100
        cum_pct = cum_var[i] * 100
        bar = '#' * int(pct / 2)
        print(f"  SV[{i:2d}]: {S[i]:10.4f} ({pct:5.2f}% var, cum: {cum_pct:6.2f}%) {bar}")

    print(f"\nEffective rank (variance thresholds):")
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        rank = np.searchsorted(cum_var, threshold) + 1
        print(f"  {threshold*100:.1f}% variance: rank = {rank} / {N_CHANNELS}")

    # Condition number
    cond = S[0] / S[-1] if S[-1] > 0 else np.inf
    print(f"\nCondition number: {cond:.2f}")
    print(f"Ratio S[0]/S[-1]: {cond:.2f}")
    print(f"Ratio S[0]/S[10]: {S[0]/S[10]:.2f}")

    # Also check rank of window-level covariance
    windows = extract_windows(data, SEQ_LEN, stride=96, max_windows=2000)
    N, T, C = windows.shape

    # Covariance of channels across time (per window, then average)
    cov_matrices = []
    for w in range(N):
        cov = np.cov(windows[w].T)  # [C, C]
        cov_matrices.append(cov)
    avg_cov = np.mean(cov_matrices, axis=0)

    eigvals = np.linalg.eigvalsh(avg_cov)[::-1]
    total_eigvar = eigvals.sum()
    cum_eigvar = np.cumsum(eigvals) / total_eigvar

    print(f"\nAverage channel covariance eigenvalues ({N} windows):")
    for i in range(N_CHANNELS):
        pct = eigvals[i] / total_eigvar * 100
        cum_pct = cum_eigvar[i] * 100
        print(f"  EV[{i:2d}]: {eigvals[i]:10.6f} ({pct:5.2f}% var, cum: {cum_pct:6.2f}%)")

    return {
        'singular_values': S,
        'cum_variance': cum_var,
        'cov_eigenvalues': eigvals,
        'cum_cov_variance': cum_eigvar,
    }


# =============================================================================
# HYPOTHESIS 3: Temporal Basis Compression
# =============================================================================
def analyze_temporal_basis(data):
    """
    Test: Can temporal patterns be captured by M << 96 basis functions?

    Method: Extract all 96-step windows, reshape to [N*C, 96], compute SVD.
    This reveals the effective dimensionality of the temporal patterns.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: TEMPORAL BASIS COMPRESSION")
    print("="*70)

    windows = extract_windows(data, SEQ_LEN, stride=48, max_windows=5000)
    N, T, C = windows.shape
    print(f"Analyzing {N} windows of length {T} across {C} channels")

    # Reshape to [N*C, T] — each row is one channel's temporal pattern in one window
    temporal_matrix = windows.transpose(0, 2, 1).reshape(-1, T)  # [N*C, 96]
    print(f"Temporal matrix shape: {temporal_matrix.shape}")

    # Subtract per-sample mean (zero-center each window individually)
    temporal_centered = temporal_matrix - temporal_matrix.mean(axis=1, keepdims=True)

    # SVD
    # For efficiency with large matrix, use randomized SVD or just compute covariance
    cov = temporal_centered.T @ temporal_centered / len(temporal_centered)  # [96, 96]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total_var = eigvals.sum()
    cum_var = np.cumsum(eigvals) / total_var

    print(f"\nTemporal PCA eigenvalue spectrum:")
    for i in range(min(30, T)):
        pct = eigvals[i] / total_var * 100
        cum_pct = cum_var[i] * 100
        bar = '#' * int(pct)
        print(f"  PC[{i:2d}]: {eigvals[i]:10.6f} ({pct:5.2f}% var, cum: {cum_pct:6.2f}%) {bar}")

    print(f"\nEffective temporal dimensionality:")
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        rank = np.searchsorted(cum_var, threshold) + 1
        print(f"  {threshold*100:.1f}% variance: {rank} / {T} components")

    # Also analyze in frequency domain — what fraction of frequency bins matter?
    # Apply FFT to each centered window
    spectra = np.fft.rfft(temporal_centered, axis=1)  # [N*C, 49]
    power = np.abs(spectra) ** 2
    avg_power = power.mean(axis=0)
    total_power = avg_power.sum()

    sorted_power = np.sort(avg_power)[::-1]
    cum_freq_power = np.cumsum(sorted_power) / total_power

    print(f"\nFrequency domain (per-window centered):")
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cum_freq_power, threshold) + 1
        print(f"  {threshold*100:.0f}% power: {k} / {spectra.shape[1]} frequency bins")

    return {
        'eigvals': eigvals,
        'cum_variance': cum_var,
        'eigvecs': eigvecs,
    }


# =============================================================================
# CROSS-CHANNEL TEMPORAL CORRELATION
# =============================================================================
def analyze_cross_channel_temporal(data):
    """
    Analyze whether channels share temporal dynamics (co-movement patterns).
    If channels are temporally correlated, a shared temporal representation
    could compress parameters.
    """
    print("\n" + "="*70)
    print("CROSS-CHANNEL TEMPORAL CORRELATION ANALYSIS")
    print("="*70)

    windows = extract_windows(data, SEQ_LEN, stride=96, max_windows=2000)
    N, T, C = windows.shape

    # For each window, compute correlation matrix of CHANGES (first differences)
    diff_corrs = []
    for w in range(N):
        diffs = np.diff(windows[w], axis=0)  # [T-1, C]
        corr = np.corrcoef(diffs.T)  # [C, C]
        diff_corrs.append(corr)
    avg_corr = np.nanmean(diff_corrs, axis=0)

    # Extract upper triangle
    upper = avg_corr[np.triu_indices(C, k=1)]

    print(f"Cross-channel correlation of first differences ({N} windows):")
    print(f"  Mean |correlation|: {np.abs(upper).mean():.4f}")
    print(f"  Median |correlation|: {np.median(np.abs(upper)):.4f}")
    print(f"  Max |correlation|: {np.abs(upper).max():.4f}")
    print(f"  % pairs with |corr| > 0.5: {(np.abs(upper) > 0.5).mean()*100:.1f}%")
    print(f"  % pairs with |corr| > 0.3: {(np.abs(upper) > 0.3).mean()*100:.1f}%")

    # Effective channel independence
    eigvals = np.linalg.eigvalsh(avg_corr)[::-1]
    cum_var = np.cumsum(eigvals) / eigvals.sum()
    print(f"\nCorrelation matrix eigenvalues (of first-diff correlation):")
    for i in range(C):
        print(f"  EV[{i:2d}]: {eigvals[i]:.4f} (cum: {cum_var[i]*100:.1f}%)")


# =============================================================================
# PREDICTABILITY ANALYSIS: How much of future is predictable from past?
# =============================================================================
def analyze_predictability(data):
    """
    Measure how much of the future signal is linearly predictable from the past.
    This establishes what fraction of the signal ANY linear model can capture.
    """
    print("\n" + "="*70)
    print("PREDICTABILITY ANALYSIS")
    print("="*70)

    for pred_len in [96, 192, 336, 720]:
        windows_x = []
        windows_y = []
        total_len = SEQ_LEN + pred_len
        n_samples = len(data) - total_len + 1
        indices = np.linspace(0, n_samples - 1, min(3000, n_samples), dtype=int)

        for i in indices:
            windows_x.append(data[i:i+SEQ_LEN])      # [96, 21]
            windows_y.append(data[i+SEQ_LEN:i+total_len])  # [pred_len, 21]

        X = np.array(windows_x)  # [N, 96, 21]
        Y = np.array(windows_y)  # [N, pred_len, 21]
        N = len(X)

        # Total variance of Y
        Y_flat = Y.reshape(N, -1)  # [N, pred_len * 21]
        total_var = np.var(Y_flat, axis=0).sum()

        # Linear prediction: use just the last value (naive)
        Y_naive = np.repeat(X[:, -1:, :], pred_len, axis=1)
        naive_mse = np.mean((Y - Y_naive) ** 2)

        # Linear prediction: channel-wise mean of input
        Y_mean = np.repeat(X.mean(axis=1, keepdims=True), pred_len, axis=1)
        mean_mse = np.mean((Y - Y_mean) ** 2)

        print(f"\npred_len={pred_len}:")
        print(f"  Target variance: {total_var/Y_flat.shape[1]:.6f}")
        print(f"  Naive (last value) MSE: {naive_mse:.6f}")
        print(f"  Mean prediction MSE: {mean_mse:.6f}")


# =============================================================================
# KEY ANALYSIS: Spectral predictability — which frequencies carry predictive signal?
# =============================================================================
def analyze_spectral_predictability(data):
    """
    For each frequency bin, measure how well the past spectrum predicts the
    future spectrum. This identifies which frequencies are predictable vs noisy.
    """
    print("\n" + "="*70)
    print("SPECTRAL PREDICTABILITY ANALYSIS")
    print("="*70)

    pred_len = 96
    total_len = SEQ_LEN + pred_len
    n_samples = len(data) - total_len + 1
    indices = np.linspace(0, n_samples - 1, min(5000, n_samples), dtype=int)

    past_spectra = []
    future_spectra = []

    for i in indices:
        past = data[i:i+SEQ_LEN]        # [96, 21]
        future = data[i+SEQ_LEN:i+total_len]  # [96, 21]

        # Per-channel rFFT
        past_fft = np.fft.rfft(past, axis=0)    # [49, 21]
        future_fft = np.fft.rfft(future, axis=0)  # [49, 21]

        past_spectra.append(past_fft)
        future_spectra.append(future_fft)

    past_spectra = np.array(past_spectra)    # [N, 49, 21]
    future_spectra = np.array(future_spectra)  # [N, 49, 21]

    n_freq = past_spectra.shape[1]

    # For each frequency bin, compute correlation between past and future amplitudes
    print(f"\nPer-frequency predictability (past→future amplitude correlation):")
    print(f"  Freq  | Past→Future amp corr | Past power | Future power")

    freq_predictability = []
    for k in range(n_freq):
        past_amp = np.abs(past_spectra[:, k, :]).flatten()
        future_amp = np.abs(future_spectra[:, k, :]).flatten()

        if past_amp.std() > 1e-10 and future_amp.std() > 1e-10:
            corr = np.corrcoef(past_amp, future_amp)[0, 1]
        else:
            corr = 0.0

        past_power = np.mean(np.abs(past_spectra[:, k, :]) ** 2)
        future_power = np.mean(np.abs(future_spectra[:, k, :]) ** 2)
        freq_predictability.append(corr)

        print(f"  [{k:2d}]  |  {corr:+.4f}              | {past_power:10.4f} | {future_power:10.4f}")

    # Also: per-frequency MSE contribution when using past spectrum as predictor
    # Reconstruct future from past spectrum
    past_as_pred = np.fft.irfft(past_spectra, n=SEQ_LEN, axis=1)  # [N, 96, 21]
    actual_future = np.array([data[i+SEQ_LEN:i+total_len] for i in indices])  # [N, 96, 21]

    # Per-frequency ablation: zero out one frequency at a time and measure MSE change
    print(f"\nFrequency ablation (zero one freq, measure MSE on reconstruction):")
    baseline_mse = np.mean((past_as_pred - actual_future) ** 2)
    print(f"  Baseline (all freqs): MSE = {baseline_mse:.6f}")

    for k in range(n_freq):
        ablated = past_spectra.copy()
        ablated[:, k, :] = 0
        ablated_pred = np.fft.irfft(ablated, n=SEQ_LEN, axis=1)
        ablated_mse = np.mean((ablated_pred - actual_future) ** 2)
        delta = ablated_mse - baseline_mse
        print(f"  Remove freq[{k:2d}]: MSE = {ablated_mse:.6f} (delta = {delta:+.6f})")

    return np.array(freq_predictability)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Loading Weather training data...")
    train_data, scaler = load_weather_train()

    # Run all analyses
    spectral_results = analyze_spectral_concentration(train_data)
    channel_results = analyze_channel_rank(train_data)
    temporal_results = analyze_temporal_basis(train_data)
    analyze_cross_channel_temporal(train_data)
    analyze_predictability(train_data)
    freq_pred = analyze_spectral_predictability(train_data)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # H1: Spectral
    cum_by_power = spectral_results['cumulative_by_power']
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cum_by_power, threshold) + 1
        print(f"H1 Spectral: {threshold*100:.0f}% energy in top {k}/{spectral_results['n_freq']} freqs "
              f"({k/spectral_results['n_freq']*100:.1f}%)")

    # H2: Channel
    cum_var = channel_results['cum_variance']
    for threshold in [0.90, 0.95, 0.99]:
        rank = np.searchsorted(cum_var, threshold) + 1
        print(f"H2 Channel: {threshold*100:.0f}% variance in {rank}/{N_CHANNELS} components "
              f"({rank/N_CHANNELS*100:.1f}%)")

    # H3: Temporal
    cum_var_t = temporal_results['cum_variance']
    for threshold in [0.90, 0.95, 0.99]:
        rank = np.searchsorted(cum_var_t, threshold) + 1
        print(f"H3 Temporal: {threshold*100:.0f}% variance in {rank}/{SEQ_LEN} components "
              f"({rank/SEQ_LEN*100:.1f}%)")

    print("\nDone. Results saved conceptually — see output above.")
