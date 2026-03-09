"""
Frequency-Dependent Channel Correlation Analysis on ECL Dataset

Question: Are channels more correlated at specific frequencies (e.g., the daily cycle) than at others?

Method:
1. Load ECL training data (first 70%)
2. Extract 5000 random windows of length 96
3. rFFT along time dimension
4. For each frequency bin: compute cross-channel covariance of magnitudes, do SVD
5. Report effective rank, top SV share, and mean off-diagonal correlation
"""

import numpy as np
import pandas as pd
import os
import sys

# Use GPU 1 if torch is needed
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# 1. Load ECL data
# ─────────────────────────────────────────────────────────────────────
print("=" * 80)
print("FREQUENCY-DEPENDENT CHANNEL CORRELATION ANALYSIS — ECL DATASET")
print("=" * 80)

data_path = "/workspace/dataset/electricity/electricity.csv"
df = pd.read_csv(data_path)
print(f"\nLoaded ECL data: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
print(f"Channels: {df.shape[1] - 1}")

# Drop date column, get numeric data
data = df.iloc[:, 1:].values.astype(np.float32)
n_timestamps, n_channels = data.shape
print(f"Data shape: [{n_timestamps}, {n_channels}]")

# ─────────────────────────────────────────────────────────────────────
# 2. Training split (first 70%)
# ─────────────────────────────────────────────────────────────────────
train_end = int(n_timestamps * 0.7)
train_data = data[:train_end]
print(f"\nTraining split: first {train_end} rows ({train_end/n_timestamps*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 3. Standardize each channel (zero mean, unit variance)
# ─────────────────────────────────────────────────────────────────────
mean = train_data.mean(axis=0, keepdims=True)
std = train_data.std(axis=0, keepdims=True)
std[std < 1e-8] = 1.0  # avoid division by zero
train_data = (train_data - mean) / std
print(f"Standardized: mean range [{train_data.mean(axis=0).min():.6f}, {train_data.mean(axis=0).max():.6f}]")
print(f"              std range  [{train_data.std(axis=0).min():.4f}, {train_data.std(axis=0).max():.4f}]")

# ─────────────────────────────────────────────────────────────────────
# 4. Extract 5000 random windows of length 96
# ─────────────────────────────────────────────────────────────────────
window_len = 96
n_windows = 5000
max_start = train_end - window_len

starts = np.random.randint(0, max_start, size=n_windows)
windows = np.stack([train_data[s:s + window_len] for s in starts], axis=0)
print(f"\nExtracted {n_windows} windows of length {window_len}")
print(f"Windows shape: {windows.shape}")  # [5000, 96, 321]

# ─────────────────────────────────────────────────────────────────────
# 5. rFFT along time dimension (dim=1)
# ─────────────────────────────────────────────────────────────────────
fft_result = np.fft.rfft(windows, axis=1)  # [5000, 49, 321]
n_freqs = fft_result.shape[1]
print(f"rFFT result shape: {fft_result.shape}")
print(f"Number of frequency bins: {n_freqs} (0 to {n_freqs - 1})")

# Compute magnitudes squared (power spectrum)
magnitudes = np.abs(fft_result)  # [5000, 49, 321]
power = magnitudes ** 2  # [5000, 49, 321]

# ─────────────────────────────────────────────────────────────────────
# 5.5 Compute energy share per frequency bin
# ─────────────────────────────────────────────────────────────────────
total_energy = power.sum()
energy_per_freq = power.sum(axis=(0, 2))  # [49]
energy_share = energy_per_freq / total_energy

print(f"\nTotal energy: {total_energy:.2e}")
print(f"Energy shares (top 10 freq bins):")
top_freqs = np.argsort(energy_share)[::-1][:10]
for k in top_freqs:
    period = window_len / k if k > 0 else float('inf')
    hours_str = f"{period:.1f}h" if k > 0 else "DC"
    print(f"  freq[{k:2d}] ({hours_str:>8s}): {energy_share[k]*100:6.2f}%")

# ─────────────────────────────────────────────────────────────────────
# 6. For each frequency bin: cross-channel covariance of magnitudes → SVD
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CROSS-CHANNEL COVARIANCE ANALYSIS (MAGNITUDE) PER FREQUENCY BIN")
print("=" * 80)

# Storage for results
results = {}

for k in range(n_freqs):
    # Extract magnitude at frequency k: [5000, 321]
    mag_k = magnitudes[:, k, :]  # [5000, 321]

    # Compute cross-channel covariance matrix [321, 321]
    # Center the magnitudes across windows
    mag_centered = mag_k - mag_k.mean(axis=0, keepdims=True)
    cov_matrix = (mag_centered.T @ mag_centered) / (n_windows - 1)  # [321, 321]

    # SVD
    U, S, Vt = np.linalg.svd(cov_matrix, full_matrices=False)

    # Effective rank for 90% and 99% of variance
    cumulative = np.cumsum(S) / S.sum()
    rank_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Top singular value share
    top_sv_share = S[0] / S.sum()

    # ─────────────────────────────────────────────────────────────────
    # 9. Mean off-diagonal correlation coefficient
    # ─────────────────────────────────────────────────────────────────
    # Compute correlation matrix from covariance
    diag = np.sqrt(np.diag(cov_matrix))
    diag[diag < 1e-10] = 1.0
    corr_matrix = cov_matrix / np.outer(diag, diag)
    # Clip to [-1, 1] for numerical stability
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    # Mean of off-diagonal elements
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_offdiag_corr = corr_matrix[mask].mean()

    # Also compute mean absolute off-diagonal correlation
    mean_abs_offdiag_corr = np.abs(corr_matrix[mask]).mean()

    results[k] = {
        'rank_90': rank_90,
        'rank_99': rank_99,
        'top_sv_share': top_sv_share,
        'mean_corr': mean_offdiag_corr,
        'mean_abs_corr': mean_abs_offdiag_corr,
        'energy_share': energy_share[k],
        'top_5_sv': S[:5] / S.sum(),
    }

# ─────────────────────────────────────────────────────────────────────
# 7 & 8. Summary table for key frequencies
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE — KEY FREQUENCY BINS")
print("=" * 80)

key_freqs = [
    (0, "DC (mean level)"),
    (1, "96h (trend)"),
    (2, "48h"),
    (4, "24h (daily cycle)"),
    (8, "12h (half-day)"),
    (12, "8h (work shift)"),
    (16, "6h"),
    (20, "4.8h"),
    (24, "4h"),
    (30, "3.2h"),
    (40, "2.4h"),
    (48, "2h (Nyquist)"),
]

header = f"{'Freq':>5s} | {'Period':>18s} | {'Energy%':>8s} | {'Rank90':>6s} | {'Rank99':>6s} | {'TopSV%':>7s} | {'MeanCorr':>9s} | {'MeanAbsCorr':>11s}"
print(header)
print("-" * len(header))

for k, label in key_freqs:
    r = results[k]
    print(f"  [{k:2d}] | {label:>18s} | {r['energy_share']*100:7.2f}% | {r['rank_90']:6d} | {r['rank_99']:6d} | {r['top_sv_share']*100:6.2f}% | {r['mean_corr']:+9.4f} | {r['mean_abs_corr']:11.4f}")

# ─────────────────────────────────────────────────────────────────────
# Full frequency sweep summary
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FULL FREQUENCY SWEEP — ALL 49 BINS")
print("=" * 80)

header2 = f"{'Freq':>5s} | {'Period':>8s} | {'Energy%':>8s} | {'Rank90':>6s} | {'Rank99':>6s} | {'TopSV%':>7s} | {'MeanCorr':>9s} | {'MeanAbsCorr':>11s}"
print(header2)
print("-" * len(header2))

for k in range(n_freqs):
    r = results[k]
    period = f"{96/k:.1f}h" if k > 0 else "DC"
    print(f"  [{k:2d}] | {period:>8s} | {r['energy_share']*100:7.2f}% | {r['rank_90']:6d} | {r['rank_99']:6d} | {r['top_sv_share']*100:6.2f}% | {r['mean_corr']:+9.4f} | {r['mean_abs_corr']:11.4f}")

# ─────────────────────────────────────────────────────────────────────
# 10. Analysis: Answer the key question
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ANALYSIS: ARE CHANNELS MORE CORRELATED AT SPECIFIC FREQUENCIES?")
print("=" * 80)

# Collect metrics across all frequencies
all_mean_corr = np.array([results[k]['mean_corr'] for k in range(n_freqs)])
all_abs_corr = np.array([results[k]['mean_abs_corr'] for k in range(n_freqs)])
all_rank90 = np.array([results[k]['rank_90'] for k in range(n_freqs)])
all_rank99 = np.array([results[k]['rank_99'] for k in range(n_freqs)])
all_top_sv = np.array([results[k]['top_sv_share'] for k in range(n_freqs)])
all_energy = np.array([results[k]['energy_share'] for k in range(n_freqs)])

print(f"\n--- Cross-frequency statistics ---")
print(f"Mean off-diagonal correlation:  min={all_mean_corr.min():+.4f} (freq[{all_mean_corr.argmin()}]),  max={all_mean_corr.max():+.4f} (freq[{all_mean_corr.argmax()}])")
print(f"Mean abs off-diagonal corr:     min={all_abs_corr.min():.4f} (freq[{all_abs_corr.argmin()}]),  max={all_abs_corr.max():.4f} (freq[{all_abs_corr.argmax()}])")
print(f"Rank for 90% variance:          min={all_rank90.min()} (freq[{all_rank90.argmin()}]),  max={all_rank90.max()} (freq[{all_rank90.argmax()}])")
print(f"Rank for 99% variance:          min={all_rank99.min()} (freq[{all_rank99.argmin()}]),  max={all_rank99.max()} (freq[{all_rank99.argmax()}])")
print(f"Top SV share:                   min={all_top_sv.min()*100:.2f}% (freq[{all_top_sv.argmin()}]),  max={all_top_sv.max()*100:.2f}% (freq[{all_top_sv.argmax()}])")

print(f"\n--- Correlation vs frequency relationship ---")
# Split into low, mid, high frequency bands
low_band = list(range(0, 6))    # DC to ~16h
mid_band = list(range(6, 20))   # ~16h to ~5h
high_band = list(range(20, 49)) # ~5h to Nyquist

for band_name, band_indices in [("Low (0-5)", low_band), ("Mid (6-19)", mid_band), ("High (20-48)", high_band)]:
    band_corr = np.mean([results[k]['mean_corr'] for k in band_indices])
    band_abs_corr = np.mean([results[k]['mean_abs_corr'] for k in band_indices])
    band_rank90 = np.mean([results[k]['rank_90'] for k in band_indices])
    band_rank99 = np.mean([results[k]['rank_99'] for k in band_indices])
    band_top_sv = np.mean([results[k]['top_sv_share'] for k in band_indices])
    band_energy = sum([results[k]['energy_share'] for k in band_indices])
    print(f"\n  {band_name} frequencies:")
    print(f"    Energy share:     {band_energy*100:.1f}%")
    print(f"    Avg mean corr:    {band_corr:+.4f}")
    print(f"    Avg abs corr:     {band_abs_corr:.4f}")
    print(f"    Avg rank (90%):   {band_rank90:.1f}")
    print(f"    Avg rank (99%):   {band_rank99:.1f}")
    print(f"    Avg top SV:       {band_top_sv*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 11. Specifically: daily cycle (k=4) vs rest
# ─────────────────────────────────────────────────────────────────────
print(f"\n--- Daily cycle (freq[4], 24h) vs overall ---")
r4 = results[4]
print(f"  Daily cycle rank for 90%: {r4['rank_90']}  (overall avg: {all_rank90.mean():.1f})")
print(f"  Daily cycle rank for 99%: {r4['rank_99']}  (overall avg: {all_rank99.mean():.1f})")
print(f"  Daily cycle top SV share: {r4['top_sv_share']*100:.2f}%  (overall avg: {all_top_sv.mean()*100:.2f}%)")
print(f"  Daily cycle mean corr:    {r4['mean_corr']:+.4f}  (overall avg: {all_mean_corr.mean():+.4f})")
print(f"  Daily cycle abs corr:     {r4['mean_abs_corr']:.4f}  (overall avg: {all_abs_corr.mean():.4f})")

# ─────────────────────────────────────────────────────────────────────
# 12. Implications for CompactFreq
# ─────────────────────────────────────────────────────────────────────
print(f"\n" + "=" * 80)
print("IMPLICATIONS FOR LOW-RANK CHANNEL COMPRESSION (CompactFreq)")
print("=" * 80)

print(f"""
CompactFreq uses a fixed rank-R projection across ALL frequency bins.
This analysis reveals whether a single R can work:

If channels are highly correlated at low frequencies but independent at high:
  → A single R is a bad fit (need low R for correlated, high R for independent)
  → Frequency-adaptive rank would help

If rank is roughly constant across frequencies:
  → A single R is reasonable
  → But if that rank is close to 321, compression doesn't help

Current CompactFreq uses R=64 for ECL (20% of 321 channels).
""")

# Check what R=64 would capture at each key frequency
print("What fraction of variance does R=64 capture at each frequency?")
for k, label in key_freqs:
    r = results[k]
    sv = r['top_5_sv']  # We only stored top 5, need full SVD for this
    # Redo quick SVD for R=64 check
    mag_k = magnitudes[:, k, :]
    mag_centered = mag_k - mag_k.mean(axis=0, keepdims=True)
    cov_matrix = (mag_centered.T @ mag_centered) / (n_windows - 1)
    _, S, _ = np.linalg.svd(cov_matrix, full_matrices=False)
    cumulative = np.cumsum(S) / S.sum()
    var_at_64 = cumulative[min(63, len(cumulative)-1)]
    var_at_12 = cumulative[min(11, len(cumulative)-1)]
    print(f"  freq[{k:2d}] ({label:>18s}): R=12 captures {var_at_12*100:.1f}%, R=64 captures {var_at_64*100:.1f}%")

print("\nDone.")
