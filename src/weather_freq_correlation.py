"""
Frequency-Dependent Channel Correlation Analysis on WEATHER Dataset

Question: Do Weather channels show frequency-dependent correlation similar to ECL?

Method:
1. Load Weather training data (first 70% of ~52,696 rows)
2. Extract 5000 random windows of length 96
3. rFFT along time dimension
4. For each frequency bin: compute cross-channel covariance of magnitudes, do SVD
5. Report effective rank, top SV share, and mean off-diagonal correlation
6. Compare to ECL findings (ECL freq[4] had rank90=23 out of 321 channels)

Weather specifics:
- 21 channels, 52,696 timestamps, 10-minute granularity
- Window length 96 = 16 hours (vs 96 hours for ECL)
- freq[1] = 16h period, freq[4] = 4h, freq[6] ≈ 2.67h, etc.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# 1. Load Weather data
# ─────────────────────────────────────────────────────────────────────
print("=" * 90)
print("FREQUENCY-DEPENDENT CHANNEL CORRELATION ANALYSIS — WEATHER DATASET")
print("=" * 90)

data_path = "/workspace/dataset/weather/weather.csv"
df = pd.read_csv(data_path)
print(f"\nLoaded Weather data: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
print(f"Channels: {df.shape[1] - 1}")
print(f"Sampling interval: 10 minutes")

# Show channel names
channel_names = list(df.columns[1:])
print(f"Channel names: {channel_names}")

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
print(f"Training duration: {train_end * 10 / 60:.0f} hours = {train_end * 10 / 60 / 24:.1f} days")

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
print(f"Windows shape: {windows.shape}")  # [5000, 96, 21]
print(f"Each window spans {window_len * 10 / 60:.1f} hours = {window_len * 10 / 60 / 24:.2f} days")

# ─────────────────────────────────────────────────────────────────────
# 5. rFFT along time dimension (dim=1)
# ─────────────────────────────────────────────────────────────────────
fft_result = np.fft.rfft(windows, axis=1)  # [5000, 49, 21]
n_freqs = fft_result.shape[1]
print(f"rFFT result shape: {fft_result.shape}")
print(f"Number of frequency bins: {n_freqs} (0 to {n_freqs - 1})")

# Frequency resolution: each bin k corresponds to k cycles per 96 samples
# Period in minutes = 96*10 / k = 960/k
# Period in hours = 16/k
print(f"\nFrequency resolution:")
print(f"  freq[1] = 1 cycle per window = period of {window_len*10/60:.1f}h")
print(f"  freq[k] = k cycles per window = period of {window_len*10/60:.1f}/k hours")

# Compute magnitudes
magnitudes = np.abs(fft_result)  # [5000, 49, 21]
power = magnitudes ** 2  # [5000, 49, 21]

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
    if k > 0:
        period_min = window_len * 10.0 / k
        if period_min >= 60:
            period_str = f"{period_min/60:.1f}h"
        else:
            period_str = f"{period_min:.0f}min"
    else:
        period_str = "DC"
    print(f"  freq[{k:2d}] ({period_str:>8s}): {energy_share[k]*100:6.2f}%")

# ─────────────────────────────────────────────────────────────────────
# 6. For each frequency bin: cross-channel covariance of magnitudes → SVD
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("CROSS-CHANNEL COVARIANCE ANALYSIS (MAGNITUDE) PER FREQUENCY BIN")
print("=" * 90)

# Storage for results
results = {}
# Also store full singular values for later R-capture analysis
full_svs = {}

for k in range(n_freqs):
    # Extract magnitude at frequency k: [5000, 21]
    mag_k = magnitudes[:, k, :]  # [5000, 21]

    # Compute cross-channel covariance matrix [21, 21]
    mag_centered = mag_k - mag_k.mean(axis=0, keepdims=True)
    cov_matrix = (mag_centered.T @ mag_centered) / (n_windows - 1)  # [21, 21]

    # SVD
    U, S, Vt = np.linalg.svd(cov_matrix, full_matrices=False)
    full_svs[k] = S

    # Effective rank for 90% and 99% of variance
    cumulative = np.cumsum(S) / S.sum()
    rank_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1

    # Top singular value share
    top_sv_share = S[0] / S.sum()

    # Top-3 SV cumulative share
    top3_sv_share = S[:3].sum() / S.sum()

    # Compute correlation matrix from covariance
    diag = np.sqrt(np.diag(cov_matrix))
    diag[diag < 1e-10] = 1.0
    corr_matrix = cov_matrix / np.outer(diag, diag)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    # Mean of off-diagonal elements
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_offdiag_corr = corr_matrix[mask].mean()
    mean_abs_offdiag_corr = np.abs(corr_matrix[mask]).mean()

    results[k] = {
        'rank_90': rank_90,
        'rank_99': rank_99,
        'top_sv_share': top_sv_share,
        'top3_sv_share': top3_sv_share,
        'mean_corr': mean_offdiag_corr,
        'mean_abs_corr': mean_abs_offdiag_corr,
        'energy_share': energy_share[k],
        'top_5_sv': S[:5] / S.sum(),
    }

# ─────────────────────────────────────────────────────────────────────
# 7. Summary table for key frequencies
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY TABLE — KEY FREQUENCY BINS")
print("=" * 90)

# For Weather (10-min sampling, window=96 = 16h):
# freq[k] has period = 960/k minutes = 16/k hours
key_freqs = [
    (0,  "DC (mean level)"),
    (1,  "16.0h (trend)"),
    (4,  "4.0h"),
    (8,  "2.0h"),
    (12, "1.33h (80min)"),
    (20, "48min"),
    (30, "32min"),
    (48, "20min (Nyquist)"),
]

header = f"{'Freq':>5s} | {'Period':>18s} | {'Energy%':>8s} | {'Rank90':>6s}/{n_channels} | {'Rank99':>6s}/{n_channels} | {'TopSV%':>7s} | {'Top3SV%':>8s} | {'MeanCorr':>9s} | {'MeanAbsCorr':>11s}"
print(header)
print("-" * len(header))

for k, label in key_freqs:
    r = results[k]
    print(f"  [{k:2d}] | {label:>18s} | {r['energy_share']*100:7.2f}% | {r['rank_90']:6d}/{n_channels}  | {r['rank_99']:6d}/{n_channels}  | {r['top_sv_share']*100:6.2f}% | {r['top3_sv_share']*100:7.2f}% | {r['mean_corr']:+9.4f} | {r['mean_abs_corr']:11.4f}")

# ─────────────────────────────────────────────────────────────────────
# Full frequency sweep summary
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("FULL FREQUENCY SWEEP — ALL 49 BINS")
print("=" * 90)

header2 = f"{'Freq':>5s} | {'Period':>10s} | {'Energy%':>8s} | {'Rank90':>6s} | {'Rank99':>6s} | {'TopSV%':>7s} | {'Top3SV%':>8s} | {'MeanCorr':>9s} | {'MeanAbsCorr':>11s}"
print(header2)
print("-" * len(header2))

for k in range(n_freqs):
    r = results[k]
    if k > 0:
        period_min = 960.0 / k
        if period_min >= 60:
            period = f"{period_min/60:.1f}h"
        else:
            period = f"{period_min:.0f}min"
    else:
        period = "DC"
    print(f"  [{k:2d}] | {period:>10s} | {r['energy_share']*100:7.2f}% | {r['rank_90']:6d} | {r['rank_99']:6d} | {r['top_sv_share']*100:6.2f}% | {r['top3_sv_share']*100:7.2f}% | {r['mean_corr']:+9.4f} | {r['mean_abs_corr']:11.4f}")

# ─────────────────────────────────────────────────────────────────────
# 8. Band-level summary
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("BAND-LEVEL SUMMARY")
print("=" * 90)

# Bands adapted for Weather 10-min sampling:
# Low 0-5: DC, 16h, 8h, 5.3h, 4h, 3.2h (slow weather trends)
# Mid 6-19: 2.67h down to 50.5min (mesoscale)
# High 20-48: 48min down to 20min (high-frequency noise/turbulence)
low_band = list(range(0, 6))
mid_band = list(range(6, 20))
high_band = list(range(20, 49))

for band_name, band_indices, period_range in [
    ("Low (0-5)", low_band, "DC to 3.2h"),
    ("Mid (6-19)", mid_band, "2.67h to 50min"),
    ("High (20-48)", high_band, "48min to 20min"),
]:
    band_corr = np.mean([results[k]['mean_corr'] for k in band_indices])
    band_abs_corr = np.mean([results[k]['mean_abs_corr'] for k in band_indices])
    band_rank90 = np.mean([results[k]['rank_90'] for k in band_indices])
    band_rank99 = np.mean([results[k]['rank_99'] for k in band_indices])
    band_top_sv = np.mean([results[k]['top_sv_share'] for k in band_indices])
    band_top3_sv = np.mean([results[k]['top3_sv_share'] for k in band_indices])
    band_energy = sum([results[k]['energy_share'] for k in band_indices])
    print(f"\n  {band_name} — {period_range}:")
    print(f"    Energy share:       {band_energy*100:.1f}%")
    print(f"    Avg mean corr:      {band_corr:+.4f}")
    print(f"    Avg abs corr:       {band_abs_corr:.4f}")
    print(f"    Avg rank (90%):     {band_rank90:.1f} / {n_channels}")
    print(f"    Avg rank (99%):     {band_rank99:.1f} / {n_channels}")
    print(f"    Avg top SV share:   {band_top_sv*100:.1f}%")
    print(f"    Avg top-3 SV share: {band_top3_sv*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 9. Cross-frequency statistics
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("CROSS-FREQUENCY STATISTICS")
print("=" * 90)

all_mean_corr = np.array([results[k]['mean_corr'] for k in range(n_freqs)])
all_abs_corr = np.array([results[k]['mean_abs_corr'] for k in range(n_freqs)])
all_rank90 = np.array([results[k]['rank_90'] for k in range(n_freqs)])
all_rank99 = np.array([results[k]['rank_99'] for k in range(n_freqs)])
all_top_sv = np.array([results[k]['top_sv_share'] for k in range(n_freqs)])
all_energy = np.array([results[k]['energy_share'] for k in range(n_freqs)])

print(f"\n--- Across all {n_freqs} frequency bins ---")
print(f"Mean off-diagonal correlation:  min={all_mean_corr.min():+.4f} (freq[{all_mean_corr.argmin()}]),  max={all_mean_corr.max():+.4f} (freq[{all_mean_corr.argmax()}])")
print(f"Mean abs off-diagonal corr:     min={all_abs_corr.min():.4f} (freq[{all_abs_corr.argmin()}]),  max={all_abs_corr.max():.4f} (freq[{all_abs_corr.argmax()}])")
print(f"Rank for 90% variance:          min={all_rank90.min()} (freq[{all_rank90.argmin()}]),  max={all_rank90.max()} (freq[{all_rank90.argmax()}])")
print(f"Rank for 99% variance:          min={all_rank99.min()} (freq[{all_rank99.argmin()}]),  max={all_rank99.max()} (freq[{all_rank99.argmax()}])")
print(f"Top SV share:                   min={all_top_sv.min()*100:.2f}% (freq[{all_top_sv.argmin()}]),  max={all_top_sv.max()*100:.2f}% (freq[{all_top_sv.argmax()}])")

# ─────────────────────────────────────────────────────────────────────
# 10. Comparison with ECL
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("COMPARISON WITH ECL DATASET")
print("=" * 90)

print(f"""
ECL reference values (from prior analysis):
  - 321 channels, 1-hour granularity, window=96 = 96 hours (4 days)
  - freq[4] (24h daily cycle): rank90 = 23/321 (7.2%), mean_abs_corr = high
  - Very strong channel correlation at daily cycle
  - ECL dominant frequency was k=4 (daily cycle)

Weather dataset:
  - {n_channels} channels, 10-min granularity, window=96 = 16 hours
  - NOTE: freq[4] in Weather = 4h period (NOT daily cycle!)
  - To compare "dominant frequency" behavior, look at highest-energy bins
""")

# Find the most energetic non-DC frequency
non_dc_energy = energy_share.copy()
non_dc_energy[0] = 0
dominant_k = np.argmax(non_dc_energy)
r_dom = results[dominant_k]
period_dom_min = 960.0 / dominant_k
period_dom_str = f"{period_dom_min/60:.1f}h" if period_dom_min >= 60 else f"{period_dom_min:.0f}min"

print(f"Weather's dominant non-DC frequency: freq[{dominant_k}] (period={period_dom_str})")
print(f"  Energy share:   {r_dom['energy_share']*100:.2f}%")
print(f"  Rank (90%):     {r_dom['rank_90']} / {n_channels}  ({r_dom['rank_90']/n_channels*100:.1f}%)")
print(f"  Rank (99%):     {r_dom['rank_99']} / {n_channels}  ({r_dom['rank_99']/n_channels*100:.1f}%)")
print(f"  Top SV share:   {r_dom['top_sv_share']*100:.2f}%")
print(f"  Mean abs corr:  {r_dom['mean_abs_corr']:.4f}")

print(f"\nComparison of rank90 as fraction of total channels:")
print(f"  ECL  freq[4] (24h):          rank90 = 23/321 = 7.2%")
print(f"  Weather freq[{dominant_k}] ({period_dom_str}): rank90 = {r_dom['rank_90']}/{n_channels} = {r_dom['rank_90']/n_channels*100:.1f}%")

# Ratio comparison across all key bins
print(f"\nRank90 as fraction of channels — all key Weather frequencies:")
for k, label in key_freqs:
    r = results[k]
    pct = r['rank_90'] / n_channels * 100
    print(f"  freq[{k:2d}] ({label:>18s}): rank90 = {r['rank_90']:2d}/{n_channels} = {pct:5.1f}%")

# ─────────────────────────────────────────────────────────────────────
# 11. What rank R captures at key frequencies
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("VARIANCE CAPTURED BY DIFFERENT RANK R VALUES")
print("=" * 90)

# CompactFreq for Weather uses R=18 (from MEMORY.md)
# Also check R=5, R=10, R=15 for reference
r_values = [3, 5, 8, 10, 12, 15, 18]

header3 = f"{'Freq':>5s} | {'Period':>18s}"
for rv in r_values:
    header3 += f" | {'R='+str(rv):>7s}"
header3 += f" | {'R90':>4s} | {'R99':>4s}"
print(header3)
print("-" * len(header3))

for k, label in key_freqs:
    S = full_svs[k]
    cumulative = np.cumsum(S) / S.sum()
    line = f"  [{k:2d}] | {label:>18s}"
    for rv in r_values:
        var_captured = cumulative[min(rv - 1, len(cumulative) - 1)]
        line += f" | {var_captured*100:6.1f}%"
    line += f" | {results[k]['rank_90']:4d} | {results[k]['rank_99']:4d}"
    print(line)

# Band-averaged version
print(f"\nBand-averaged variance captured:")
for band_name, band_indices in [("Low (0-5)", low_band), ("Mid (6-19)", mid_band), ("High (20-48)", high_band)]:
    line = f"  {band_name:>15s}"
    for rv in r_values:
        avg_var = np.mean([np.cumsum(full_svs[k])[min(rv-1, len(full_svs[k])-1)] / full_svs[k].sum() for k in band_indices])
        line += f" | {avg_var*100:6.1f}%"
    avg_r90 = np.mean([results[k]['rank_90'] for k in band_indices])
    avg_r99 = np.mean([results[k]['rank_99'] for k in band_indices])
    line += f" | {avg_r90:4.1f} | {avg_r99:4.1f}"
    print(line)

# ─────────────────────────────────────────────────────────────────────
# 12. Frequency-dependence assessment
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("ANSWER: DO WEATHER CHANNELS SHOW FREQUENCY-DEPENDENT CORRELATION SIMILAR TO ECL?")
print("=" * 90)

# Compute variance of rank90 across frequencies (normalized)
rank90_normalized = all_rank90 / n_channels
rank90_cv = rank90_normalized.std() / rank90_normalized.mean()  # coefficient of variation

abs_corr_cv = all_abs_corr.std() / all_abs_corr.mean()

print(f"\nFrequency-dependence metrics:")
print(f"  Rank90 range:       {all_rank90.min()} to {all_rank90.max()} (out of {n_channels})")
print(f"  Rank90 as % range:  {all_rank90.min()/n_channels*100:.1f}% to {all_rank90.max()/n_channels*100:.1f}%")
print(f"  Rank90 CV:          {rank90_cv:.3f} (higher = more frequency-dependent)")
print(f"  AbsCorr range:      {all_abs_corr.min():.4f} to {all_abs_corr.max():.4f}")
print(f"  AbsCorr CV:         {abs_corr_cv:.3f} (higher = more frequency-dependent)")

# Low vs high frequency comparison
low_rank90_avg = np.mean([results[k]['rank_90'] for k in low_band])
high_rank90_avg = np.mean([results[k]['rank_90'] for k in high_band])
low_abs_corr_avg = np.mean([results[k]['mean_abs_corr'] for k in low_band])
high_abs_corr_avg = np.mean([results[k]['mean_abs_corr'] for k in high_band])

print(f"\n  Low-freq avg rank90:  {low_rank90_avg:.1f} / {n_channels} ({low_rank90_avg/n_channels*100:.1f}%)")
print(f"  High-freq avg rank90: {high_rank90_avg:.1f} / {n_channels} ({high_rank90_avg/n_channels*100:.1f}%)")
print(f"  Low-freq avg absCorr: {low_abs_corr_avg:.4f}")
print(f"  High-freq avg absCorr:{high_abs_corr_avg:.4f}")

# Verdict
print(f"""
SUMMARY:
  Weather has {n_channels} channels vs ECL's 321.
  With only {n_channels} channels, rank90 is bounded by [{all_rank90.min()}, {all_rank90.max()}].

  Key findings:
  1. Rank90 as % of channels: Weather {all_rank90.mean()/n_channels*100:.1f}% vs ECL ~7% at dominant freq
     → If Weather rank90 is much higher %, channels are less compressible
  2. Low-freq rank ({low_rank90_avg:.1f}) vs high-freq rank ({high_rank90_avg:.1f}):
     → Ratio {low_rank90_avg/max(high_rank90_avg, 0.01):.2f}x shows frequency-dependence
  3. CompactFreq R=18 (86% of {n_channels} channels):
     → Minimal compression if rank90 is already near 18
""")

print("\nDone.")
