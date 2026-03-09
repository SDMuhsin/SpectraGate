"""
Spectral Energy Analysis of ECL (Electricity) Dataset
- Training split only (first 70%)
- 5000 random windows of length 96
- rFFT analysis across all 321 channels
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# 1. Load ECL CSV
print("Loading ECL dataset...")
df = pd.read_csv("/workspace/dataset/electricity/electricity.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: date + {df.shape[1] - 1} channels")
print(f"  Timestamps: {df.shape[0]}")

# Extract numeric data (all columns except the first date column)
data = df.iloc[:, 1:].values.astype(np.float64)
n_timestamps, n_channels = data.shape
print(f"  Data shape: {data.shape} (timestamps x channels)")

# 2. Training split: first 70%
train_end = int(n_timestamps * 0.7)
train_data = data[:train_end]
print(f"\nTraining split: first {train_end} rows ({train_end/n_timestamps*100:.1f}%)")

# 3. Standardize each channel (zero mean, unit variance) on training data
means = train_data.mean(axis=0)
stds = train_data.std(axis=0)
stds[stds == 0] = 1.0  # avoid division by zero
train_std = (train_data - means) / stds
print(f"  Standardized: mean ~ {train_std.mean():.6f}, std ~ {train_std.std():.4f}")

# 4. Extract 5000 random windows of length 96 across all channels
window_len = 96
n_windows = 5000
max_start = train_end - window_len

# Random start indices and random channel indices
start_indices = np.random.randint(0, max_start + 1, size=n_windows)
channel_indices = np.random.randint(0, n_channels, size=n_windows)

windows = np.zeros((n_windows, window_len))
for i in range(n_windows):
    windows[i] = train_std[start_indices[i]:start_indices[i] + window_len, channel_indices[i]]

print(f"\nExtracted {n_windows} random windows of length {window_len}")
print(f"  Window shape: {windows.shape}")

# 5. Compute rFFT of each window
fft_coeffs = np.fft.rfft(windows, axis=1)  # shape: (5000, 49)
n_freqs = fft_coeffs.shape[1]
print(f"  rFFT coefficients: {n_freqs} frequency bins (0 to {n_freqs - 1})")

# 6. Compute mean energy (|FFT coeff|^2) at each frequency bin
energy = np.abs(fft_coeffs) ** 2  # shape: (5000, 49)
mean_energy = energy.mean(axis=0)  # shape: (49,)
total_energy = mean_energy.sum()

print(f"\n{'='*65}")
print(f"SPECTRAL ENERGY ANALYSIS — ECL Dataset (Training Split)")
print(f"{'='*65}")
print(f"  Windows: {n_windows}, Length: {window_len}, Channels: {n_channels}")
print(f"  Frequency bins: {n_freqs} (DC + {n_freqs - 1} harmonics)")
print(f"  Total mean energy: {total_energy:.4f}")

# 7a. Energy in freq[0] (DC component)
dc_energy = mean_energy[0]
dc_pct = dc_energy / total_energy * 100
print(f"\n--- DC Component (freq[0]) ---")
print(f"  DC energy: {dc_energy:.4f} ({dc_pct:.2f}% of total)")

# 7b. Per-frequency energy breakdown
print(f"\n--- Per-Frequency Energy (all {n_freqs} bins) ---")
print(f"  {'Freq':>4s}  {'Energy':>12s}  {'% Total':>8s}  {'Cumul %':>8s}")
print(f"  {'----':>4s}  {'------':>12s}  {'-------':>8s}  {'-------':>8s}")
cumul = 0.0
for i in range(n_freqs):
    pct = mean_energy[i] / total_energy * 100
    cumul += pct
    print(f"  {i:4d}  {mean_energy[i]:12.4f}  {pct:7.2f}%  {cumul:7.2f}%")

# 7c. Cumulative energy by top-K frequencies (sorted by energy)
sorted_indices = np.argsort(mean_energy)[::-1]  # descending
sorted_energy = mean_energy[sorted_indices]
cumul_sorted = np.cumsum(sorted_energy) / total_energy * 100

print(f"\n--- Cumulative Energy in Top-K Frequencies (sorted by energy) ---")
top_ks = [2, 4, 6, 10, 16, 24, 36, 48]
print(f"  {'Top-K':>6s}  {'Cumul Energy %':>14s}  {'Freq indices (sorted by energy)'}")
print(f"  {'-----':>6s}  {'--------------':>14s}  {'-------------------------------'}")
for k in top_ks:
    if k <= n_freqs:
        pct = cumul_sorted[k - 1]
        indices_str = str(sorted_indices[:k].tolist())
        if len(indices_str) > 60:
            indices_str = str(sorted_indices[:min(k, 10)].tolist())[:-1] + ", ...]"
        print(f"  {k:6d}  {pct:13.2f}%  {indices_str}")

# 7d. Number of frequencies needed for 90%, 95%, 99%
print(f"\n--- Frequencies Needed for Energy Thresholds ---")
thresholds = [90, 95, 99]
for thr in thresholds:
    n_needed = np.searchsorted(cumul_sorted, thr) + 1
    print(f"  {thr}% of energy: {n_needed} frequencies (out of {n_freqs})")

# Additional: show top-10 frequencies by energy
print(f"\n--- Top 10 Frequencies by Energy ---")
print(f"  {'Rank':>4s}  {'Freq':>4s}  {'Energy':>12s}  {'% Total':>8s}")
print(f"  {'----':>4s}  {'----':>4s}  {'------':>12s}  {'-------':>8s}")
for rank in range(10):
    idx = sorted_indices[rank]
    pct = mean_energy[idx] / total_energy * 100
    print(f"  {rank+1:4d}  {idx:4d}  {mean_energy[idx]:12.4f}  {pct:7.2f}%")

# Bonus: energy distribution summary
print(f"\n--- Energy Distribution Summary ---")
non_dc_energy = total_energy - dc_energy
print(f"  DC (freq 0):        {dc_pct:6.2f}%")
print(f"  Low freq (1-4):     {mean_energy[1:5].sum()/total_energy*100:6.2f}%")
print(f"  Mid-low freq (5-12):{mean_energy[5:13].sum()/total_energy*100:6.2f}%")
print(f"  Mid freq (13-24):   {mean_energy[13:25].sum()/total_energy*100:6.2f}%")
print(f"  Mid-high (25-36):   {mean_energy[25:37].sum()/total_energy*100:6.2f}%")
print(f"  High freq (37-48):  {mean_energy[37:49].sum()/total_energy*100:6.2f}%")
print(f"  Non-DC total:       {non_dc_energy/total_energy*100:6.2f}%")

print(f"\n{'='*65}")
print("Analysis complete.")
