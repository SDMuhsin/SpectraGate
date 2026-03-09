"""SVD analysis of ECL (electricity) dataset to determine effective channel rank."""
import numpy as np
import pandas as pd

# ─── Load data ───
df = pd.read_csv("/workspace/dataset/electricity/electricity.csv")
print(f"ECL dataset shape: {df.shape}")
print(f"Columns: date + {df.shape[1]-1} channels")
print(f"Date range: {df.iloc[0,0]} to {df.iloc[-1,0]}")

# Drop date column, get numeric data
data = df.iloc[:, 1:].values.astype(np.float64)
n_total, n_channels = data.shape
print(f"Total timestamps: {n_total}, Channels: {n_channels}")

# ─── Training split (first 70%) ───
n_train = int(n_total * 0.7)
train_data = data[:n_train]
print(f"\nTraining split: first {n_train} rows ({n_train/n_total*100:.1f}%)")
print(f"Training matrix shape: {train_data.shape}")

# ─── Standardize each channel (zero mean, unit variance) ───
means = train_data.mean(axis=0)
stds = train_data.std(axis=0)
# Check for zero-variance channels
zero_var = np.sum(stds < 1e-10)
print(f"Zero-variance channels: {zero_var}")
if zero_var > 0:
    print("  WARNING: Some channels have near-zero variance!")

# Standardize
X = (train_data - means) / (stds + 1e-10)
print(f"Standardized matrix shape: {X.shape}")

# ─── SVD ───
print("\nComputing SVD...")
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"Singular values shape: {S.shape}")

# ─── Singular values (first 30) ───
print("\n" + "="*70)
print("SINGULAR VALUES (first 50)")
print("="*70)
print(f"{'Rank':>5} {'Sigma':>12} {'Sigma/Sigma1':>14} {'Var%':>8} {'CumVar%':>9}")
print("-"*50)

total_var = np.sum(S**2)
cum_var = np.cumsum(S**2) / total_var

for i in range(min(50, len(S))):
    print(f"{i+1:5d} {S[i]:12.4f} {S[i]/S[0]:14.6f} {S[i]**2/total_var*100:8.4f} {cum_var[i]*100:9.4f}")

# ─── Full decay summary ───
print(f"\n{'...':>5}")
for i in [99, 149, 199, 249, 299, 320]:
    if i < len(S):
        print(f"{i+1:5d} {S[i]:12.4f} {S[i]/S[0]:14.6f} {S[i]**2/total_var*100:8.4f} {cum_var[i]*100:9.4f}")

# ─── Rank needed for various variance thresholds ───
print("\n" + "="*70)
print("RANK NEEDED FOR VARIANCE THRESHOLDS")
print("="*70)
thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999]
print(f"{'Threshold':>12} {'Rank':>6} {'Rank/Total':>12}")
print("-"*32)
for t in thresholds:
    rank = np.searchsorted(cum_var, t) + 1  # +1 for 1-indexed
    print(f"{t*100:11.1f}% {rank:6d} {rank/n_channels*100:11.1f}%")

# ─── Condition number ───
print(f"\n{'='*70}")
print("CONDITION NUMBER")
print("="*70)
cond = S[0] / S[-1]
print(f"Condition number (sigma_1 / sigma_n): {cond:.2f}")
print(f"Largest singular value:  {S[0]:.4f}")
print(f"Smallest singular value: {S[-1]:.4f}")
print(f"Ratio sigma_1/sigma_10: {S[0]/S[9]:.2f}")
print(f"Ratio sigma_1/sigma_50: {S[0]/S[49]:.2f}")
print(f"Ratio sigma_1/sigma_100: {S[0]/S[99]:.2f}")

# ─── Comparison with Weather ───
print(f"\n{'='*70}")
print("COMPARISON WITH WEATHER DATASET")
print("="*70)
print("Weather (21 channels):")
print("  90% variance at rank 8  (38% of channels)")
print("  95% variance at rank 9  (43% of channels)")
print("  99% variance at rank 11 (52% of channels)")
print()
print("ECL (321 channels):")
for t in [0.90, 0.95, 0.99]:
    rank = np.searchsorted(cum_var, t) + 1
    print(f"  {t*100:.0f}% variance at rank {rank} ({rank/n_channels*100:.1f}% of channels)")

# ─── Effective dimensionality metrics ───
print(f"\n{'='*70}")
print("EFFECTIVE DIMENSIONALITY METRICS")
print("="*70)

# Participation ratio (PR) = (sum s_i^2)^2 / sum(s_i^4)
pr = (np.sum(S**2))**2 / np.sum(S**4)
print(f"Participation ratio: {pr:.1f} (out of {n_channels})")
print(f"  (ratio: {pr/n_channels*100:.1f}% of channels)")

# Shannon entropy of normalized eigenvalue distribution
p = S**2 / np.sum(S**2)
entropy = -np.sum(p * np.log(p + 1e-30))
max_entropy = np.log(n_channels)
print(f"Spectral entropy: {entropy:.3f} (max: {max_entropy:.3f})")
print(f"  Normalized entropy: {entropy/max_entropy:.3f} (1.0 = uniform, 0.0 = rank-1)")

# ─── What does R=64 capture? ───
print(f"\n{'='*70}")
print("COMPACTFREQ RANK CHOICES")
print("="*70)
for R in [12, 18, 24, 32, 48, 64, 96, 128, 160, 200]:
    if R <= len(S):
        print(f"  R={R:4d}: captures {cum_var[R-1]*100:.2f}% variance ({R/n_channels*100:.1f}% of channels)")

# ─── Also do Weather for direct comparison ───
print(f"\n{'='*70}")
print("WEATHER SVD (for comparison)")
print("="*70)
df_w = pd.read_csv("/workspace/dataset/weather/weather.csv")
data_w = df_w.iloc[:, 1:].values.astype(np.float64)
n_total_w = data_w.shape[0]
n_train_w = int(n_total_w * 0.7)
train_w = data_w[:n_train_w]
means_w = train_w.mean(axis=0)
stds_w = train_w.std(axis=0)
X_w = (train_w - means_w) / (stds_w + 1e-10)
U_w, S_w, Vt_w = np.linalg.svd(X_w, full_matrices=False)
total_var_w = np.sum(S_w**2)
cum_var_w = np.cumsum(S_w**2) / total_var_w

print(f"Shape: {X_w.shape}")
print(f"{'Rank':>5} {'Sigma':>12} {'CumVar%':>9}")
for i in range(min(21, len(S_w))):
    print(f"{i+1:5d} {S_w[i]:12.4f} {cum_var_w[i]*100:9.4f}")

pr_w = (np.sum(S_w**2))**2 / np.sum(S_w**4)
p_w = S_w**2 / np.sum(S_w**2)
entropy_w = -np.sum(p_w * np.log(p_w + 1e-30))
max_entropy_w = np.log(len(S_w))
print(f"\nParticipation ratio: {pr_w:.1f} (out of {len(S_w)})")
print(f"  (ratio: {pr_w/len(S_w)*100:.1f}% of channels)")
print(f"Spectral entropy: {entropy_w:.3f} (max: {max_entropy_w:.3f})")
print(f"  Normalized entropy: {entropy_w/max_entropy_w:.3f}")

cond_w = S_w[0] / S_w[-1]
print(f"Condition number: {cond_w:.2f}")

# ─── Summary ───
print(f"\n{'='*70}")
print("SUMMARY: ECL vs WEATHER")
print("="*70)
ecl_90 = np.searchsorted(cum_var, 0.90) + 1
ecl_95 = np.searchsorted(cum_var, 0.95) + 1
ecl_99 = np.searchsorted(cum_var, 0.99) + 1
w_90 = np.searchsorted(cum_var_w, 0.90) + 1
w_95 = np.searchsorted(cum_var_w, 0.95) + 1
w_99 = np.searchsorted(cum_var_w, 0.99) + 1

print(f"{'Metric':<30} {'Weather (21ch)':>16} {'ECL (321ch)':>16}")
print("-"*64)
print(f"{'Rank for 90% var':<30} {w_90:>10d} ({w_90/21*100:.0f}%) {ecl_90:>10d} ({ecl_90/321*100:.0f}%)")
print(f"{'Rank for 95% var':<30} {w_95:>10d} ({w_95/21*100:.0f}%) {ecl_95:>10d} ({ecl_95/321*100:.0f}%)")
print(f"{'Rank for 99% var':<30} {w_99:>10d} ({w_99/21*100:.0f}%) {ecl_99:>10d} ({ecl_99/321*100:.0f}%)")
print(f"{'Participation ratio':<30} {pr_w:>16.1f} {pr:>16.1f}")
print(f"{'PR / n_channels':<30} {pr_w/21*100:>15.1f}% {pr/321*100:>15.1f}%")
print(f"{'Normalized entropy':<30} {entropy_w/max_entropy_w:>16.3f} {entropy/max_entropy:>16.3f}")
print(f"{'Condition number':<30} {cond_w:>16.2f} {cond:>16.2f}")

