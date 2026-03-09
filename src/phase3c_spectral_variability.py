"""Phase 3c Experiment: Per-channel spectral variability analysis.

Hypothesis: The optimal temporal filter varies per-channel and per-sample.
Test: Measure how much the spectral energy profile varies (a) across channels
within a sample, and (b) across samples for the same channel.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/workspace/FilterNet')
from data_provider.data_factory import data_provider
from argparse import Namespace

def get_data(dataset_name, flag='test'):
    if dataset_name == 'Weather':
        args = Namespace(
            data='custom', root_path='../dataset/weather/',
            data_path='weather.csv', features='M', target='OT',
            seq_len=96, label_len=48, pred_len=96,
            freq='h', embed='timeF', num_workers=0,
            scale=True, timeenc=1, batch_size=32,
            task_name='long_term_forecast', seasonal_patterns=None,
        )
    elif dataset_name == 'ECL':
        args = Namespace(
            data='custom', root_path='../dataset/electricity/',
            data_path='electricity.csv', features='M', target='OT',
            seq_len=96, label_len=48, pred_len=96,
            freq='h', embed='timeF', num_workers=0,
            scale=True, timeenc=1, batch_size=32,
            task_name='long_term_forecast', seasonal_patterns=None,
        )
    _, loader = data_provider(args, flag)
    return loader

def revin_normalize(x):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-5
    return (x - mean) / std

def analyze_spectral_variability(dataset_name, n_samples=2000):
    print(f"\n{'='*60}")
    print(f"SPECTRAL VARIABILITY: {dataset_name}")
    print(f"{'='*60}")

    loader = get_data(dataset_name)

    # Collect per-channel spectral profiles
    all_profiles = []  # will be [n_samples, C, K]
    count = 0

    for batch_x, batch_y, _, _ in loader:
        if count >= n_samples:
            break
        B, L, C = batch_x.shape
        x_norm = revin_normalize(batch_x)  # [B, L, C]
        x_freq = torch.fft.rfft(x_norm, dim=1)  # [B, L//2+1, C]
        energy = (x_freq.real**2 + x_freq.imag**2)  # [B, K, C]

        for i in range(min(B, n_samples - count)):
            profile = energy[i].T  # [C, K]
            # Normalize to relative energy (sum to 1 per channel)
            profile_norm = profile / (profile.sum(dim=1, keepdim=True) + 1e-10)
            all_profiles.append(profile_norm.numpy())
            count += 1
        if count >= n_samples:
            break

    profiles = np.array(all_profiles)  # [n_samples, C, K]
    n, C, K = profiles.shape
    print(f"Analyzed {n} samples, C={C}, K={K}")

    # 1. Cross-channel variability within each sample
    # For each sample, compute std of spectral profiles across channels
    # Use Jensen-Shannon divergence between channels
    within_sample_std = []
    for i in range(n):
        # Average spectral profile across channels for this sample
        mean_profile = profiles[i].mean(axis=0)  # [K]
        # Per-channel deviation from average
        deviations = profiles[i] - mean_profile  # [C, K]
        # Root mean squared deviation
        rmsd = np.sqrt((deviations**2).mean())
        within_sample_std.append(rmsd)

    ws = np.array(within_sample_std)
    print(f"\n--- Cross-channel variability (within each sample) ---")
    print(f"  Mean RMSD of spectral profiles across channels: {ws.mean():.6f}")
    print(f"  Std:  {ws.std():.6f}")
    print(f"  This measures how differently channels are spectrally distributed within a sample.")

    # 2. Temporal variability for each channel across samples
    # For each channel, compute std of spectral profiles across samples
    per_channel_temporal_var = []
    for c in range(C):
        channel_profiles = profiles[:, c, :]  # [n_samples, K]
        mean_profile = channel_profiles.mean(axis=0)  # [K]
        deviations = channel_profiles - mean_profile
        rmsd = np.sqrt((deviations**2).mean())
        per_channel_temporal_var.append(rmsd)

    tv = np.array(per_channel_temporal_var)
    print(f"\n--- Temporal variability (across samples, per channel) ---")
    print(f"  Mean RMSD across channels: {tv.mean():.6f}")
    print(f"  Std:  {tv.std():.6f}")
    print(f"  Min:  {tv.min():.6f}  Max: {tv.max():.6f}")
    print(f"  This measures how much each channel's spectral shape changes over time.")

    # 3. Compare: how much of the spectral variation is cross-channel vs temporal?
    total_var = profiles.var()
    cross_channel_var = profiles.mean(axis=0).var()  # variance of per-channel means
    temporal_var = profiles.var(axis=0).mean()  # mean of per-channel temporal variances

    print(f"\n--- Variance decomposition ---")
    print(f"  Total variance of spectral profiles: {total_var:.6f}")
    print(f"  Cross-channel variance (mean profile differs): {cross_channel_var:.6f} ({cross_channel_var/total_var*100:.1f}%)")
    print(f"  Temporal variance (profiles shift over time): {temporal_var:.6f} ({temporal_var/total_var*100:.1f}%)")

    # 4. Per-frequency-bin analysis: which bins vary most?
    print(f"\n--- Per-frequency-bin variability ---")
    print(f"{'Bin':>4} | {'Mean Energy':>11} | {'Cross-Ch Std':>12} | {'Temporal Std':>12}")
    print("-" * 50)
    top_bins = np.argsort(-profiles.mean(axis=(0,1)))[:10]  # top 10 by energy
    for k in sorted(top_bins):
        mean_e = profiles[:, :, k].mean()
        cc_std = profiles[:, :, k].mean(axis=0).std()  # std across channels of mean profile
        t_std = profiles[:, :, k].std(axis=0).mean()  # mean across channels of temporal std
        print(f"{k:>4} | {mean_e:>11.6f} | {cc_std:>12.6f} | {t_std:>12.6f}")

    # 5. Key test: correlation between spectral profile and prediction error
    # If we KNEW which channels were "hard" in a given sample, adaptive processing could help
    # Proxy: compute entropy of spectral profile (concentrated = easier, spread = harder)
    entropies = []
    for i in range(n):
        for c in range(C):
            p = profiles[i, c] + 1e-10
            ent = -(p * np.log(p)).sum()
            entropies.append(ent)
    ent_arr = np.array(entropies)
    print(f"\n--- Spectral entropy (concentration measure) ---")
    print(f"  Mean entropy: {ent_arr.mean():.4f}")
    print(f"  Std:  {ent_arr.std():.4f}")
    print(f"  Min:  {ent_arr.min():.4f}  Max: {ent_arr.max():.4f}")
    print(f"  Low entropy = concentrated (few dominant freqs), High = spread")
    print(f"  High std = channels/samples differ in spectral concentration = adaptive processing helps")

    # 6. Quantify how much the "dominant frequency" varies
    dominant_freq = profiles.argmax(axis=2)  # [n_samples, C]
    unique_per_sample = np.array([len(np.unique(dominant_freq[i])) for i in range(n)])
    print(f"\n--- Dominant frequency diversity ---")
    print(f"  Mean unique dominant freqs per sample: {unique_per_sample.mean():.1f} / {C}")
    print(f"  If this is close to C, channels are spectrally diverse within each sample.")

    dominant_per_channel = np.array([len(np.unique(dominant_freq[:, c])) for c in range(C)])
    print(f"  Mean unique dominant freqs per channel across time: {dominant_per_channel.mean():.1f}")
    print(f"  If >1, channels' dominant freq shifts over time → need adaptive processing.")

    return profiles


if __name__ == '__main__':
    for ds in ['Weather', 'ECL']:
        analyze_spectral_variability(ds, n_samples=2000)
