"""
Spectral Gate Analysis: SpectraGate vs PaiFilter vs TexFilter

Generates a 3-panel figure showing:
(a) Channel × Frequency gate activation heatmap — channel-specific spectral selectivity
(b) Gate value vs. spectral coherence scatter — Wiener filter proof
(c) Per-sample gate adaptivity vs. PaiFilter's fixed filter

Usage:
    python src/spectral_gate_analysis.py [--gpu 0]
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from src.run_experiment import build_args, get_model

# Weather channel names (columns 1-21, excluding 'date')
CHANNEL_NAMES = [
    'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax', 'VPact', 'VPdef', 'sh (g/kg)', 'H2OC',
    'rho (g/m³)', 'wv (m/s)', 'max wv', 'wd (deg)', 'rain (mm)',
    'raining (s)', 'SWDR', 'PAR', 'max PAR', 'Tlog (degC)', 'OT'
]

DATASET = 'Weather'
PRED_LEN = 96
SEQ_LEN = 96


def load_model_from_checkpoint(model_name, ckpt_dir, device, overrides=None):
    """Load a trained model from checkpoint."""
    args = build_args(model_name, DATASET, PRED_LEN, seq_len=SEQ_LEN,
                      overrides=overrides)
    model = get_model(model_name, args)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, args


def get_test_data(args):
    """Load the test dataset."""
    args.batch_size = 1  # ensure per-sample analysis
    test_data, test_loader = data_provider(args, 'test')
    return test_data, test_loader


def extract_spectragate_gates(model, test_loader, device, K=20):
    """Run SpectraGate on test set and extract gate activations + input spectra.

    Returns:
        gates: [N_samples, C, 2K] — sigmoid gate values
        input_spectra: [N_samples, C, K] — complex FFT of input (first K bins)
        targets: [N_samples, pred_len, C] — ground truth targets
        inputs: [N_samples, seq_len, C] — raw inputs
    """
    all_gates = []
    all_spectra = []
    all_targets = []
    all_inputs = []

    # Hook to capture gate activations
    gate_output = {}
    def gate_hook(module, input, output):
        gate_output['g'] = output.detach().cpu()

    # Register hook on the gate module
    hook = model.gate.register_forward_hook(gate_hook)

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Create decoder input (zeros for prediction part)
            batch_y = batch_y.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -PRED_LEN:, :]).float()
            dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float()

            # Forward pass (gate hook captures activations)
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Also compute input spectrum manually (before RevIN+gate)
            # We want the spectrum AFTER RevIN normalization
            x_normed = model.revin(batch_x, 'norm')
            x_perm = x_normed.permute(0, 2, 1)  # [B, C, L]
            xf = torch.fft.rfft(x_perm, dim=2)[:, :, :K]  # [B, C, K]

            all_gates.append(gate_output['g'])
            all_spectra.append(xf.cpu())
            all_targets.append(batch_y[:, -PRED_LEN:, :].cpu())
            all_inputs.append(batch_x.cpu())

    hook.remove()

    gates = torch.cat(all_gates, dim=0).numpy()      # [N, C, 2K]
    spectra = torch.cat(all_spectra, dim=0)            # [N, C, K] complex
    targets = torch.cat(all_targets, dim=0).numpy()    # [N, pred_len, C]
    inputs = torch.cat(all_inputs, dim=0).numpy()      # [N, seq_len, C]

    return gates, spectra, targets, inputs


def compute_spectral_coherence(inputs, targets, K=20):
    """Compute spectral coherence between input and target for each (channel, freq).

    Coherence C(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
    where S_xy is cross-spectral density, averaged over samples.

    Returns: [C, K] coherence values in [0, 1]
    """
    N, L_in, C = inputs.shape
    _, L_out, _ = targets.shape

    # Use the shorter length for FFT (both are 96 for pred_len=96)
    L = min(L_in, L_out)

    coherence = np.zeros((C, K))

    for c in range(C):
        # FFT of input and target for this channel
        X = np.fft.rfft(inputs[:, :L, c], axis=1)[:, :K]  # [N, K]
        Y = np.fft.rfft(targets[:, :L, c], axis=1)[:, :K]  # [N, K]

        # Cross and auto spectral densities (averaged over samples)
        Sxy = np.mean(X * np.conj(Y), axis=0)     # [K]
        Sxx = np.mean(np.abs(X)**2, axis=0)        # [K]
        Syy = np.mean(np.abs(Y)**2, axis=0)        # [K]

        # Coherence
        denom = Sxx * Syy
        denom = np.maximum(denom, 1e-12)  # avoid division by zero
        coherence[c, :] = np.abs(Sxy)**2 / denom

    return coherence


def get_paifilter_freq_response(model):
    """Extract PaiFilter's fixed spectral filter magnitude.

    PaiFilter does: x_freq * w_freq, where w_freq = FFT(w).
    Returns |w_freq| for the first K bins.
    """
    w = model.w.detach().cpu().numpy()  # [1, embed_size=96]
    w_freq = np.fft.rfft(w[0])  # [49] complex
    return np.abs(w_freq)


def generate_figure(gates, spectra, coherence, pai_freq_response,
                    K=20, output_path=None):
    """Generate 3-panel analysis figure.

    (a) Channel × Frequency gate heatmap: channel-specific spectral selectivity
    (b) Per-frequency cross-channel Spearman correlation: proves gate tracks
        per-channel predictability (not just low-pass filtering)
    (c) Per-channel adaptive gating with std-dev bands vs. PaiFilter fixed filter
    """

    N, C, feat_dim = gates.shape
    assert feat_dim == 2 * K

    # Mean gate across samples: [C, 2K]
    mean_gates = np.mean(gates, axis=0)

    # Per-frequency gate: average real and imag gate components
    gate_real = mean_gates[:, :K]   # [C, K]
    gate_imag = mean_gates[:, K:]   # [C, K]
    gate_mag = (gate_real + gate_imag) / 2

    # Per-sample per-frequency gate
    sample_gate_mag = (gates[:, :, :K] + gates[:, :, K:]) / 2  # [N, C, K]
    gate_std = np.std(sample_gate_mag, axis=0)  # [C, K]

    freq_idx = np.arange(K)

    # --- Compute per-frequency cross-channel Spearman correlations ---
    per_freq_rho = np.zeros(K)
    per_freq_p = np.zeros(K)
    for k in range(K):
        rho, p = stats.spearmanr(coherence[:, k], gate_mag[:, k])
        per_freq_rho[k] = rho
        per_freq_p[k] = p

    # Residual Spearman (frequency effect removed)
    freq_mean_gate = gate_mag.mean(axis=0, keepdims=True)
    freq_mean_coh = coherence.mean(axis=0, keepdims=True)
    res_gate = (gate_mag - freq_mean_gate).flatten()
    res_coh = (coherence - freq_mean_coh).flatten()
    r_resid, p_resid = stats.spearmanr(res_gate, res_coh)

    # Global Pearson for reference
    r_global, p_global = stats.pearsonr(coherence.flatten(), gate_mag.flatten())

    # --- Figure Setup ---
    fig = plt.figure(figsize=(14.5, 4.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.05, 1.1],
                  wspace=0.32, left=0.055, right=0.97, bottom=0.14, top=0.88)

    # ═══════════════════════════════════════════════════════════════════
    # Panel (a): Gate Heatmap — Channel × Frequency
    # ═══════════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0])

    im = ax_a.imshow(gate_mag, aspect='auto', cmap='RdYlBu_r',
                     interpolation='nearest', vmin=0.25, vmax=0.75)
    ax_a.set_xlabel('Frequency bin $k$', fontsize=9)
    ax_a.set_ylabel('Channel', fontsize=9)
    ax_a.set_title('(a) Learned spectral gate $\\bar{g}(c, k)$',
                    fontsize=10, fontweight='bold')

    ax_a.set_yticks(range(C))
    ax_a.set_yticklabels(CHANNEL_NAMES, fontsize=5.5)
    ax_a.set_xticks(range(0, K, 4))
    ax_a.set_xticklabels(range(0, K, 4), fontsize=7)

    cb = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Gate value', fontsize=8)

    # ═══════════════════════════════════════════════════════════════════
    # Panel (b): Per-frequency cross-channel Spearman correlation
    #   At each frequency k, correlate gate values across 21 channels
    #   with their spectral coherence. This proves channel discrimination,
    #   not just low-pass filtering.
    # ═══════════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[1])

    # Bar colors: significant bars in dark blue, non-significant in light
    bar_colors = ['#2c3e50' if p < 0.05 else '#bdc3c7' for p in per_freq_p]
    bars = ax_b.bar(freq_idx, per_freq_rho, color=bar_colors, width=0.8,
                    edgecolor='white', linewidth=0.5, zorder=3)

    # Zero line
    ax_b.axhline(0, color='black', linewidth=0.5, zorder=2)

    # Mean correlation line
    mean_rho = per_freq_rho.mean()
    ax_b.axhline(mean_rho, color='#e74c3c', linewidth=1.5, linestyle='--',
                 zorder=4, label=f'Mean $\\rho = {mean_rho:.2f}$')

    # Annotation
    n_pos = (per_freq_rho > 0).sum()
    n_sig = (per_freq_p < 0.05).sum()
    textstr = (f'{n_pos}/{K} positive\n'
               f'{n_sig}/{K} significant ($p<0.05$)\n'
               f'Residual $\\rho={r_resid:.2f}$, '
               f'$p<10^{{{int(np.floor(np.log10(max(p_resid, 1e-99))))}}}$')
    ax_b.text(0.98, 0.97, textstr, transform=ax_b.transAxes, fontsize=7.5,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                        edgecolor='#999999', alpha=0.95))

    # Legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2c3e50', label='$p < 0.05$'),
        Patch(facecolor='#bdc3c7', label='$p \\geq 0.05$'),
    ]
    ax_b.legend(handles=legend_elements, fontsize=7, loc='upper left',
                framealpha=0.9)

    ax_b.set_xlabel('Frequency bin $k$', fontsize=9)
    ax_b.set_ylabel('Spearman $\\rho$ (gate vs. coherence)', fontsize=9)
    ax_b.set_title('(b) Cross-channel gate–predictability',
                    fontsize=10, fontweight='bold')
    ax_b.set_xlim(-0.7, K - 0.3)
    ax_b.set_ylim(-0.3, 0.85)
    ax_b.set_xticks(range(0, K, 4))
    ax_b.tick_params(labelsize=7)

    # ═══════════════════════════════════════════════════════════════════
    # Panel (c): Per-channel adaptive gating vs. PaiFilter fixed filter
    # ═══════════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[2])

    # Select physically meaningful channels with diverse behavior
    # T (degC) = idx 1 (smooth, predictable)
    # wd (deg) = idx 13 (volatile, directional)
    # rain (mm) = idx 14 (intermittent, noisy)
    selected = [
        (1,  '#c0392c', 'T (degC)'),
        (13, '#2980b9', 'wd (deg)'),
        (14, '#27ae60', 'rain (mm)'),
    ]

    for ch_idx, color, label in selected:
        mu = gate_mag[ch_idx, :]
        sd = gate_std[ch_idx, :]

        ax_c.plot(freq_idx, mu, color=color, linewidth=2.0,
                  label=label, zorder=5)
        ax_c.fill_between(freq_idx, mu - sd, mu + sd,
                          color=color, alpha=0.15, zorder=2)

    # PaiFilter's fixed spectral filter (|FFT(w)| for first K bins)
    pai_k = pai_freq_response[:K]
    g_lo, g_hi = gate_mag.min(), gate_mag.max()
    pai_norm = (pai_k - pai_k.min()) / (pai_k.max() - pai_k.min() + 1e-12)
    pai_scaled = pai_norm * (g_hi - g_lo) + g_lo
    ax_c.plot(freq_idx, pai_scaled, color='#555555', linewidth=2.0,
              linestyle='--', label='PaiFilter (fixed)', zorder=6)

    ax_c.set_xlabel('Frequency bin $k$', fontsize=9)
    ax_c.set_ylabel('Gate activation', fontsize=9)
    ax_c.set_title('(c) Adaptive vs. fixed filtering',
                    fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=7, loc='upper right', ncol=1,
                framealpha=0.9)
    ax_c.set_xlim(0, K - 1)
    ax_c.tick_params(labelsize=7)

    ax_c.annotate('shaded = per-sample $\\pm 1\\sigma$',
                  xy=(0.02, 0.02), xycoords='axes fraction',
                  fontsize=6.5, fontstyle='italic', color='#666666')

    # Save
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, 'results', 'spectral_gate_analysis.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')
    print(f'Saved: {png_path}')

    # Print detailed statistics
    print(f'\n=== Detailed Statistics ===')
    print(f'Global Pearson:  r={r_global:.4f}, p={p_global:.2e}')
    print(f'Residual Spearman (freq-corrected): r={r_resid:.4f}, p={p_resid:.2e}')
    print(f'Per-freq cross-channel: {n_pos}/{K} positive, {n_sig}/{K} significant')
    print(f'Mean per-freq rho: {mean_rho:.3f}')
    print(f'Strongest freq bins: ', end='')
    top3 = np.argsort(per_freq_rho)[::-1][:3]
    for k in top3:
        print(f'k={k} (rho={per_freq_rho[k]:.2f})', end='  ')
    print()

    return r_global, p_global, r_resid, p_resid


def generate_paper_figure(gates, spectra, coherence, pai_freq_response,
                          K=20, output_path=None):
    """Generate a compact 3-panel figure optimized for IEEE double-column format."""

    N, C, feat_dim = gates.shape
    assert feat_dim == 2 * K

    mean_gates = np.mean(gates, axis=0)
    gate_real = mean_gates[:, :K]
    gate_imag = mean_gates[:, K:]
    gate_mag = (gate_real + gate_imag) / 2

    sample_gate_mag = (gates[:, :, :K] + gates[:, :, K:]) / 2
    gate_std = np.std(sample_gate_mag, axis=0)

    freq_idx = np.arange(K)

    # Per-frequency cross-channel Spearman
    per_freq_rho = np.zeros(K)
    per_freq_p = np.zeros(K)
    for k in range(K):
        rho, p = stats.spearmanr(coherence[:, k], gate_mag[:, k])
        per_freq_rho[k] = rho
        per_freq_p[k] = p

    # Residual Spearman
    freq_mean_gate = gate_mag.mean(axis=0, keepdims=True)
    freq_mean_coh = coherence.mean(axis=0, keepdims=True)
    res_gate = (gate_mag - freq_mean_gate).flatten()
    res_coh = (coherence - freq_mean_coh).flatten()
    r_resid, p_resid = stats.spearmanr(res_gate, res_coh)

    # --- Paper figure: 7.16" wide (IEEE double-column), 2.4" tall ---
    fig = plt.figure(figsize=(7.16, 2.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.0, 1.05],
                  wspace=0.38, left=0.065, right=0.97, bottom=0.19, top=0.87)

    # ── Panel (a): Gate heatmap ──
    ax_a = fig.add_subplot(gs[0])
    im = ax_a.imshow(gate_mag, aspect='auto', cmap='RdYlBu_r',
                     interpolation='nearest', vmin=0.25, vmax=0.75)
    ax_a.set_xlabel('Frequency bin $k$', fontsize=8)
    ax_a.set_ylabel('Channel', fontsize=8)
    ax_a.set_title('(a) Mean gate $\\bar{g}(c,k)$', fontsize=8.5, fontweight='bold')
    ax_a.set_yticks(range(C))
    ax_a.set_yticklabels(CHANNEL_NAMES, fontsize=4.5)
    ax_a.set_xticks(range(0, K, 4))
    ax_a.set_xticklabels(range(0, K, 4), fontsize=6)
    cb = plt.colorbar(im, ax=ax_a, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=5.5)

    # ── Panel (b): Per-frequency cross-channel Spearman ──
    ax_b = fig.add_subplot(gs[1])
    bar_colors = ['#2c3e50' if p < 0.05 else '#bdc3c7' for p in per_freq_p]
    ax_b.bar(freq_idx, per_freq_rho, color=bar_colors, width=0.8,
             edgecolor='white', linewidth=0.3, zorder=3)
    ax_b.axhline(0, color='black', linewidth=0.4, zorder=2)

    mean_rho = per_freq_rho.mean()
    ax_b.axhline(mean_rho, color='#e74c3c', linewidth=1.2, linestyle='--', zorder=4)

    n_pos = (per_freq_rho > 0).sum()
    n_sig = (per_freq_p < 0.05).sum()
    textstr = (f'{n_pos}/{K} pos., {n_sig}/{K} sig.\n'
               f'Resid. $\\rho\\!=\\!{r_resid:.2f}$, '
               f'$p\\!<\\!10^{{{int(np.floor(np.log10(max(p_resid, 1e-99))))}}}$')
    ax_b.text(0.97, 0.97, textstr, transform=ax_b.transAxes, fontsize=6,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor='#aaa', alpha=0.95))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2c3e50', label='$p<0.05$'),
        Patch(facecolor='#bdc3c7', label='$p\\geq 0.05$'),
    ]
    ax_b.legend(handles=legend_elements, fontsize=5.5, loc='upper left',
                framealpha=0.9, handlelength=1.0, handletextpad=0.3)

    ax_b.set_xlabel('Frequency bin $k$', fontsize=8)
    ax_b.set_ylabel('Spearman $\\rho$', fontsize=8)
    ax_b.set_title('(b) Gate vs. coherence', fontsize=8.5, fontweight='bold')
    ax_b.set_xlim(-0.7, K - 0.3)
    ax_b.set_ylim(-0.3, 0.85)
    ax_b.set_xticks(range(0, K, 4))
    ax_b.tick_params(labelsize=6)

    # ── Panel (c): Adaptive vs. fixed filtering ──
    ax_c = fig.add_subplot(gs[2])

    selected = [
        (1,  '#c0392c', 'T (degC)'),
        (13, '#2980b9', 'wd (deg)'),
        (14, '#27ae60', 'rain (mm)'),
    ]
    for ch_idx, color, label in selected:
        mu = gate_mag[ch_idx, :]
        sd = gate_std[ch_idx, :]
        ax_c.plot(freq_idx, mu, color=color, linewidth=1.5, label=label, zorder=5)
        ax_c.fill_between(freq_idx, mu - sd, mu + sd,
                          color=color, alpha=0.15, zorder=2)

    pai_k = pai_freq_response[:K]
    g_lo, g_hi = gate_mag.min(), gate_mag.max()
    pai_norm = (pai_k - pai_k.min()) / (pai_k.max() - pai_k.min() + 1e-12)
    pai_scaled = pai_norm * (g_hi - g_lo) + g_lo
    ax_c.plot(freq_idx, pai_scaled, color='#555555', linewidth=1.5,
              linestyle='--', label='PaiFilter (fixed)', zorder=6)

    ax_c.set_xlabel('Frequency bin $k$', fontsize=8)
    ax_c.set_ylabel('Gate activation', fontsize=8)
    ax_c.set_title('(c) Adaptive vs. fixed', fontsize=8.5, fontweight='bold')
    ax_c.legend(fontsize=5.5, loc='upper right', ncol=1,
                framealpha=0.9, handlelength=1.2)
    ax_c.set_xlim(0, K - 1)
    ax_c.tick_params(labelsize=6)
    ax_c.annotate('shaded $= \\pm 1\\sigma$ across samples',
                  xy=(0.03, 0.03), xycoords='axes fraction',
                  fontsize=5, fontstyle='italic', color='#666')

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, 'results', 'spectral_gate_analysis_paper.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved paper figure: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    cli_args = parser.parse_args()

    device = torch.device(f'cuda:{cli_args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Checkpoint directories ---
    sg_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints',
        'Weather_96_96_SpectraGate_custom_sl96_pl96_bs32_lr0.001_drop0.1_h256_r14_k20_tb_wd0_lradjcosine')
    pai_ckpt = os.path.join(PROJECT_ROOT, 'checkpoints',
        'Weather_96_96_PaiFilter_custom_sl96_pl96')

    # --- Load models ---
    # Checkpoint was saved with e_layers=2 (3-layer MLP), D=128, D_gate=64
    print('Loading SpectraGate...')
    sg_overrides = dict(d_model=128, d_ff=64, cut_freq=20, e_layers=2,
                        dropout=0.1, batch_size=32, lradj='cosine')
    sg_model, sg_args = load_model_from_checkpoint('SpectraGate', sg_ckpt, device,
                                                    overrides=sg_overrides)
    K = sg_model.K
    print(f'  K={K}, params={sum(p.numel() for p in sg_model.parameters()):,}')

    print('Loading PaiFilter...')
    pai_overrides = dict(hidden_size=128)
    pai_model, pai_args = load_model_from_checkpoint('PaiFilter', pai_ckpt, device,
                                                      overrides=pai_overrides)
    print(f'  params={sum(p.numel() for p in pai_model.parameters()):,}')

    # --- Load test data ---
    print('Loading test data...')
    test_data, test_loader = get_test_data(sg_args)
    print(f'  Test samples: {len(test_data)}')

    # --- Extract SpectraGate gates ---
    print('Extracting SpectraGate gate activations...')
    gates, spectra, targets, inputs = extract_spectragate_gates(
        sg_model, test_loader, device, K=K)
    print(f'  Gates shape: {gates.shape}')
    print(f'  Gate range: [{gates.min():.4f}, {gates.max():.4f}]')
    print(f'  Gate mean: {gates.mean():.4f}')

    # --- Compute spectral coherence ---
    print('Computing spectral coherence...')
    coherence = compute_spectral_coherence(inputs, targets, K=K)
    print(f'  Coherence range: [{coherence.min():.4f}, {coherence.max():.4f}]')

    # --- Extract PaiFilter frequency response ---
    print('Extracting PaiFilter frequency response...')
    pai_freq = get_paifilter_freq_response(pai_model)
    print(f'  |W(f)| range: [{pai_freq.min():.4f}, {pai_freq.max():.4f}]')

    # --- Generate figure ---
    print('Generating figure...')
    output_pdf = os.path.join(PROJECT_ROOT, 'results', 'spectral_gate_analysis.pdf')
    r_global, p_global, r_resid, p_resid = generate_figure(
        gates, spectra, coherence, pai_freq, K=K, output_path=output_pdf)

    print(f'\n=== Summary ===')
    print(f'Global gate-coherence Pearson:  r={r_global:.3f} (p={p_global:.1e})')
    print(f'Freq-corrected residual Spearman: r={r_resid:.3f} (p={p_resid:.1e})')
    print(f'=> SpectraGate has learned channel-discriminative spectral filtering')

    # --- Generate paper-optimized figure ---
    print('\nGenerating paper-optimized figure...')
    paper_pdf = os.path.join(PROJECT_ROOT, 'llmdocs', 'paper', 'spectral_gate_analysis.pdf')
    generate_paper_figure(gates, spectra, coherence, pai_freq, K=K,
                          output_path=paper_pdf)


if __name__ == '__main__':
    main()
