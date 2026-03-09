"""
Benchmark script: parameter counts and inference timing for SpectraGate, TexFilter, PaiFilter.
"""
import sys
import os
import time
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.run_experiment import build_args, get_model

# ─── Dataset configurations ──────────────────────────────────────────────────
DATASETS = {
    'Weather':  dict(enc_in=21),
    'Exchange': dict(enc_in=8),
    'ECL':      dict(enc_in=321),
    'ETTh1':    dict(enc_in=7),
    'ETTh2':    dict(enc_in=7),
    'ETTm1':    dict(enc_in=7),
    'ETTm2':    dict(enc_in=7),
    'Traffic':  dict(enc_in=862),
}

PRED_LENS = [96, 192, 336, 720]
MODELS = ['TexFilter', 'PaiFilter', 'SpectraGate']
SEQ_LEN = 96


def count_params(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model):
    """Count all parameters (trainable + non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def benchmark_param_counts():
    """Compute parameter counts for all model/dataset/pred_len combos."""
    print("=" * 100)
    print("PARAMETER COUNTS")
    print("=" * 100)

    results = []

    for dataset_name, ds_info in DATASETS.items():
        enc_in = ds_info['enc_in']

        for model_name in MODELS:
            for pred_len in PRED_LENS:
                try:
                    args = build_args(model_name, dataset_name, pred_len, seq_len=SEQ_LEN)
                    model = get_model(model_name, args)
                    n_params = count_params(model)
                    n_all = count_all_params(model)

                    results.append({
                        'dataset': dataset_name,
                        'enc_in': enc_in,
                        'model': model_name,
                        'pred_len': pred_len,
                        'trainable_params': n_params,
                        'total_params': n_all,
                        'embed_size': getattr(args, 'embed_size', '-'),
                        'hidden_size': getattr(args, 'hidden_size', '-'),
                        'd_model': getattr(args, 'd_model', '-'),
                        'd_ff': getattr(args, 'd_ff', '-'),
                        'cut_freq': getattr(args, 'cut_freq', '-'),
                    })
                except Exception as e:
                    print(f"  ERROR: {dataset_name}/{model_name}/pred_len={pred_len}: {e}")
                    results.append({
                        'dataset': dataset_name,
                        'enc_in': enc_in,
                        'model': model_name,
                        'pred_len': pred_len,
                        'trainable_params': 'ERROR',
                        'total_params': 'ERROR',
                    })

    # Print table grouped by dataset
    print(f"\n{'Dataset':<10} {'enc_in':>6} {'Model':<12} {'pred_len':>8} {'Trainable':>12} {'Total':>12} {'Size(KB)':>10} {'Key Config'}")
    print("-" * 100)

    prev_dataset = None
    for r in results:
        if r['dataset'] != prev_dataset:
            if prev_dataset is not None:
                print("-" * 100)
            prev_dataset = r['dataset']

        if r['trainable_params'] == 'ERROR':
            print(f"{r['dataset']:<10} {r['enc_in']:>6} {r['model']:<12} {r['pred_len']:>8} {'ERROR':>12} {'ERROR':>12}")
            continue

        size_kb = r['trainable_params'] * 4 / 1024  # float32
        config_str = ""
        if r['model'] == 'TexFilter':
            config_str = f"E={r.get('embed_size','-')}, H={r.get('hidden_size','-')}"
        elif r['model'] == 'PaiFilter':
            config_str = f"H={r.get('hidden_size','-')}"
        elif r['model'] == 'SpectraGate':
            config_str = f"D={r.get('d_model','-')}, Dff={r.get('d_ff','-')}, K={r.get('cut_freq','-')}"

        print(f"{r['dataset']:<10} {r['enc_in']:>6} {r['model']:<12} {r['pred_len']:>8} {r['trainable_params']:>12,} {r['total_params']:>12,} {size_kb:>10.1f} {config_str}")

    return results


def benchmark_timing(device_id=0, batch_size=32, n_warmup=20, n_iters=100):
    """Benchmark forward pass timing on GPU for all 3 models across all datasets."""
    device = torch.device(f'cuda:{device_id}')

    print("\n\n" + "=" * 100)
    print(f"INFERENCE TIMING BENCHMARK (batch_size={batch_size}, seq_len={SEQ_LEN}, "
          f"n_warmup={n_warmup}, n_iters={n_iters})")
    print(f"Device: {torch.cuda.get_device_name(device_id)}")
    print("=" * 100)

    timing_results = []

    for dataset_name, ds_info in DATASETS.items():
        enc_in = ds_info['enc_in']
        pred_len = 96  # Representative

        for model_name in MODELS:
            try:
                args = build_args(model_name, dataset_name, pred_len, seq_len=SEQ_LEN)
                model = get_model(model_name, args).to(device)
                model.eval()

                n_params = count_params(model)

                # Create dummy inputs
                x = torch.randn(batch_size, SEQ_LEN, enc_in, device=device)
                x_mark = torch.randn(batch_size, SEQ_LEN, 4, device=device)  # time features
                dec_inp = torch.randn(batch_size, SEQ_LEN, enc_in, device=device)
                dec_mark = torch.randn(batch_size, SEQ_LEN, 4, device=device)

                # Warmup
                with torch.no_grad():
                    for _ in range(n_warmup):
                        _ = model(x, x_mark, dec_inp, dec_mark)

                torch.cuda.synchronize(device)

                # Timed runs
                times = []
                with torch.no_grad():
                    for _ in range(n_iters):
                        torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        _ = model(x, x_mark, dec_inp, dec_mark)
                        torch.cuda.synchronize(device)
                        t1 = time.perf_counter()
                        times.append((t1 - t0) * 1000)  # ms

                avg_ms = np.mean(times)
                std_ms = np.std(times)
                min_ms = np.min(times)
                max_ms = np.max(times)
                throughput = batch_size / (avg_ms / 1000)  # samples/sec

                timing_results.append({
                    'dataset': dataset_name,
                    'enc_in': enc_in,
                    'model': model_name,
                    'params': n_params,
                    'avg_ms': avg_ms,
                    'std_ms': std_ms,
                    'min_ms': min_ms,
                    'max_ms': max_ms,
                    'throughput': throughput,
                })

                # Free GPU memory
                del model, x, x_mark, dec_inp, dec_mark
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  TIMING ERROR: {dataset_name}/{model_name}: {e}")
                import traceback
                traceback.print_exc()

    # Print timing table
    print(f"\n{'Dataset':<10} {'enc_in':>6} {'Model':<12} {'Params':>12} {'Avg(ms)':>10} {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Throughput':>12}")
    print("-" * 100)

    prev_dataset = None
    for r in timing_results:
        if r['dataset'] != prev_dataset:
            if prev_dataset is not None:
                print("-" * 100)
            prev_dataset = r['dataset']

        print(f"{r['dataset']:<10} {r['enc_in']:>6} {r['model']:<12} {r['params']:>12,} "
              f"{r['avg_ms']:>10.3f} {r['std_ms']:>10.3f} {r['min_ms']:>10.3f} {r['max_ms']:>10.3f} "
              f"{r['throughput']:>10.0f}/s")

    return timing_results


def print_summary_table(param_results, timing_results):
    """Print a compact summary comparing the three models."""
    print("\n\n" + "=" * 100)
    print("COMPACT SUMMARY: Parameter Counts (pred_len=96 only)")
    print("=" * 100)

    # Filter to pred_len=96
    p96 = [r for r in param_results if r['pred_len'] == 96 and r['trainable_params'] != 'ERROR']

    print(f"\n{'Dataset':<10} {'enc_in':>6} | {'TexFilter':>12} {'PaiFilter':>12} {'SpectraGate':>12} | {'SG/Tex%':>8} {'SG/Pai%':>8}")
    print("-" * 85)

    for ds_name in DATASETS:
        ds_items = [r for r in p96 if r['dataset'] == ds_name]
        tex = next((r for r in ds_items if r['model'] == 'TexFilter'), None)
        pai = next((r for r in ds_items if r['model'] == 'PaiFilter'), None)
        sg = next((r for r in ds_items if r['model'] == 'SpectraGate'), None)

        if tex and pai and sg:
            enc_in = tex['enc_in']
            tex_p = tex['trainable_params']
            pai_p = pai['trainable_params']
            sg_p = sg['trainable_params']
            sg_tex_pct = 100.0 * sg_p / tex_p
            sg_pai_pct = 100.0 * sg_p / pai_p
            print(f"{ds_name:<10} {enc_in:>6} | {tex_p:>12,} {pai_p:>12,} {sg_p:>12,} | {sg_tex_pct:>7.1f}% {sg_pai_pct:>7.1f}%")

    # Check if params vary by pred_len
    print("\n\n" + "=" * 100)
    print("PARAMETER VARIATION BY pred_len (showing datasets where params change)")
    print("=" * 100)

    for ds_name in DATASETS:
        for model_name in MODELS:
            items = [r for r in param_results
                     if r['dataset'] == ds_name and r['model'] == model_name
                     and r['trainable_params'] != 'ERROR']
            if not items:
                continue
            param_vals = [r['trainable_params'] for r in items]
            if len(set(param_vals)) > 1:
                print(f"\n{ds_name} / {model_name}:")
                for r in items:
                    print(f"  pred_len={r['pred_len']}: {r['trainable_params']:>12,}")

    # Timing summary
    if timing_results:
        print("\n\n" + "=" * 100)
        print("TIMING COMPARISON (pred_len=96, batch_size=32)")
        print("=" * 100)
        print(f"\n{'Dataset':<10} {'enc_in':>6} | {'TexFilter':>12} {'PaiFilter':>12} {'SpectraGate':>12} | {'SG/Tex':>8} {'SG/Pai':>8}")
        print(f"{'':10} {'':>6} | {'avg(ms)':>12} {'avg(ms)':>12} {'avg(ms)':>12} | {'ratio':>8} {'ratio':>8}")
        print("-" * 85)

        for ds_name in DATASETS:
            ds_items = [r for r in timing_results if r['dataset'] == ds_name]
            tex = next((r for r in ds_items if r['model'] == 'TexFilter'), None)
            pai = next((r for r in ds_items if r['model'] == 'PaiFilter'), None)
            sg = next((r for r in ds_items if r['model'] == 'SpectraGate'), None)

            if tex and pai and sg:
                enc_in = tex['enc_in']
                print(f"{ds_name:<10} {enc_in:>6} | {tex['avg_ms']:>12.3f} {pai['avg_ms']:>12.3f} {sg['avg_ms']:>12.3f} | {sg['avg_ms']/tex['avg_ms']:>7.2f}x {sg['avg_ms']/pai['avg_ms']:>7.2f}x")


if __name__ == '__main__':
    param_results = benchmark_param_counts()
    timing_results = benchmark_timing(device_id=0, batch_size=32, n_warmup=20, n_iters=100)
    print_summary_table(param_results, timing_results)
