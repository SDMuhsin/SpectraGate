"""
Unified experiment runner for FilterNet multi-dataset reproduction.
Reuses existing data_provider, utils, layers, and models from the FilterNet codebase.
Supports: Weather, Exchange (and extensible to ETT, ECL, Traffic).
"""
import argparse
import fcntl
import os
import sys
import time
import csv
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# Add project root to path so we can import from existing codebase
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')

# ─── Per-dataset configuration ────────────────────────────────────────────────
DATASET_CONFIGS = {
    'Weather': dict(
        enc_in=21, dec_in=21, c_out=21,
        data='custom',
        root_path=lambda root: os.path.join(root, 'data', 'weather'),
        data_path='weather.csv',
        freq='h',
        target='OT',
        csv_name='weather_results.csv',
    ),
    'Exchange': dict(
        enc_in=8, dec_in=8, c_out=8,
        data='custom',
        root_path=lambda root: os.path.join(root, 'data', 'exchange_rate'),
        data_path='exchange_rate.csv',
        freq='d',
        target='OT',
        csv_name='exchange_results.csv',
    ),
    'ECL': dict(
        enc_in=321, dec_in=321, c_out=321,
        data='custom',
        root_path=lambda root: os.path.join(root, 'data', 'electricity'),
        data_path='electricity.csv',
        freq='h',
        target='OT',
        csv_name='ecl_results.csv',
    ),
    'ETTh1': dict(
        enc_in=7, dec_in=7, c_out=7,
        data='ETTh1',
        root_path=lambda root: os.path.join(root, 'data', 'ETT-small'),
        data_path='ETTh1.csv',
        freq='h',
        target='OT',
        csv_name='etth1_results.csv',
    ),
    'ETTh2': dict(
        enc_in=7, dec_in=7, c_out=7,
        data='ETTh2',
        root_path=lambda root: os.path.join(root, 'data', 'ETT-small'),
        data_path='ETTh2.csv',
        freq='h',
        target='OT',
        csv_name='etth2_results.csv',
    ),
    'ETTm1': dict(
        enc_in=7, dec_in=7, c_out=7,
        data='ETTm1',
        root_path=lambda root: os.path.join(root, 'data', 'ETT-small'),
        data_path='ETTm1.csv',
        freq='t',
        target='OT',
        csv_name='ettm1_results.csv',
    ),
    'ETTm2': dict(
        enc_in=7, dec_in=7, c_out=7,
        data='ETTm2',
        root_path=lambda root: os.path.join(root, 'data', 'ETT-small'),
        data_path='ETTm2.csv',
        freq='t',
        target='OT',
        csv_name='ettm2_results.csv',
    ),
    'Traffic': dict(
        enc_in=862, dec_in=862, c_out=862,
        data='custom',
        root_path=lambda root: os.path.join(root, 'data', 'traffic'),
        data_path='traffic.csv',
        freq='h',
        target='OT',
        csv_name='traffic_results.csv',
    ),
}

# ─── Default hyperparameters per model for Weather dataset ───────────────────
WEATHER_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=128,
        learning_rate=0.01, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=32,
        learning_rate=0.005, train_epochs=20, patience=6, dropout=0,
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'CompactFreq': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, rank=18,
    ),
    'SpectralMixer': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0, cut_freq=20, hidden_size=64, d_model=160, e_layers=1,
        lradj='cosine',
    ),
    'SpectralAttn': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.05, cut_freq=20, n_heads=1, d_model=164, e_layers=1, d_layers=1,
        lradj='cosine',
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=20, d_model=128, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    # Table 4 models — standard TSLib defaults
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Default hyperparameters per model for Exchange dataset ──────────────────
# Exchange: 8 features, ~7588 rows, daily, non-stationary financial data.
# Closest to ETT family (enc_in=7). Paper tuned over hidden_size {64,128,256,512},
# batch_size {4,8,16,32}, lr {0.01,0.05,0.001,0.005,0.0001,0.0005}.
EXCHANGE_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=32,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=64, batch_size=32,
        learning_rate=0.005, train_epochs=15, patience=5, dropout=0,
    ),
    'CompactFreq': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0.3, cut_freq=24, rank=8,
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=12, d_model=64, d_ff=24,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Default hyperparameters per model for ECL dataset ────────────────────────
# ECL: 321 features, ~26304 rows, hourly electricity consumption.
# TexFilter settings from official ECL.sh script.
ECL_DEFAULTS = {
    'TexFilter': dict(
        embed_size=512, hidden_size=512, batch_size=4,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=32,
        learning_rate=0.005, train_epochs=20, patience=6, dropout=0,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'CompactFreq': dict(
        batch_size=16, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0.1, cut_freq=24, rank=32,
    ),
    'SpectralMixer': dict(
        batch_size=8, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.2, cut_freq=24, hidden_size=384, d_model=512, e_layers=2,
        lradj='cosine',
    ),
    'SpectralAttn': dict(
        batch_size=8, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=24, n_heads=4, d_model=630, e_layers=1, d_layers=2,
        lradj='cosine',
    ),
    'SpectraGate': dict(
        batch_size=8, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=24, d_model=192, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'iTransformer': dict(
        batch_size=16, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Default hyperparameters per model for ETTh1 dataset ─────────────────────
# ETTh1: 7 features, ~17420 rows, hourly. Similar to Exchange (enc_in=8).
# TexFilter: no official script; using Exchange-like config.
# PaiFilter: no official script; using ETTh2-like settings.
ETTH1_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=32,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=16,
        learning_rate=0.005, train_epochs=15, patience=5, dropout=0,
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=20, d_model=128, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Default hyperparameters per model for ETTh2 dataset ─────────────────────
# ETTh2: 7 features, ~17420 rows, hourly.
# PaiFilter: from official ETTh2.sh script (pred_len-dependent settings).
# TexFilter: no official script; using Exchange-like config.
ETTH2_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=32,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=16,
        learning_rate=0.005, train_epochs=15, patience=5, dropout=0,
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=20, d_model=128, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Per-pred_len PaiFilter overrides for ETTh2 (from official script) ───────
ETTH2_PAIFILTER_PRED_LEN = {
    96: dict(hidden_size=256, batch_size=16, learning_rate=0.005),
    192: dict(hidden_size=128, batch_size=8, learning_rate=0.001),
    336: dict(hidden_size=128, batch_size=64, learning_rate=0.001),
    720: dict(hidden_size=256, batch_size=64, learning_rate=0.001),
}

# ─── Per-pred_len PaiFilter overrides for ETTh1 (tuned from ETTh2 pattern) ──
ETTH1_PAIFILTER_PRED_LEN = {
    192: dict(hidden_size=128, batch_size=8, learning_rate=0.001),
    336: dict(hidden_size=128, batch_size=64, learning_rate=0.001),
    720: dict(hidden_size=256, batch_size=64, learning_rate=0.001),
}

# ─── Default hyperparameters per model for ETTm1 dataset ─────────────────────
# ETTm1: 7 features, ~69680 rows, 15-minute.
# PaiFilter: from official ETTm1.sh script.
# TexFilter: no official script; using Exchange-like config.
ETTM1_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=32,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=32,
        learning_rate=0.01, train_epochs=15, patience=5, dropout=0,
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=20, d_model=128, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Per-pred_len TexFilter overrides for ETTm1 (best found via tuning) ──────
ETTM1_TEXFILTER_PRED_LEN = {
    192: dict(learning_rate=0.01),
    720: dict(learning_rate=0.01, hidden_size=256),
}

# ─── Default hyperparameters per model for ETTm2 dataset ─────────────────────
# ETTm2: 7 features, ~69680 rows, 15-minute.
# PaiFilter: from official ETTm2.sh script.
# TexFilter: no official script; using Exchange-like config.
ETTM2_DEFAULTS = {
    'TexFilter': dict(
        embed_size=128, hidden_size=128, batch_size=32,
        learning_rate=0.001, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=128, batch_size=32,
        learning_rate=0.005, train_epochs=15, patience=5, dropout=0,
    ),
    'SpectraGate': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=20, d_model=96, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=32, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=32,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=32, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=32, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Default hyperparameters per model for Traffic dataset ───────────────────
# Traffic: 862 features, ~17544 rows, hourly.
# TexFilter: from official Traffic.sh script. hidden_size=1024 for pred_720.
# PaiFilter: no official script; using reasonable defaults.
TRAFFIC_DEFAULTS = {
    'TexFilter': dict(
        embed_size=256, hidden_size=512, batch_size=16,
        learning_rate=0.005, train_epochs=20, patience=6, dropout=0,
    ),
    'PaiFilter': dict(
        embed_size=96, hidden_size=256, batch_size=16,
        learning_rate=0.005, train_epochs=20, patience=6, dropout=0,
    ),
    'SpectraGate': dict(
        batch_size=8, learning_rate=0.001, train_epochs=100, patience=15,
        dropout=0.1, cut_freq=24, d_model=192, d_ff=64,
        e_layers=1, d_layers=0, lradj='cosine',
    ),
    'DLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, moving_avg=25, individual=False,
    ),
    'RLinear': dict(
        batch_size=32, learning_rate=0.005, train_epochs=20, patience=6,
        dropout=0, individual=False, rev=True,
    ),
    'FITS': dict(
        batch_size=32, learning_rate=0.001, train_epochs=100, patience=10,
        dropout=0, cut_freq=24, individual=True,
    ),
    'iTransformer': dict(
        batch_size=16, learning_rate=0.0005, train_epochs=10, patience=3,
        dropout=0.1, d_model=512, n_heads=8, e_layers=3, d_ff=512,
    ),
    'PatchTST': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=20, patience=5,
        dropout=0.2, d_model=512, n_heads=16, e_layers=3, d_ff=512,
        patch_len=16, stride=8,
    ),
    'FEDformer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
    'TimesNet': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.1, d_model=32, d_ff=32, e_layers=2,
        top_k=5, num_kernels=6,
    ),
    'FreTS': dict(
        embed_size=128, hidden_size=256, batch_size=16,
        learning_rate=0.001, train_epochs=10, patience=3, dropout=0,
        channel_independence=0,
    ),
    'Autoformer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3,
    ),
    'Informer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, factor=5,
    ),
    'Pyraformer': dict(
        batch_size=16, learning_rate=0.0001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, e_layers=2,
        d_ff=2048,
    ),
    'MICN': dict(
        batch_size=16, learning_rate=0.001, train_epochs=10, patience=3,
        dropout=0.05, d_model=512, n_heads=8, d_layers=1,
        d_ff=2048, moving_avg=25,
    ),
}

# ─── Per-pred_len TexFilter overrides for ETTm2 (best found via tuning) ──────
ETTM2_TEXFILTER_PRED_LEN = {
    336: dict(learning_rate=0.01),
    720: dict(learning_rate=0.01, hidden_size=256),
}

# ─── Per-pred_len TexFilter overrides for Traffic (from official script) ─────
TRAFFIC_TEXFILTER_PRED_LEN = {
    720: dict(hidden_size=1024),
}

# ─── Dataset → defaults mapping ──────────────────────────────────────────────
DATASET_DEFAULTS = {
    'Weather': WEATHER_DEFAULTS,
    'Exchange': EXCHANGE_DEFAULTS,
    'ECL': ECL_DEFAULTS,
    'ETTh1': ETTH1_DEFAULTS,
    'ETTh2': ETTH2_DEFAULTS,
    'ETTm1': ETTM1_DEFAULTS,
    'ETTm2': ETTM2_DEFAULTS,
    'Traffic': TRAFFIC_DEFAULTS,
}


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_args(model_name, dataset, pred_len, seq_len=96, label_len=48,
               overrides=None):
    """Build an args namespace compatible with the existing codebase."""
    # Look up per-dataset per-model defaults (fall back to Weather if dataset unknown)
    ds_defaults = DATASET_DEFAULTS.get(dataset, WEATHER_DEFAULTS)
    defaults = ds_defaults.get(model_name, {}).copy()

    # Apply per-pred_len overrides for specific dataset/model combos
    if dataset == 'ETTh1' and model_name == 'PaiFilter' and pred_len in ETTH1_PAIFILTER_PRED_LEN:
        defaults.update(ETTH1_PAIFILTER_PRED_LEN[pred_len])
    if dataset == 'ETTh2' and model_name == 'PaiFilter' and pred_len in ETTH2_PAIFILTER_PRED_LEN:
        defaults.update(ETTH2_PAIFILTER_PRED_LEN[pred_len])
    if dataset == 'ETTm1' and model_name == 'TexFilter' and pred_len in ETTM1_TEXFILTER_PRED_LEN:
        defaults.update(ETTM1_TEXFILTER_PRED_LEN[pred_len])
    if dataset == 'ETTm2' and model_name == 'TexFilter' and pred_len in ETTM2_TEXFILTER_PRED_LEN:
        defaults.update(ETTM2_TEXFILTER_PRED_LEN[pred_len])
    if dataset == 'Traffic' and model_name == 'TexFilter' and pred_len in TRAFFIC_TEXFILTER_PRED_LEN:
        defaults.update(TRAFFIC_TEXFILTER_PRED_LEN[pred_len])

    # CLI overrides take highest priority
    if overrides:
        defaults.update(overrides)

    # Look up dataset configuration
    ds_cfg = DATASET_CONFIGS.get(dataset)
    if ds_cfg is None:
        raise ValueError(f'Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}')

    root_path = ds_cfg['root_path'](PROJECT_ROOT)
    enc_in = ds_cfg['enc_in']

    args = argparse.Namespace(
        # Task
        task_name='long_term_forecast',
        is_training=1,
        model=model_name,
        model_id=f'{dataset}_{seq_len}_{pred_len}',

        # Data
        data=ds_cfg['data'],
        root_path=root_path,
        data_path=ds_cfg['data_path'],
        features='M',
        target=ds_cfg['target'],
        freq=ds_cfg['freq'],
        embed='timeF',
        seasonal_patterns='Monthly',

        # Forecasting
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        inverse=False,

        # Model general
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=defaults.get('d_model', 512),
        n_heads=defaults.get('n_heads', 8),
        e_layers=defaults.get('e_layers', 2),
        d_layers=defaults.get('d_layers', 1),
        d_ff=defaults.get('d_ff', 2048),
        moving_avg=defaults.get('moving_avg', 25),
        factor=defaults.get('factor', 1),
        distil=True,
        activation='gelu',
        output_attention=False,
        channel_independence=0,
        top_k=defaults.get('top_k', 5),
        num_kernels=defaults.get('num_kernels', 6),

        # FilterNet specific
        embed_size=defaults.get('embed_size', 128),
        hidden_size=defaults.get('hidden_size', 256),

        # FITS / CompactFreq specific
        cut_freq=defaults.get('cut_freq', 12),

        # CompactFreq specific
        rank=defaults.get('rank', 14),
        time_bottleneck=defaults.get('time_bottleneck', None),

        # PatchTST specific
        patch_len=defaults.get('patch_len', 16),
        stride=defaults.get('stride', 8),

        # RLinear specific
        individual=defaults.get('individual', False),
        rev=defaults.get('rev', True),

        # Training
        dropout=defaults.get('dropout', 0),
        batch_size=defaults.get('batch_size', 32),
        learning_rate=defaults.get('learning_rate', 0.01),
        train_epochs=defaults.get('train_epochs', 20),
        patience=defaults.get('patience', 6),
        weight_decay=defaults.get('weight_decay', 0),
        num_workers=0,
        itr=1,
        des='Exp',
        loss='MSE',
        lradj=defaults.get('lradj', 'type1'),
        use_amp=False,

        # GPU
        use_gpu=torch.cuda.is_available(),
        gpu=defaults.get('gpu', 0),
        use_multi_gpu=False,
        devices='0',

        # Misc
        checkpoints=os.path.join(PROJECT_ROOT, 'checkpoints'),
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        seg_len=48,
    )

    return args


def get_model(model_name, args):
    """Import and instantiate the model."""
    if model_name == 'TexFilter':
        from models.TexFilter import Model
    elif model_name == 'PaiFilter':
        from models.PaiFilter import Model
    elif model_name == 'DLinear':
        from models.DLinear import Model
    elif model_name == 'RLinear':
        from models.RLinear import Model
    elif model_name == 'FITS':
        from models.FITS import Model
    elif model_name == 'iTransformer':
        from models.iTransformer import Model
    elif model_name == 'PatchTST':
        from models.PatchTST import Model
    elif model_name == 'FEDformer':
        from models.FEDformer import Model
    elif model_name == 'TimesNet':
        from models.TimesNet import Model
    elif model_name == 'CompactFreq':
        from models.CompactFreq import Model
    elif model_name == 'SpectralMixer':
        from models.SpectralMixer import Model
    elif model_name == 'SpectralAttn':
        from models.SpectralAttn import Model
    elif model_name == 'SpectraGate':
        from models.SpectraGate import Model
    elif model_name == 'FreTS':
        from models.FreTS import Model
    elif model_name == 'Autoformer':
        from models.Autoformer import Model
    elif model_name == 'Informer':
        from models.Informer import Model
    elif model_name == 'Pyraformer':
        from models.Pyraformer import Model
    elif model_name == 'MICN':
        from models.MICN import Model
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model = Model(args).float()
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train(model, args, device):
    """Training loop matching exp_long_term_forecasting.py exactly."""
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')

    setting = (f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}'
                f'_bs{args.batch_size}_lr{args.learning_rate}_drop{args.dropout}'
                f'_h{getattr(args, "hidden_size", "")}_r{getattr(args, "rank", "")}'
                f'_k{getattr(args, "cut_freq", "")}'
                f'_tb{getattr(args, "time_bottleneck", "") or ""}'
                f'_wd{getattr(args, "weight_decay", 0)}'
                f'_lradj{getattr(args, "lradj", "type1")}'
                f'_dm{getattr(args, "d_model", "")}_df{getattr(args, "d_ff", "")}'
                f'_el{getattr(args, "e_layers", "")}_dl{getattr(args, "d_layers", "")}')
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    wd = getattr(args, 'weight_decay', 0) or 0
    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=wd)
    criterion = nn.MSELoss()

    total_params, trainable_params = count_parameters(model)
    print(f'Model {model._get_name()} : total params: {total_params:,} | trainable: {trainable_params:,} ({trainable_params * 4 / 1024 / 1024:.4f}MB)')

    # Reset peak GPU memory tracking before training
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    train_start_time = time.time()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f'\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}')
                speed = (time.time() - epoch_time) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

            loss.backward()
            model_optim.step()

        print(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s')
        train_loss_avg = np.average(train_loss)
        vali_loss = validate(model, vali_loader, criterion, args, device)

        print(f'Epoch: {epoch + 1}, Steps: {train_steps} | '
              f'Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}')

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    train_time_s = time.time() - train_start_time
    peak_gpu_memory_mb = 0.0
    if device.type == 'cuda':
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    print(f'Training time: {train_time_s:.1f}s | Peak GPU memory: {peak_gpu_memory_mb:.1f}MB')

    # Load best model
    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, total_params, trainable_params, train_time_s, peak_gpu_memory_mb


def validate(model, data_loader, criterion, args, device):
    """Validation loop."""
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def test(model, args, device):
    """Test loop matching exp_long_term_forecasting.py exactly."""
    test_data, test_loader = data_provider(args, 'test')
    preds = []
    trues = []
    total_samples = 0

    model.eval()
    inference_start = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            total_samples += batch_x.shape[0]

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]

            preds.append(outputs)
            trues.append(batch_y)

    inference_time_s = time.time() - inference_start
    samples_per_sec = total_samples / inference_time_s if inference_time_s > 0 else 0

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'MSE: {mse:.6f}, MAE: {mae:.6f}')
    print(f'Inference: {inference_time_s:.2f}s | {total_samples} samples | {samples_per_sec:.1f} samples/sec')
    return mse, mae, inference_time_s, samples_per_sec


CSV_HEADER = [
    'model', 'dataset', 'pred_len', 'seq_len', 'mse', 'mae',
    'total_params', 'trainable_params', 'peak_gpu_memory_mb',
    'train_time_s', 'inference_time_s', 'inference_samples_per_sec',
    'batch_size', 'learning_rate', 'train_epochs', 'patience', 'dropout',
    'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff',
    'embed_size', 'hidden_size', 'cut_freq', 'rank', 'time_bottleneck',
    'patch_len', 'stride', 'moving_avg', 'factor',
    'weight_decay', 'lradj',
]


def save_results(model_name, dataset, pred_len, seq_len, mse, mae, args,
                 csv_path=None, total_params=0, trainable_params=0,
                 peak_gpu_memory_mb=0.0, train_time_s=0.0,
                 inference_time_s=0.0, samples_per_sec=0.0):
    """Append results to per-dataset CSV with file locking for concurrent safety."""
    if csv_path is None:
        ds_cfg = DATASET_CONFIGS.get(dataset, {})
        csv_name = ds_cfg.get('csv_name', f'{dataset.lower()}_results.csv')
        csv_path = os.path.join(PROJECT_ROOT, 'results', csv_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    row = [
        model_name, dataset, pred_len, seq_len, f'{mse:.6f}', f'{mae:.6f}',
        total_params, trainable_params, f'{peak_gpu_memory_mb:.1f}',
        f'{train_time_s:.1f}', f'{inference_time_s:.2f}', f'{samples_per_sec:.1f}',
        args.batch_size, args.learning_rate, args.train_epochs, args.patience, args.dropout,
        getattr(args, 'd_model', ''), getattr(args, 'n_heads', ''),
        getattr(args, 'e_layers', ''), getattr(args, 'd_layers', ''),
        getattr(args, 'd_ff', ''),
        getattr(args, 'embed_size', ''), getattr(args, 'hidden_size', ''),
        getattr(args, 'cut_freq', ''), getattr(args, 'rank', ''),
        getattr(args, 'time_bottleneck', '') or '',
        getattr(args, 'patch_len', ''), getattr(args, 'stride', ''),
        getattr(args, 'moving_avg', ''), getattr(args, 'factor', ''),
        getattr(args, 'weight_decay', 0), getattr(args, 'lradj', 'type1'),
    ]

    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # Check if header needed (file might have been created by another process)
            f.seek(0, 2)  # seek to end
            if f.tell() == 0:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            writer = csv.writer(f)
            writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    print(f'Results saved to {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='FilterNet Multi-Dataset Reproduction')
    parser.add_argument('--model', type=str, required=True,
                        choices=['TexFilter', 'PaiFilter', 'DLinear', 'RLinear',
                                 'FITS', 'iTransformer', 'PatchTST', 'FEDformer',
                                 'TimesNet', 'CompactFreq', 'SpectralMixer', 'SpectralAttn',
                                 'SpectraGate', 'FreTS', 'Autoformer', 'Informer',
                                 'Pyraformer', 'MICN'],
                        help='Model name')
    parser.add_argument('--dataset', type=str, default='Weather',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name')
    parser.add_argument('--pred_len', type=int, required=True,
                        choices=[96, 192, 336, 720])
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--seed', type=int, default=2021)

    # Overridable hyperparameters
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--embed_size', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--train_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--e_layers', type=int, default=None)
    parser.add_argument('--d_layers', type=int, default=None)
    parser.add_argument('--d_ff', type=int, default=None)
    parser.add_argument('--cut_freq', type=int, default=None)
    parser.add_argument('--patch_len', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--time_bottleneck', type=int, default=None)
    parser.add_argument('--lradj', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV filename (in ./results/). Default: per-dataset naming.')

    cli_args = parser.parse_args()

    # Build overrides from CLI
    overrides = {}
    for key in ['batch_size', 'learning_rate', 'embed_size', 'hidden_size',
                'train_epochs', 'patience', 'dropout', 'd_model', 'n_heads',
                'e_layers', 'd_layers', 'd_ff', 'cut_freq', 'patch_len',
                'stride', 'rank', 'time_bottleneck', 'lradj', 'weight_decay', 'gpu']:
        val = getattr(cli_args, key)
        if val is not None:
            overrides[key] = val

    set_seed(cli_args.seed)

    args = build_args(
        cli_args.model, cli_args.dataset, cli_args.pred_len,
        seq_len=cli_args.seq_len, label_len=cli_args.label_len,
        overrides=overrides,
    )

    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')
    print(f'Using device: {device}')
    print(f'Model: {args.model}, Dataset: {cli_args.dataset}, '
          f'pred_len: {args.pred_len}, seq_len: {args.seq_len}')

    model = get_model(args.model, args).to(device)

    print('\n>>> Training...')
    model, total_params, trainable_params, train_time_s, peak_gpu_memory_mb = train(model, args, device)

    print('\n>>> Testing...')
    mse, mae, inference_time_s, samples_per_sec = test(model, args, device)

    csv_path = None
    if cli_args.output_csv:
        csv_path = os.path.join(PROJECT_ROOT, 'results', cli_args.output_csv)
    save_results(args.model, cli_args.dataset, args.pred_len, args.seq_len,
                 mse, mae, args, csv_path=csv_path,
                 total_params=total_params, trainable_params=trainable_params,
                 peak_gpu_memory_mb=peak_gpu_memory_mb, train_time_s=train_time_s,
                 inference_time_s=inference_time_s, samples_per_sec=samples_per_sec)

    return mse, mae


if __name__ == '__main__':
    main()
