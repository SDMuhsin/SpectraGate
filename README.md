# SpectraGate

**Parameter-Efficient Time Series Forecasting via Adaptive Spectral Gating**

> **Paper status:** Under review. No preprint is currently available. Please contact the corresponding author for a copy.

## Overview

SpectraGate is a lightweight spectral architecture for multivariate long-term time series forecasting. It processes each channel independently through three stages:

1. **Spectral extraction** via truncated real FFT for dimensionality reduction
2. **Adaptive spectral gating** that learns per-frequency attenuation weights from the observed spectrum (a learned Wiener-like filter)
3. **Direct spectral-to-temporal prediction** without inverse FFT reconstruction

With only 7K--36K trainable parameters, SpectraGate achieves the best average MSE rank across eight standard benchmarks, surpassing transformer baselines that require 150--700x more parameters.

## Setup

```bash
# Clone the repository
git clone https://github.com/SDMuhsin/SpectraGate.git
cd SpectraGate

# Create and activate a virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data

Download the standard benchmark datasets and place them under `data/`:

```
data/
  weather/weather.csv
  electricity/electricity.csv
  traffic/traffic.csv
  exchange_rate/exchange_rate.csv
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
```

All datasets are publicly available from the [Autoformer repository](https://github.com/thuml/Autoformer) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

## Usage

### Running experiments

```bash
python src/run_experiment.py \
    --model SpectraGate \
    --dataset Weather \
    --pred_len 96 \
    --gpu 0
```

### Available models

| Model | Type | Source |
|-------|------|--------|
| SpectraGate | Spectral gating (ours) | `models/SpectraGate.py` |
| TexFilter | Frequency filter | `models/TexFilter.py` |
| PaiFilter | Frequency filter | `models/PaiFilter.py` |
| iTransformer | Transformer | `models/iTransformer.py` |
| PatchTST | Transformer | `models/PatchTST.py` |
| FEDformer | Transformer | `models/FEDformer.py` |
| Autoformer | Transformer | `models/Autoformer.py` |
| TimesNet | Temporal | `models/TimesNet.py` |
| MICN | Convolution | `models/MICN.py` |
| DLinear | Linear | `models/DLinear.py` |
| RLinear | Linear | `models/RLinear.py` |
| FITS | Spectral linear | `models/FITS.py` |
| FreTS | Frequency MLP | `models/FreTS.py` |

### Available datasets

Weather, ECL, Traffic, ETTh1, ETTh2, ETTm1, ETTm2, Exchange

### CLI options

```bash
--model          # Model name (see table above)
--dataset        # Dataset name
--pred_len       # Prediction horizon: 96, 192, 336, or 720
--gpu            # GPU device index
--batch_size     # Training batch size
--learning_rate  # Initial learning rate
--train_epochs   # Maximum training epochs
--patience       # Early stopping patience
--dropout        # Dropout rate
--d_model        # Hidden dimension (D)
--cut_freq       # Spectral truncation order (K)
--d_ff           # Gate bottleneck dimension (Dg)
--lradj          # LR schedule: type1 or cosine
```

### Reproducing paper results

To reproduce all results from Table I:

```bash
# SpectraGate on all datasets (pred_len=96)
for dataset in Weather ECL Traffic ETTh1 ETTh2 ETTm1 ETTm2 Exchange; do
    python src/run_experiment.py --model SpectraGate --dataset $dataset --pred_len 96 --gpu 0
done
```

Default hyperparameters for each model-dataset combination are built into `src/run_experiment.py` and match the configurations used in the paper.

### SLURM (HPC)

An sbatch script for large-scale parallel runs is provided in `sbatch/run_filternet_experiments.sh`. Adjust the module loads and GPU partition names for your cluster.

## Results

Results are appended to per-dataset CSV files in `results/` (e.g., `results/weather_results.csv`).

## Repository structure

```
SpectraGate/
  models/             # All model implementations
    SpectraGate.py    # Proposed method
    TexFilter.py      # FilterNet (NeurIPS 2024)
    PaiFilter.py      # FilterNet (NeurIPS 2024)
    ...               # Other baselines
  src/
    run_experiment.py # Unified experiment runner
  data_provider/      # Dataset loading
  exp/                # Training/evaluation loops
  layers/             # Shared layers (RevIN, embeddings, attention)
  utils/              # Metrics, early stopping, LR scheduling
  scripts/            # Original FilterNet experiment scripts
  sbatch/             # SLURM job scripts
```

## Citation

```bibtex
@inproceedings{spectragate2025,
  title={SpectraGate: Parameter-Efficient Time Series Forecasting via Adaptive Spectral Gating},
  author={TBD},
  booktitle={TBD},
  year={2025}
}
```

## Acknowledgments

This work was supported by the Technology Innovation Program (RS-2025-02312262) funded by the Ministry of Trade, Industry & Energy (MOTIE), Korea.

This codebase builds on [FilterNet](https://github.com/aikunyi/FilterNet) (NeurIPS 2024).
