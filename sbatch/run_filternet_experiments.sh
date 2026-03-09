#!/bin/bash
# ============================================================================
# FilterNet Paper Experiments — SLURM Submission Script
# ============================================================================
#
# Submits all experiments from FilterNet paper Tables 1 & 4 + SpectraGate.
# Each job runs src/run_experiment.py which trains one (model, dataset, pred_len)
# combo and appends results to the output CSV with file locking.
#
# Models (15 total):
#   Table 1 (9): TexFilter, PaiFilter, iTransformer, PatchTST, FEDformer,
#                TimesNet, DLinear, RLinear, FITS
#   Table 4 (5): FreTS, Autoformer, Informer, Pyraformer, MICN
#   Ours (1):    SpectraGate
#
# Datasets (8): Weather, Exchange, ECL, ETTh1, ETTh2, ETTm1, ETTm2, Traffic
# Pred lens (4): 96, 192, 336, 720
# Total: up to 15 × 8 × 4 = 480 jobs
#
# SENSITIVE CONFIG — DO NOT CHANGE:
#   - Module load order: module load gcc arrow scipy-stack cuda cudnn
#   - GPU slice configs (Rorqual H100 MIG: h100_2g.20gb, h100_3g.40gb, h100)
#   - Environment activation steps
#
# Usage:
#   ./sbatch/run_filternet_experiments.sh
#   ./sbatch/run_filternet_experiments.sh --account def-myprof
#   ./sbatch/run_filternet_experiments.sh --output_csv paper_results.csv
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT=""
OUTPUT_CSV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --output_csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--output_csv FILENAME]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION — Modify these arrays to control what runs
# ============================================================================

# Models to benchmark (comment/uncomment as needed)
models=(
    # --- Table 1: Main baselines (9 models) ---
    "TexFilter"
    "PaiFilter"
    "iTransformer"
    "PatchTST"
    "FEDformer"
    "TimesNet"
    "DLinear"
    "RLinear"
    "FITS"
    # --- Table 4: Additional baselines (5 models) ---
    "FreTS"
    "Autoformer"
    "Informer"
    "Pyraformer"
    "MICN"
    # --- Ours ---
    "SpectraGate"
)

# Datasets to evaluate
datasets=(
    #"Weather"
    #"Exchange"
    #"ECL"
    #"ETTh1"
    #"ETTh2"
    #"ETTm1"
    "ETTm2"
    #"Traffic"
)

# Prediction lengths
pred_lens=(96) # 192 336 720)

# ============================================================================
# SLURM RESOURCE CONFIGURATION
# ============================================================================

get_job_config() {
    # Sets: gpu_type, mem, time_limit
    # Based on model complexity and dataset size
    # Rorqual MIG slices: h100_1g.10gb, h100_2g.20gb, h100_3g.40gb, h100 (full 80GB)
    local model=$1
    local dataset=$2

    # Default: 20GB MIG slice, sufficient for most combos
    gpu_type="h100_2g.20gb:1"
    mem="20000M"
    time_limit="4:00:00"

    # Large models (transformer-based) on large datasets need more resources
    case $dataset in
        ECL|Traffic)
            gpu_type="h100_3g.40gb:1"
            mem="40000M"
            time_limit="12:00:00"
            ;;
        ETTm1|ETTm2)
            # Longer datasets (69K rows)
            time_limit="8:00:00"
            ;;
    esac

    # SpectraGate / CompactFreq train longer (100 epochs)
    case $model in
        SpectraGate|CompactFreq|SpectralMixer|SpectralAttn|FITS)
            case $dataset in
                ECL|Traffic)
                    time_limit="24:00:00"
                    ;;
                ETTm1|ETTm2)
                    time_limit="12:00:00"
                    ;;
                *)
                    time_limit="8:00:00"
                    ;;
            esac
            ;;
    esac
}

# ============================================================================
# HYPERPARAMETER BUILDER
# ============================================================================
#
# All hyperparameters are encoded in src/run_experiment.py's per-dataset
# defaults dicts. The runner handles all config lookup automatically.
# We only need to pass --model, --dataset, --pred_len, and optional overrides.
#
# Per-model CLI overrides can be added here if needed for specific combos.

build_python_cmd() {
    local model=$1
    local dataset=$2
    local pred_len=$3

    local cmd="python src/run_experiment.py"
    cmd+=" --model $model"
    cmd+=" --dataset $dataset"
    cmd+=" --pred_len $pred_len"

    if [[ -n "$OUTPUT_CSV" ]]; then
        cmd+=" --output_csv $OUTPUT_CSV"
    fi

    echo "$cmd"
}

# ============================================================================
# MAIN LOOP — Submit one job per (model, dataset, pred_len)
# ============================================================================

job_count=0
mkdir -p ./logs ./results

echo "============================================"
echo "FilterNet Paper Experiments — Job Submission"
echo "============================================"
echo "Models:    ${models[*]}"
echo "Datasets:  ${datasets[*]}"
echo "Pred lens: ${pred_lens[*]}"
echo "Output CSV: ${OUTPUT_CSV:-per-dataset default}"
echo "============================================"
echo ""

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        get_job_config "$model" "$dataset"

        for pred_len in "${pred_lens[@]}"; do
            job_name="fn_${model}_${dataset}_${pred_len}"
            python_cmd=$(build_python_cmd "$model" "$dataset" "$pred_len")

            account_line=""
            if [[ -n "$ACCOUNT" ]]; then
                account_line="#SBATCH --account=$ACCOUNT"
            fi

            sbatch_id=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=./logs/${job_name}_%j.out
#SBATCH --error=./logs/${job_name}_%j.err
#SBATCH --time=$time_limit
#SBATCH --gpus=$gpu_type
#SBATCH --mem=$mem
#SBATCH --cpus-per-task=4
$account_line

            module load gcc arrow scipy-stack cuda cudnn
            source ./env/bin/activate

            export HF_HOME=\$(pwd)/data
            export HF_DATASETS_CACHE=\$(pwd)/data
            export TRANSFORMERS_CACHE=\$(pwd)/data
            export TORCH_HOME=\$(pwd)/data
            mkdir -p \$HF_HOME

            echo '========================================'
            echo 'Job: $job_name'
            echo 'Model: $model'
            echo 'Dataset: $dataset'
            echo 'Pred len: $pred_len'
            echo 'Time limit: $time_limit'
            echo 'GPU: $gpu_type'
            echo 'Started: '\$(date)
            echo '========================================'
            nvidia-smi
            $python_cmd
            echo '========================================'
            echo 'Finished: '\$(date)
            echo '========================================'
EOF
)
            echo "  [$sbatch_id] $job_name  ($model, $dataset, pred_len=$pred_len, $time_limit)"
            ((job_count++))
        done
    done
done

echo ""
echo "============================================"
echo "Total jobs submitted: $job_count"
echo "Results directory:    ./results/"
echo "Logs directory:       ./logs/"
echo "============================================"
