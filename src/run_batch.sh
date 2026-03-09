#!/bin/bash
# Batch runner for TexFilter and PaiFilter baselines on 5 datasets
# GPU 0: TexFilter, GPU 1: PaiFilter
# Usage: bash src/run_batch.sh <model> <gpu>
set -e
source env/bin/activate

MODEL=$1
GPU=$2

DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 Traffic"
PRED_LENS="96 192 336 720"

for ds in $DATASETS; do
    for pl in $PRED_LENS; do
        echo "========================================"
        echo "Running $MODEL on $ds pred_len=$pl GPU=$GPU"
        echo "========================================"
        python src/run_experiment.py --model $MODEL --dataset $ds --pred_len $pl --gpu $GPU 2>&1 | grep -E "^(MSE|Results|Model|Using|test shape|Early|Epoch:.*Vali)" | tail -5
        echo ""
    done
done
echo "DONE: All $MODEL experiments complete"
