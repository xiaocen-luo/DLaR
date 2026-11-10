#!/bin/bash
# run_extract_all.sh

cd /public/home/xiangyuduan/xcluo/lepo-main
export PYTHONPATH=$PWD:$PYTHONPATH

MODEL_BASE="/public/home/xiangyuduan/models/hf/Llama-3.2-1B-Instruct/"
MODEL_GRPO="/public/home/xiangyuduan/xcluo/lepo-main/outputs/-GRPO-llama/checkpoint-150/"
MODEL_LEPO="/public/home/xiangyuduan/xcluo/lepo-main/outputs/llama-lepo/checkpoint-150/"
MODEL_DLAR="/public/home/xiangyuduan/xcluo/lepo-main/outputs/-llama-DLaR/checkpoint-50/"

DATASETS=("gsm8k" "math500" "minervamath" "amc23" "mmlu_stem" "arc_challenge")
#DATASETS=("mmlu_stem" "arc_challenge" "minervamath" "amc23")
NUM_SAMPLES=200

for dataset in "${DATASETS[@]}"; do
    echo "Processing $dataset..."

    # Baseline
    python scripts/extract_hidden_states.py \
        --model_path $MODEL_BASE \
        --output_dir ./analysis/hidden_states/init/ \
        --dataset $dataset \
        --num_samples $NUM_SAMPLES \
        --baseline

    # GRPO
    python scripts/extract_hidden_states.py \
        --model_path $MODEL_GRPO \
        --output_dir ./analysis/hidden_states/grpo/ \
        --dataset $dataset \
        --num_samples $NUM_SAMPLES \
        --pca_ref_dir ./analysis/hidden_states/init/

    # LEPO
    python scripts/extract_hidden_states.py \
        --model_path $MODEL_LEPO \
        --output_dir ./analysis/hidden_states/lepo/ \
        --dataset $dataset \
        --num_samples $NUM_SAMPLES \
        --pca_ref_dir ./analysis/hidden_states/init/

    # DLaR
    python scripts/extract_hidden_states.py \
        --model_path $MODEL_DLAR \
        --output_dir ./analysis/hidden_states/dlar/ \
        --dataset $dataset \
        --num_samples $NUM_SAMPLES \
        --pca_ref_dir ./analysis/hidden_states/init/
done