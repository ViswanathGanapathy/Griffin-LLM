#!/bin/bash
# Stage 1: train on commerce-1, validate on commerce-2, with --target_normalize.
#
# Why this direction is the clearest test of target normalization:
#   - commerce-1 regression task (rel-hm-item-sales): counts, 0 to ~100s
#   - commerce-2 regression task (rel-avito-ad-ctr): CTR fractions 0 to ~0.1
#   - Scale ratio ~1000x — largest mismatch in the project
#   - Current best on ad-ctr without normalization: MAE 0.79 (focused FT)
#   - Rel-LLM benchmark: MAE 0.033 (24x better)
#
# Strong hypothesis: normalization closes most of the 24x gap because the
# bottleneck is regression scale, not architectural capacity.
#
# Modes:
#   norm        - --target_normalize on (default; the key experiment)
#   noflag      - no normalization (control / Cell A — should reproduce existing -1.91 ZS)
#   huber-norm  - Huber + normalization (Cell D)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_c1_to_c2_norm.sh        # Cell C
#   CUDA_VISIBLE_DEVICES=1 ./run_train_c1_to_c2_norm.sh huber-norm

MODE=${1:-norm}
case "$MODE" in
    norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="c1-to-c2-norm"
        ;;
    noflag|raw)
        EXTRA_FLAGS="--no_target_normalize"
        TAG="c1-to-c2-raw"
        ;;
    huber-norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="c1-to-c2-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

echo "================================================"
echo "Stage 1: commerce-1 -> commerce-2 ($MODE)"
echo "Tag: $TAG"
echo "Key target: rel-avito-ad-ctr MAE (current: 0.79; Rel-LLM: 0.033)"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/$TAG $TAG \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks commerce-1 --eval_tasks commerce-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 --pool_mode entity --eval_per_epoch 1 \
    --savepath checkpoints/$TAG
