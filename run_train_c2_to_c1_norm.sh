#!/bin/bash
# Stage 1: train on commerce-2, validate on commerce-1, with --target_normalize.
#
# Reverse direction of c1->c2 (which lifted ZS avg from -0.706 to +0.041).
# Tests whether normalization helps when scale mismatch is also significant
# but in the opposite direction:
#   - c2 source regression: amazon-rating (1-5), rel-avito-ad-ctr (0-0.1)
#   - c1 eval regression:   rel-hm-item-sales (counts 0-100s)
#   - Source mean ~3, target mean ~10s — different but real scale gap
#
# Modes:
#   norm        - --target_normalize on (default)
#   noflag      - control / Cell A
#   huber-norm  - Huber + normalization
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_c2_to_c1_norm.sh \
#     2>&1 | tee logs/c2-to-c1-norm.log

MODE=${1:-norm}
case "$MODE" in
    norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="c2-to-c1-norm"
        ;;
    noflag|raw)
        EXTRA_FLAGS="--no_target_normalize"
        TAG="c2-to-c1-raw"
        ;;
    huber-norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="c2-to-c1-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

echo "================================================"
echo "Stage 1: commerce-2 -> commerce-1 ($MODE)"
echo "Tag: $TAG"
echo "Key target: rel-hm-item-sales (current best: -1.98)"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/$TAG $TAG \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks commerce-2 --eval_tasks commerce-1 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 --pool_mode entity --eval_per_epoch 1 \
    --savepath checkpoints/$TAG
