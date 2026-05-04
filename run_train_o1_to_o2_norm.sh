#!/bin/bash
# Stage 1: train on others-1, validate on others-2, with --target_normalize.
# Mirrors run_train_o2_to_o1_norm.sh but for the canonical o1->o2 direction.
#
# Hypothesis: target_normalize lifts site-success / study-adverse ZS MAE
# substantially. Both other directions (c1->c2, o2->o1) have shown
# normalization to be a major win. This applies the same lever to the
# most-explored transfer direction.
#
# Modes:
#   norm        - --target_normalize on (default)
#   noflag      - control (reproduces v3 baseline)
#   huber-norm  - Huber + normalization
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_o1_to_o2_norm.sh \
#     2>&1 | tee logs/o1-to-o2-norm.log

MODE=${1:-norm}
case "$MODE" in
    norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="o1-to-o2-norm"
        ;;
    noflag|raw)
        EXTRA_FLAGS=""
        TAG="o1-to-o2-raw"
        ;;
    huber-norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="o1-to-o2-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

echo "================================================"
echo "Stage 1: others-1 -> others-2 ($MODE)"
echo "Tag: $TAG"
echo "Key targets: rel-trial-site-success (-0.94), rel-trial-study-adverse (-1.88)"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/$TAG $TAG \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 --pool_mode entity --eval_per_epoch 1 \
    --savepath checkpoints/$TAG
