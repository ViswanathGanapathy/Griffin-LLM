#!/bin/bash
# Stage 1: train Griffin + Projector + LLM (frozen, +LoRA) + TaskTypeHeads
# on commerce-1 or commerce-2, validating per-epoch on the other split.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_commerce_tth_lora.sh c1_to_c2 \
#     2>&1 | tee logs/c1-tth-lora-v3.log
#   CUDA_VISIBLE_DEVICES=1 ./run_train_commerce_tth_lora.sh c2_to_c1 \
#     2>&1 | tee logs/c2-tth-lora-v3.log
#
# Saves checkpoints/c1-tth-lora-v3 or checkpoints/c2-tth-lora-v3 respectively.

set -e
DIR=${1:-}

case "$DIR" in
    c1_to_c2)
        TRAIN=commerce-1
        EVAL=commerce-2
        TAG=c1-tth-lora-v3
        ;;
    c2_to_c1)
        TRAIN=commerce-2
        EVAL=commerce-1
        TAG=c2-tth-lora-v3
        ;;
    *)
        echo "Usage: $0 {c1_to_c2|c2_to_c1}"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 1: $TRAIN -> $EVAL (TTH + LoRA + neighbor_tokens=8)"
echo "Save: checkpoints/$TAG"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/$TAG $TAG \
    --head llm_mlp --llm_model $MODEL \
    --tasks $TRAIN --eval_tasks $EVAL \
    --task_type_heads \
    --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 \
    --pool_mode entity \
    --eval_per_epoch 1 \
    --savepath checkpoints/$TAG
