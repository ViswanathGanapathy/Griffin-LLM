#!/bin/bash
# Two-stage pipeline:
#   Stage 1: train Griffin + Projector + LLM (frozen) + UnifiedDecoder on
#            ALL tasks except rel-hm (13 tasks across rel-amazon, rel-avito,
#            rel-f1, rel-stack, rel-trial), starting from single-sft.
#   Stage 2: zero-shot test + per-task FT sweep on rel-hm (2 held-out tasks).
#
# NOTE: rel-event tasks are not in joint-v65 — they were excluded.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_no_hm_eval_hm.sh 2>&1 | tee logs/no-hm-to-hm.log

set -e

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"
START_CKPT="checkpoints/downloaded/single-sft"
SAVE_CKPT="checkpoints/train-no-hm-unified"
TRAIN_TAG="train-no-hm-unified"
TRAIN_LOG_DIR="logs/${TRAIN_TAG}"

echo "================================================"
echo "Stage 1: train UnifiedDecoder on rel-train-no-hm (13 tasks)"
echo "  Source ckpt: $START_CKPT"
echo "  Save ckpt:   $SAVE_CKPT"
echo "  Validation:  rel-hm-eval (rel-hm-user-churn, rel-hm-item-sales)"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET $TRAIN_LOG_DIR $TRAIN_TAG \
    --head llm_mlp --llm_model $MODEL \
    --tasks rel-train-no-hm --eval_tasks rel-hm-eval --unified_decoder \
    --loadpath $START_CKPT \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 16 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 \
    --pool_mode entity \
    --eval_per_epoch 1 \
    --savepath $SAVE_CKPT

BEST_CKPT="${SAVE_CKPT}/best_checkpoint"

echo ""
echo "================================================"
echo "Stage 2a: zero-shot test on rel-hm"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/hm-from-broad-0 hm-from-broad-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks rel-train-no-hm --eval_tasks rel-hm-eval --unified_decoder \
    --loadpath $BEST_CKPT \
    --batchsize 8 --pool_mode entity

echo ""
echo "================================================"
echo "Stage 2b: per-task FT sweep on rel-hm (N = 1024..4096)"
echo "================================================"

for N in 1024 2048 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/hm-from-broad-$N hm-from-broad-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks rel-train-no-hm --eval_tasks rel-hm-eval --unified_decoder \
        --loadpath $BEST_CKPT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3 \
        --finetune_projector
done

echo ""
echo "================================================"
echo "Done. Parse with:"
echo "  python parse_sweep_results.py logs/no-hm-to-hm.log"
echo "================================================"
