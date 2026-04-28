#!/bin/bash
# Cross-validation across rel-* datasets (joint-v65).
# For each holdout: train UnifiedDecoder on all other rel-* tasks starting
# from single-sft, then zero-shot + per-task FT sweep on the held-out tasks.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 ./run_train_holdout.sh stack 2>&1 | tee logs/holdout-stack.log
#   CUDA_VISIBLE_DEVICES=2 ./run_train_holdout.sh f1    2>&1 | tee logs/holdout-f1.log
#   CUDA_VISIBLE_DEVICES=3 ./run_train_holdout.sh avito 2>&1 | tee logs/holdout-avito.log
#   CUDA_VISIBLE_DEVICES=0 ./run_train_holdout.sh hm    2>&1 | tee logs/holdout-hm.log
#
# rel-event is excluded everywhere (not in joint-v65).

set -e

HOLDOUT=${1:-}

case "$HOLDOUT" in
    stack)
        TRAIN_GROUP=rel-train-no-stack
        EVAL_GROUP=rel-stack-eval
        ;;
    f1)
        TRAIN_GROUP=rel-train-no-f1
        EVAL_GROUP=rel-f1-eval
        ;;
    avito)
        TRAIN_GROUP=rel-train-no-avito
        EVAL_GROUP=rel-avito-eval
        ;;
    hm)
        TRAIN_GROUP=rel-train-no-hm
        EVAL_GROUP=rel-hm-eval
        ;;
    *)
        echo "Usage: $0 {stack|f1|avito|hm}"
        echo "  stack: train on all rel-*, hold out rel-stack (stackexchange-*)"
        echo "  f1:    train on all rel-*, hold out rel-f1 (rel-f1-driver-*)"
        echo "  avito: train on all rel-*, hold out rel-avito"
        echo "  hm:    train on all rel-*, hold out rel-hm"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"
START_CKPT="checkpoints/downloaded/single-sft"
SAVE_CKPT="checkpoints/holdout-${HOLDOUT}-unified"
TRAIN_TAG="holdout-${HOLDOUT}-unified"

echo "================================================"
echo "Cross-validation: holdout = rel-${HOLDOUT}"
echo "  Train tasks group:  $TRAIN_GROUP"
echo "  Eval  tasks group:  $EVAL_GROUP"
echo "  Source ckpt:        $START_CKPT"
echo "  Save ckpt:          $SAVE_CKPT"
echo "================================================"

echo ""
echo "=== Stage 1: train UnifiedDecoder ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/$TRAIN_TAG $TRAIN_TAG \
    --head llm_mlp --llm_model $MODEL \
    --tasks $TRAIN_GROUP --eval_tasks $EVAL_GROUP --unified_decoder \
    --loadpath $START_CKPT \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 16 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 \
    --pool_mode entity \
    --eval_per_epoch 1 \
    --savepath $SAVE_CKPT

BEST_CKPT="${SAVE_CKPT}/best_checkpoint"

echo ""
echo "=== Stage 2a: zero-shot test on held-out rel-${HOLDOUT} ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/${HOLDOUT}-eval-0 ${HOLDOUT}-eval-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks $TRAIN_GROUP --eval_tasks $EVAL_GROUP --unified_decoder \
    --loadpath $BEST_CKPT \
    --batchsize 8 --pool_mode entity

echo ""
echo "=== Stage 2b: per-task FT sweep on held-out rel-${HOLDOUT} ==="
for N in 1024 2048 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/${HOLDOUT}-eval-$N ${HOLDOUT}-eval-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks $TRAIN_GROUP --eval_tasks $EVAL_GROUP --unified_decoder \
        --loadpath $BEST_CKPT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3 \
        --finetune_projector
done

echo ""
echo "================================================"
echo "Done (holdout=$HOLDOUT). Parse with:"
echo "  python parse_sweep_results.py logs/holdout-${HOLDOUT}.log"
echo "================================================"
