#!/bin/bash
# Per-task fine-tuning sweep for commerce transfer checkpoints (UnifiedDecoder).
# For each N in {512..4096}, and for each eval task, the code:
#   1. Resets head/projector to checkpoint state
#   2. Fine-tunes on N samples of THAT task only
#   3. Tests on THAT task
# Handled internally by hmaintask_combine_llm.py per-task FT loop.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 ./run_finetune_sweep_commerce.sh c1_to_c2
#   CUDA_VISIBLE_DEVICES=3 ./run_finetune_sweep_commerce.sh c2_to_c1
# Run both in parallel on different GPUs.

DIRECTION=${1:-c1_to_c2}

if [ "$DIRECTION" = "c1_to_c2" ]; then
    CHECKPOINT="checkpoints/transfer-c1-to-c2-unified/best_checkpoint"
    TRAIN_TASKS="commerce-1"
    EVAL_TASKS="commerce-2"
    TAG="c1c2"
elif [ "$DIRECTION" = "c2_to_c1" ]; then
    CHECKPOINT="checkpoints/transfer-c2-to-c1-unified/best_checkpoint"
    TRAIN_TASKS="commerce-2"
    EVAL_TASKS="commerce-1"
    TAG="c2c1"
else
    echo "Usage: $0 {c1_to_c2|c2_to_c1}"
    exit 1
fi

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Per-task fine-tune sweep: $DIRECTION (UnifiedDecoder)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (no fine-tuning)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/ft-${TAG}-0 ft-${TAG}-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks $TRAIN_TASKS --eval_tasks $EVAL_TASKS --unified_decoder \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep N = 512..4096 in steps of 512
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/ft-${TAG}-$N ft-${TAG}-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks $TRAIN_TASKS --eval_tasks $EVAL_TASKS --unified_decoder \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3
done

echo ""
echo "================================================"
echo "Sweep complete for $DIRECTION!"
echo "================================================"
