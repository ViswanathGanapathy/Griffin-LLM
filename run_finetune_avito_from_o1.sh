#!/bin/bash
# Per-task fine-tune sweep on rel-avito tasks, starting from the others-1 (TaskTypeHeads) transfer checkpoint.
# For each N: the inner per-task FT loop in hmaintask_combine_llm.py
#   1. Resets head/projector to checkpoint state for each task
#   2. Fine-tunes on N samples of that task only
#   3. Tests on that task
# So all three avito tasks are evaluated independently in a single invocation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_avito_from_o1.sh 2>&1 | tee logs/avito-from-o1-sweep.log

CHECKPOINT="checkpoints/transfer-o1-to-o2-tth/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"
AVITO_TASKS="rel-avito-user-visits rel-avito-user-clicks rel-avito-ad-ctr"

echo "================================================"
echo "Per-task FT sweep: o1 checkpoint → rel-avito tasks"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: $AVITO_TASKS"
echo "================================================"

# Zero-shot baseline (no FT)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/avito-o1-0 avito-o1-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks $AVITO_TASKS --task_type_heads \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep N = 512..4096 in steps of 512
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/avito-o1-$N avito-o1-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks $AVITO_TASKS --task_type_heads \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "Parse with:"
echo "  python parse_sweep_results.py logs/avito-from-o1-sweep.log"
echo "================================================"
