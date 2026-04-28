#!/bin/bash
# Per-task FT sweep — classification only — UnifiedDecoder o1 checkpoint.
# 3 binary OOD tasks. Sweep N = 1024..4096 in steps of 512.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_ood_classification.sh 2>&1 | tee logs/ood-cls-sweep.log

CHECKPOINT="checkpoints/transfer-o1-to-o2-unified/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

TASKS="amazon-churn rel-hm-user-churn rel-trial-study-outcome"

echo "================================================"
echo "OOD classification FT sweep (UnifiedDecoder, from o1)"
echo "Tasks: $TASKS"
echo "================================================"

echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/ood-cls-0 ood-cls-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks $TASKS --unified_decoder \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

for N in 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/ood-cls-$N ood-cls-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks $TASKS --unified_decoder \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3 \
        --finetune_projector
done

echo ""
echo "================================================"
echo "Classification sweep complete!"
echo "Parse: python parse_sweep_results.py logs/ood-cls-sweep.log"
echo "================================================"
