#!/bin/bash
# Per-task fine-tune sweep on 7 OOD RelBench tasks (rel-amazon + rel-hm + rel-trial),
# starting from the UnifiedDecoder transfer checkpoint (others-1 → others-2 with
# Griffin-style back_proj + reg_dec / @y.T).
#
# Mix: 3 binary classification + 4 regression. None of these tasks are in others-1,
# so all are out-of-distribution for this checkpoint.
#
# Per-task isolation: for each task → reset back_proj/reg_dec to checkpoint state
# → fine-tune on N samples of that task only → test on that task.
#
# --finetune_projector is enabled because the UnifiedDecoder's @y.T classification
# path has no learnable classification head — adapting the projector gives binary
# tasks more capacity to align with the target distribution.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_ood_from_unified.sh 2>&1 | tee logs/ood-from-unified-sweep.log
#   python parse_sweep_results.py logs/ood-from-unified-sweep.log

CHECKPOINT="checkpoints/transfer-o1-to-o2-unified/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

TASKS="amazon-churn amazon-rating \
       rel-hm-user-churn rel-hm-item-sales \
       rel-trial-study-outcome rel-trial-study-adverse rel-trial-site-success"

echo "================================================"
echo "Per-task FT sweep: UnifiedDecoder o1 checkpoint → 7 OOD RelBench tasks"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: $TASKS"
echo "================================================"

# Zero-shot baseline (no FT)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/ood-uni-0 ood-uni-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks $TASKS --unified_decoder \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep N = 512..4096 in steps of 512
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/ood-uni-$N ood-uni-$N \
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
echo "Sweep complete!"
echo "Parse with:"
echo "  python parse_sweep_results.py logs/ood-from-unified-sweep.log"
echo "================================================"
