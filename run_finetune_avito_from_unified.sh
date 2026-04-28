#!/bin/bash
# Per-task fine-tune sweep on rel-avito tasks, starting from the UnifiedDecoder
# transfer checkpoint (others-1 → others-2 with Griffin-style back_proj + dec / @y.T).
#
# Inner per-task FT loop in hmaintask_combine_llm.py handles isolation:
#   For each task → reset head/projector to checkpoint → fine-tune on N samples
#   of that task only → test on that task.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_avito_from_unified.sh 2>&1 | tee logs/avito-from-unified-sweep.log

CHECKPOINT="checkpoints/transfer-o1-to-o2-unified/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"
AVITO_TASKS="rel-avito-user-visits rel-avito-user-clicks rel-avito-ad-ctr"

echo "================================================"
echo "Per-task FT sweep: UnifiedDecoder checkpoint → rel-avito tasks"
echo "Checkpoint: $CHECKPOINT"
echo "Tasks: $AVITO_TASKS"
echo "================================================"

# Zero-shot baseline (no FT)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/avito-uni-0 avito-uni-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks $AVITO_TASKS --unified_decoder \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep N = 512..4096 in steps of 512
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/avito-uni-$N avito-uni-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks $AVITO_TASKS --unified_decoder \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "Parse with:"
echo "  python parse_sweep_results.py logs/avito-from-unified-sweep.log"
echo "================================================"
