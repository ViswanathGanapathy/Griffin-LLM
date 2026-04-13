#!/bin/bash
# Sweep fine-tuning samples: 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096
# Uses best T1 checkpoint (others-1 → others-2 with TaskTypeHeads)
# Fine-tunes output heads on N samples from others-2, then tests

CHECKPOINT="checkpoints/transfer-o1-to-o2-tth/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Fine-tuning sweep: others-1 checkpoint → others-2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# First: zero-shot baseline (no fine-tuning)
echo ""
echo "=== N=0 (zero-shot, no fine-tuning) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/ft-sweep-0 ft-sweep-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 --task_type_heads \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep fine-tuning samples
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/ft-sweep-$N ft-sweep-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 --task_type_heads \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "================================================"
