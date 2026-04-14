#!/bin/bash
# Fine-tuning sweep for UnifiedDecoder transfer checkpoint (o1 → o2).
# Sweeps N = 512, 1024, ..., 4096 samples/task on others-2.

CHECKPOINT="checkpoints/transfer-o1-to-o2-unified/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Fine-tuning sweep (UnifiedDecoder): o1 → o2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot, no fine-tuning) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/ft-unified-0 ft-unified-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 --unified_decoder \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep
for N in 512 1024 1536 2048 2560 3072 3584 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/ft-unified-$N ft-unified-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 --unified_decoder \
        --loadpath $CHECKPOINT \
        --batchsize 8 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "================================================"
