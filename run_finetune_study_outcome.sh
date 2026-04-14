#!/bin/bash
# Focused fine-tuning sweep for rel-trial-study-outcome alone.
# Sweeps LR × epochs × sample count to close gap vs Rel-LLM (0.72 AUROC).

CHECKPOINT="checkpoints/transfer-o1-to-o2-tth/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"
TASK="rel-trial-study-outcome"

echo "================================================"
echo "Focused fine-tune: $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/so-zs so-zs \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks $TASK --task_type_heads \
    --loadpath $CHECKPOINT \
    --batchsize 8 --pool_mode entity

# Sweep: LR × epochs × N
for N in 1024 2048 4096; do
  for LR in 5e-4 1e-4 5e-5; do
    for EPOCHS in 10 20; do
      TAG="so-N${N}-lr${LR}-ep${EPOCHS}"
      echo ""
      echo "=== $TAG ==="
      PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
          $DATASET logs/$TAG $TAG \
          --mode test \
          --head llm_mlp --llm_model $MODEL \
          --tasks others-1 --eval_tasks $TASK --task_type_heads \
          --loadpath $CHECKPOINT \
          --batchsize 8 --pool_mode entity \
          --finetune_samples $N --finetune_epochs $EPOCHS --finetune_lr $LR \
          --finetune_projector
    done
  done
done

echo ""
echo "================================================"
echo "Sweep complete! Check logs/so-* for best AUROC."
echo "================================================"
