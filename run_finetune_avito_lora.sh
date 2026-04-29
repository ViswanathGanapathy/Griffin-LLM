#!/bin/bash
# Per-task FT sweep on rel-avito tasks, starting from the LoRA-trained
# holdout-avito-tth-lora checkpoint. Each task is fine-tuned and tested
# in isolation (per-task FT loop in hmaintask_combine_llm.py).
#
# Trainable per task (with --finetune_projector --finetune_lora):
#   TaskTypeHeads (~150K) + Projector (~2.6M) + LoRA adapters (~5-10M)
#
# Sweeps N = 1024..4096 in steps of 1024 (for speed; expand if needed).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_avito_lora.sh 2>&1 | tee logs/avito-lora-sweep.log

CHECKPOINT="checkpoints/holdout-avito-tth-lora/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Per-task FT (LoRA + projector + heads): rel-avito tasks"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (re-run for log consistency)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/avito-lora-0 avito-lora-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks rel-train-no-avito --eval_tasks rel-avito-eval \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# Sweep N
for N in 1024 2048 4096; do
    echo ""
    echo "=== N=$N samples/task (LoRA FT) ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/avito-lora-$N avito-lora-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks rel-train-no-avito --eval_tasks rel-avito-eval \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3 \
        --finetune_projector --finetune_lora
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "Parse: python parse_sweep_results.py logs/avito-lora-sweep.log"
echo "================================================"
