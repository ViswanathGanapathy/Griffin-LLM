#!/bin/bash
# Extended per-task FT sweep on commerce-2 from c1-tth-lora-v3 checkpoint.
# Sweeps N x LR to recover stuck tasks from the basic sweep:
#   - amazon-churn (0.5558 best)        — try lower LR, more N
#   - outbrain-small-ctr (0.5057 best)  — same
#   - amazon-rating (-0.76 best, climbing) — try N=8192
#   - rel-avito-ad-ctr (-0.85 best)     — try lower LR
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_c2_extended.sh \
#     2>&1 | tee logs/c2-from-c1-extended.log
#   python parse_sweep_results.py logs/c2-from-c1-extended.log

CHECKPOINT="checkpoints/c1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Extended FT: c1 -> c2 (N x LR grid)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-3; do
  for N in 2048 4096 8192; do
    TAG="c2-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks commerce-1 --eval_tasks commerce-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "================================================"
echo "Done. Parse: python parse_sweep_results.py logs/c2-from-c1-extended.log"
echo "================================================"
