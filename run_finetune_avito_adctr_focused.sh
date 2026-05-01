#!/bin/bash
# Focused grid for rel-avito-ad-ctr only — the regression scale mismatch
# task. Current best: MAE 0.7967 (Rel-LLM benchmark: 0.033, ~24x gap).
#
# The shared regression head was trained on amazon-rating (1-5) and
# rel-hm-item-sales (counts), so it outputs values in those ranges.
# Per-task FT must recalibrate down to CTR's [0, 0.1] range.
# Standard FT recipe (LR=1e-3, 5 epochs) can't achieve this in a few epochs.
#
# Strategy: longer schedule + lower LR to let the regression head's scale slowly shift.
# Grid: LR in {3e-4, 1e-4, 3e-5} x epochs in {10, 20, 30} x N in {4096, 8192}.
# = 18 cells. ~3-5h on H100.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_avito_adctr_focused.sh \
#     2>&1 | tee logs/avito-adctr-focused.log

CHECKPOINT="checkpoints/c1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Focused grid for rel-avito-ad-ctr"
echo "Goal: recalibrate regression head from rating-scale to CTR-scale"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-4 3e-5; do
  for E in 10 20 30; do
    for N in 4096 8192; do
      TAG="adctr-N${N}-lr${LR}-ep${E}"
      echo ""
      echo "=== N=${N} LR=${LR} epochs=${E} ==="
      PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
          $DATASET logs/$TAG $TAG \
          --mode test \
          --head llm_mlp --llm_model $MODEL \
          --tasks commerce-1 --eval_tasks rel-avito-ad-ctr \
          --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
          --neighbor_tokens 8 \
          --loadpath $CHECKPOINT \
          --batchsize 4 --pool_mode entity \
          --finetune_samples $N --finetune_epochs $E --finetune_lr $LR \
          --finetune_projector --finetune_lora
    done
  done
done

echo ""
echo "Done. Parse: python parse_sweep_results.py logs/avito-adctr-focused.log"
