#!/bin/bash
# Focused grid for rel-avito-user-visits only.
# Binary classification (AUROC), num_class=2, target_type=rel-avito-UserInfo.
#
# Current state on this task:
#   ZS (TTH+LoRA+NB v3, c1->c2):       0.5891
#   Basic FT best (N=4096 LR=1e-3 5ep): 0.6193  (c1->c2 basic sweep)
#   Target (RelBench/Rel-LLM ballpark): ~0.66+
#
# Strategy: explore LR x epochs x N grid centered around lower LR + more
# epochs (the recipe that worked for ad-ctr regression). For binary tasks
# the right LR might be moderately lower (3e-4, 1e-4) rather than very low
# (3e-5), since binary CE gradients have different dynamics than MSE.
#
# Grid: LR in {3e-4, 1e-4, 3e-5} x epochs in {10, 20} x N in {4096, 8192}
# = 12 cells. ~2-3h on H100.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_avito_uservisits_focused.sh \
#     2>&1 | tee logs/avito-uservisits-focused.log
#   python parse_sweep_results.py logs/avito-uservisits-focused.log

CHECKPOINT="checkpoints/c1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Focused grid for rel-avito-user-visits"
echo "Goal: lift AUROC from 0.62 (basic FT best) toward 0.66+ target"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-4 3e-5; do
  for E in 10 20; do
    for N in 4096 8192; do
      TAG="uv-N${N}-lr${LR}-ep${E}"
      echo ""
      echo "=== N=${N} LR=${LR} epochs=${E} ==="
      PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
          $DATASET logs/$TAG $TAG \
          --mode test \
          --head llm_mlp --llm_model $MODEL \
          --tasks commerce-1 --eval_tasks rel-avito-user-visits \
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
echo "================================================"
echo "Done."
echo "Parse:   python parse_sweep_results.py logs/avito-uservisits-focused.log"
echo "Compare: best vs basic-sweep best (0.6193 @ N=4096 LR=1e-3 ep=5)"
echo "================================================"
