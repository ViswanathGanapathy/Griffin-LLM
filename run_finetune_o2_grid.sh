#!/bin/bash
# Hyperparameter grid sweep for per-task FT on others-2, starting from
# o1-tth-lora-v3. Each cell of the grid runs the per-task FT loop across
# all 6 others-2 tasks; per-task winners are picked from parse_sweep_results.
#
# Grid:  N ∈ {1024, 2048, 4096}  ×  LR ∈ {3e-4, 1e-3}  ×  epochs ∈ {5, 10}
# = 12 runs covering all 6 tasks. ~10–14h on a single H100.
#
# Use this AFTER the basic run_finetune_o2_from_o1_tth_lora.sh sweep, when
# you want to hyper-tune individual tasks that haven't reached their ceiling.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_grid.sh 2>&1 | tee logs/o2-grid.log
#   python parse_sweep_results.py logs/o2-grid.log

CHECKPOINT="checkpoints/o1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "FT hyperparameter grid: N x LR x epochs on others-2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for N in 1024 2048 4096; do
  for LR in 3e-4 1e-3; do
    for E in 5 10; do
      TAG="o2-N${N}-lr${LR}-ep${E}"
      echo ""
      echo "=== N=$N LR=$LR epochs=$E ==="
      PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
          $DATASET logs/$TAG $TAG \
          --mode test \
          --head llm_mlp --llm_model $MODEL \
          --tasks others-1 --eval_tasks others-2 \
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
echo "Grid sweep complete!"
echo "Parse: python parse_sweep_results.py logs/o2-grid.log"
echo "================================================"
