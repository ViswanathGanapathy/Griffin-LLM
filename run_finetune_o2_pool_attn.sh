#!/bin/bash
# Stage 2 — per-task FT sweep on others-2 from o1-tth-lora-pa checkpoint.
# IMPORTANT: --pool_mode and --pool_layers MUST match the values used during
# stage 1 — otherwise the TaskTypeHeads dimension won't match the saved
# checkpoint (entity → 2D input, attention → D input).
#
# Sweeps N x LR grid:
#   N    in {1024, 2048, 4096}
#   LR   in {3e-4, 1e-3}
# = 6 cells. ~6-9h on H100.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_pool_attn.sh \
#     2>&1 | tee logs/o2-from-pa-sweep.log
#   python parse_sweep_results.py logs/o2-from-pa-sweep.log

CHECKPOINT="checkpoints/o1-tth-lora-pa/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: pool_layers=4 + attention | per-task FT on others-2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (no FT)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/o2-pa-0 o2-pa-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --pool_mode attention --pool_layers 4 \
    --loadpath $CHECKPOINT \
    --batchsize 4

# Per-task FT sweep
for LR in 3e-4 1e-3; do
  for N in 1024 2048 4096; do
    TAG="o2-pa-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --pool_mode attention --pool_layers 4 \
        --loadpath $CHECKPOINT \
        --batchsize 4 \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "================================================"
echo "Done."
echo "Parse: python parse_sweep_results.py logs/o2-from-pa-sweep.log"
echo "================================================"
