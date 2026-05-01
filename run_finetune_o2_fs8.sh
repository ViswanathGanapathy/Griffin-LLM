#!/bin/bash
# Stage 2 — per-task FT sweep on the fewshotfanout=8 checkpoint
# (run_train_o1_cot.sh trained o1-tth-lora-fs8). Mirrors the basic v3
# FT sweep but adds --fewshotfanout 8 to match the stage-1 training.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_fs8.sh \
#     2>&1 | tee logs/o2-from-fs8-sweep.log

CHECKPOINT="checkpoints/o1-tth-lora-fs8/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on others-2 (fewshotfanout=8)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (re-run for log consistency)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/o2-fs8-0 o2-fs8-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --fewshotfanout 8 \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# N x LR grid (matches extended o2 sweep cell choices)
for LR in 3e-4 1e-3; do
  for N in 1024 2048 4096; do
    TAG="o2-fs8-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --fewshotfanout 8 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "Done. Parse: python parse_sweep_results.py logs/o2-from-fs8-sweep.log"
