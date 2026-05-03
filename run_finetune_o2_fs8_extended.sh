#!/bin/bash
# Stage 2 — per-task FT sweep on others-2 from the fewshot=8 trained checkpoint.
# Mirrors run_finetune_o2_extended.sh but adds --fewshotfanout 8 to match
# the stage-1 graph context. Sweeps N ∈ {1024, 2048, 4096} × LR ∈ {3e-4, 1e-3}.
#
# Tests whether the +0.013 ZS gain from fewshot=8 amplifies with per-task FT.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_fs8_extended.sh \
#     2>&1 | tee logs/o2-fs8-extended.log
#   python parse_sweep_results.py logs/o2-fs8-extended.log
#   python parse_sweep_results.py logs/o2-from-o1-extended.log logs/o2-fs8-extended.log
#     # → side-by-side comparison vs the v3 (fewshot=3) baseline

CHECKPOINT="checkpoints/o1-tth-lora-fs8/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on others-2 (fewshot=8 checkpoint)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (already known: avg -1.0089) — re-printed for log consistency
echo ""
echo "=== N=0 (zero-shot, fewshot=8) ==="
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

# N × LR grid (6 cells)
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
echo "================================================"
echo "Done."
echo "Parse:  python parse_sweep_results.py logs/o2-fs8-extended.log"
echo "Compare:"
echo "  python parse_sweep_results.py logs/o2-from-o1-extended.log logs/o2-fs8-extended.log"
echo "================================================"
