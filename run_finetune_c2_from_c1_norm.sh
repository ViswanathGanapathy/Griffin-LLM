#!/bin/bash
# Stage 2 — per-task FT sweep on commerce-2 from a c1-to-c2-* checkpoint.
# IMPORTANT: --target_normalize/--huber_loss flags must match the stage-1
# checkpoint. Pass the same mode arg as the training script.
#
# Adds a regression-friendly LR axis (3e-5) for ad-ctr specifically — that
# regime worked for the focused recipe (-0.79 MAE) and may compound with
# normalization.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_c2_from_c1_norm.sh        # Cell C FT
#   CUDA_VISIBLE_DEVICES=1 ./run_finetune_c2_from_c1_norm.sh noflag # control FT
#   CUDA_VISIBLE_DEVICES=2 ./run_finetune_c2_from_c1_norm.sh huber-norm

MODE=${1:-norm}
case "$MODE" in
    norm)
        CHECKPOINT="checkpoints/c1-to-c2-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="c2-from-c1-norm"
        ;;
    noflag|raw)
        CHECKPOINT="checkpoints/c1-to-c2-raw/best_checkpoint"
        EXTRA_FLAGS=""
        TAG="c2-from-c1-raw"
        ;;
    huber-norm)
        CHECKPOINT="checkpoints/c1-to-c2-huber-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="c2-from-c1-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on commerce-2 ($MODE)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/${TAG}-0 ${TAG}-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks commerce-1 --eval_tasks commerce-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# N x LR grid — includes 3e-5 for regression-friendly tuning
for LR in 3e-5 3e-4 1e-3; do
  for N in 2048 4096; do
    SUBTAG="${TAG}-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$SUBTAG $SUBTAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks commerce-1 --eval_tasks commerce-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        $EXTRA_FLAGS \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "================================================"
echo "Done."
echo "Parse: python parse_sweep_results.py logs/${TAG}-extended.log"
echo "================================================"
