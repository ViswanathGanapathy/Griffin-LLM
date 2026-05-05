#!/bin/bash
# Stage 2: per-task FT sweep on others-1 from the o2-to-o1-* checkpoint.
# Sweeps N x LR grid (3 N x 2 LR = 6 cells) plus N=0 baseline.
#
# IMPORTANT: must pass the SAME mode arg as stage 1 — checkpoint and runtime
# must match on whether --target_normalize was used.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o1_from_o2_norm.sh        # FT for norm checkpoint
#   CUDA_VISIBLE_DEVICES=1 ./run_finetune_o1_from_o2_norm.sh noflag # FT for raw control
#   CUDA_VISIBLE_DEVICES=2 ./run_finetune_o1_from_o2_norm.sh huber-norm

MODE=${1:-norm}
case "$MODE" in
    norm)
        CHECKPOINT="checkpoints/o2-to-o1-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="o1-from-o2-norm"
        ;;
    noflag|raw)
        CHECKPOINT="checkpoints/o2-to-o1-raw/best_checkpoint"
        EXTRA_FLAGS="--no_target_normalize"
        TAG="o1-from-o2-raw"
        ;;
    huber-norm)
        CHECKPOINT="checkpoints/o2-to-o1-huber-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="o1-from-o2-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on others-1 ($MODE)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/${TAG}-0 ${TAG}-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-2 --eval_tasks others-1 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# N x LR grid
for LR in 3e-4 1e-3; do
  for N in 1024 2048 4096; do
    SUBTAG="${TAG}-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$SUBTAG $SUBTAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-2 --eval_tasks others-1 \
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
