#!/bin/bash
# Stage 2 — per-task FT sweep on others-2 from o1-to-o2-norm checkpoint.
# Uses the same N x LR grid as the v3 extended sweep so results are
# directly comparable to logs/o2-from-o1-extended.log.
#
# Pass the same mode arg as the training script.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_from_o1_norm.sh        # FT on norm checkpoint
#   CUDA_VISIBLE_DEVICES=1 ./run_finetune_o2_from_o1_norm.sh huber-norm

MODE=${1:-norm}
case "$MODE" in
    norm)
        CHECKPOINT="checkpoints/o1-to-o2-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="o2-from-o1-norm"
        ;;
    noflag|raw)
        CHECKPOINT="checkpoints/o1-to-o2-raw/best_checkpoint"
        EXTRA_FLAGS="--no_target_normalize"
        TAG="o2-from-o1-raw"
        ;;
    huber-norm)
        CHECKPOINT="checkpoints/o1-to-o2-huber-norm/best_checkpoint"
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="o2-from-o1-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on others-2 ($MODE)"
echo "Checkpoint: $CHECKPOINT"
echo "Compare against: logs/o2-from-o1-extended.log (v3 baseline oracle -0.7793)"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/${TAG}-0 ${TAG}-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# N x LR grid — mirrors v3 extended sweep + adds 3e-5 for regression
for LR in 3e-5 3e-4 1e-3; do
  for N in 1024 2048 4096 8192; do
    SUBTAG="${TAG}-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$SUBTAG $SUBTAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
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
echo "Compare:"
echo "  python parse_sweep_results.py logs/o2-from-o1-extended.log logs/${TAG}-extended.log"
echo "================================================"
