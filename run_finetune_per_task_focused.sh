#!/bin/bash
# Per-task focused grid wrapper — run a 3D LR x epochs x N grid for a single
# task. Use this for tasks where the shared sweep wasn't enough.
#
# Pattern matches what successfully recovered rel-trial-study-outcome
# (0.5741 -> 0.6563 with focused grid).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_per_task_focused.sh \
#       <checkpoint_path> <task_name> <log_tag> \
#       2>&1 | tee logs/<log_tag>.log
#
# Examples:
#   ./run_finetune_per_task_focused.sh \
#       checkpoints/c1-tth-lora-v3/best_checkpoint \
#       amazon-churn \
#       amazon-churn-focused
#
#   ./run_finetune_per_task_focused.sh \
#       checkpoints/c1-tth-lora-v3/best_checkpoint \
#       outbrain-small-ctr \
#       outbrain-focused
#
# Note: passes --tasks commerce-1 (parent set the checkpoint was trained on).
# For an o1 checkpoint, edit --tasks to others-1 below.

set -e
CHECKPOINT="${1:-}"
TASK="${2:-}"
TAG="${3:-}"

if [ -z "$CHECKPOINT" ] || [ -z "$TASK" ] || [ -z "$TAG" ]; then
    echo "Usage: $0 <checkpoint_path> <task_name> <log_tag>"
    exit 1
fi

# Auto-detect parent task set from the checkpoint name (rough heuristic)
if [[ "$CHECKPOINT" == *"c1-"* || "$CHECKPOINT" == *"commerce-1"* ]]; then
    PARENT_TASKS=commerce-1
elif [[ "$CHECKPOINT" == *"c2-"* || "$CHECKPOINT" == *"commerce-2"* ]]; then
    PARENT_TASKS=commerce-2
elif [[ "$CHECKPOINT" == *"o1-"* || "$CHECKPOINT" == *"others-1"* ]]; then
    PARENT_TASKS=others-1
elif [[ "$CHECKPOINT" == *"o2-"* || "$CHECKPOINT" == *"others-2"* ]]; then
    PARENT_TASKS=others-2
else
    PARENT_TASKS=others-1   # fallback
fi

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Focused grid: $TASK (parent: $PARENT_TASKS)"
echo "Checkpoint: $CHECKPOINT"
echo "Tag: $TAG"
echo "================================================"

for LR in 3e-4 1e-4; do
  for E in 5 10 15; do
    for N in 2048 4096 8192; do
      RUN_TAG="${TAG}-N${N}-lr${LR}-ep${E}"
      echo ""
      echo "=== N=${N} LR=${LR} epochs=${E} ==="
      PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
          $DATASET logs/$RUN_TAG $RUN_TAG \
          --mode test \
          --head llm_mlp --llm_model $MODEL \
          --tasks $PARENT_TASKS --eval_tasks $TASK \
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
echo "Done. Parse: python parse_sweep_results.py logs/${TAG}.log"
