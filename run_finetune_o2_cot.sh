#!/bin/bash
# Stage 2 — per-task FT sweep on others-2 from the CoT-trained checkpoint.
# Same N x LR grid as run_finetune_o2_extended.sh but with --cot_prompt set
# (must match the stage-1 prompt template the checkpoint was trained on).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_cot.sh \
#     2>&1 | tee logs/o2-from-o1-cot-sweep.log

CHECKPOINT="checkpoints/o1-tth-lora-cot/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "CoT Stage 2: per-task FT on others-2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-3; do
  for N in 1024 2048 4096; do
    TAG="o2-cot-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --cot_prompt \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "Parse: python parse_sweep_results.py logs/o2-from-o1-cot-sweep.log"
