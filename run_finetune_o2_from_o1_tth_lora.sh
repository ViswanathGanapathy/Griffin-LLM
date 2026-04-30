#!/bin/bash
# Stage 2: per-task FT sweep on others-2 tasks, starting from the
# o1-tth-lora-v3 checkpoint (TaskTypeHeads + LoRA + neighbor tokens).
#
# For each N in {0, 1024, 2048, 4096}:
#   For each task in others-2:
#     - reset head + projector + LoRA to the checkpoint snapshot
#     - fine-tune on N samples of THAT task only
#     - test on THAT task only
#
# Tasks evaluated separately (per-task isolation by the inner FT loop):
#   binary:       rel-trial-study-outcome
#   regression:   rel-trial-site-success, rel-trial-study-adverse
#   multi-class:  airbnb-destination (K=12), talkingdata-demo-pred (K=12), telstra-severity (K=3)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_from_o1_tth_lora.sh 2>&1 | tee logs/o2-from-o1-sweep.log
#   python parse_sweep_results.py logs/o2-from-o1-sweep.log

CHECKPOINT="checkpoints/o1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Per-task FT sweep on others-2 (TTH + LoRA + neighbor_tokens=8)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline (no FT)
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/o2-ft-0 o2-ft-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# Per-task FT sweep
for N in 1024 2048 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/o2-ft-$N o2-ft-$N \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr 1e-3 \
        --finetune_projector --finetune_lora
done

echo ""
echo "================================================"
echo "Sweep complete!"
echo "Parse: python parse_sweep_results.py logs/o2-from-o1-sweep.log"
echo "================================================"
