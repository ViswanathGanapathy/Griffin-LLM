#!/bin/bash
# Stage 2: per-task FT sweep on commerce-2 (or commerce-1), starting from
# the c1-tth-lora-v3 (or c2-tth-lora-v3) checkpoint produced by Stage 1.
#
# Per-task isolation: for each eval task → reset head + projector + LoRA
# to checkpoint state → fine-tune on N samples of that task → test on
# that task. Sweeps N ∈ {0, 1024, 2048, 4096}.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_commerce_tth_lora.sh c1_to_c2 \
#     2>&1 | tee logs/c2-from-c1-sweep.log
#   CUDA_VISIBLE_DEVICES=1 ./run_finetune_commerce_tth_lora.sh c2_to_c1 \
#     2>&1 | tee logs/c1-from-c2-sweep.log
#
# Then:  python parse_sweep_results.py logs/c2-from-c1-sweep.log

set -e
DIR=${1:-}

case "$DIR" in
    c1_to_c2)
        TRAIN=commerce-1
        EVAL=commerce-2
        CHECKPOINT="checkpoints/c1-tth-lora-v3/best_checkpoint"
        TAG=c2-from-c1
        ;;
    c2_to_c1)
        TRAIN=commerce-2
        EVAL=commerce-1
        CHECKPOINT="checkpoints/c2-tth-lora-v3/best_checkpoint"
        TAG=c1-from-c2
        ;;
    *)
        echo "Usage: $0 {c1_to_c2|c2_to_c1}"
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Stage 2: per-task FT on $EVAL (TTH + LoRA + neighbor_tokens=8)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

# Zero-shot baseline
echo ""
echo "=== N=0 (zero-shot) ==="
PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    $DATASET logs/${TAG}-0 ${TAG}-0 \
    --mode test \
    --head llm_mlp --llm_model $MODEL \
    --tasks $TRAIN --eval_tasks $EVAL \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --loadpath $CHECKPOINT \
    --batchsize 4 --pool_mode entity

# Per-task FT sweep
for N in 1024 2048 4096; do
    echo ""
    echo "=== N=$N samples/task ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/${TAG}-${N} ${TAG}-${N} \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks $TRAIN --eval_tasks $EVAL \
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
echo "Parse: python parse_sweep_results.py logs/${TAG}-sweep.log"
echo "================================================"
