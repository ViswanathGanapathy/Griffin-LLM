#!/bin/bash
# Stage 1: train on others-2, validate on others-1, with --target_normalize.
#
# Why this direction is interesting:
#   - others-2 regression tasks are 0-1 fractional (rel-trial-site-success,
#     rel-trial-study-adverse) — heavy skew toward 0
#   - others-1 regression task (rel-f1-driver-position) has range 1-22, mean~10
#   - This is the largest source/target regression-scale gap in the codebase
#   - Without --target_normalize, ZS MAE on driver-position would be ~10 (terrible)
#     because the regression head outputs values in [0, 1] range
#   - With --target_normalize, training learns normalized targets and
#     denormalizes at inference -> predictions land on the correct scale
#
# Pass `noflag` as arg to disable normalization (control / ablation):
#   ./run_train_o2_to_o1_norm.sh        # with normalization (Cell C)
#   ./run_train_o2_to_o1_norm.sh noflag # without (Cell A control)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_o2_to_o1_norm.sh \
#     2>&1 | tee logs/o2-to-o1-norm.log

MODE=${1:-norm}
case "$MODE" in
    norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200"
        TAG="o2-to-o1-norm"
        ;;
    noflag|raw)
        EXTRA_FLAGS="--no_target_normalize"
        TAG="o2-to-o1-raw"
        ;;
    huber-norm)
        EXTRA_FLAGS="--target_normalize --target_stats_samples 200 --huber_loss --huber_delta 1.0"
        TAG="o2-to-o1-huber-norm"
        ;;
    *)
        echo "Usage: $0 {norm|noflag|huber-norm}"
        echo "  norm        - target_normalize on (default)"
        echo "  noflag      - no normalization (control / Cell A)"
        echo "  huber-norm  - target_normalize + huber_loss (Cell D)"
        exit 1
        ;;
esac

echo "================================================"
echo "Stage 1: others-2 -> others-1 ($MODE)"
echo "Tag: $TAG"
echo "Extra flags: $EXTRA_FLAGS"
echo "================================================"

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/$TAG $TAG \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks others-2 --eval_tasks others-1 \
    --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    $EXTRA_FLAGS \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 --pool_mode entity --eval_per_epoch 1 \
    --savepath checkpoints/$TAG
