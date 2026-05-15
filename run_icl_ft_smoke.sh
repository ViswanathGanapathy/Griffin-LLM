#!/bin/bash
# FT smoke test — single regression task.
#
# Validates that the TabICL fine-tuning machinery works end-to-end without
# touching the classification path. If this run completes (FT progress bar
# advances past epoch 1), the FT pipeline is healthy and the airbnb-destination
# crash is classification-label specific.
#
# CUDA_LAUNCH_BLOCKING=1 turns CUDA errors synchronous so the traceback
# points at the real failing op (vs. a misleading later op).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_icl_ft_smoke.sh \
#     2>&1 | tee logs/icl-ft-smoke.log

export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-smoke icl-ft-smoke \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks rel-trial-site-success \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 2000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_finetune_epochs 5 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 3 \
    --tabicl_finetune_n_estimators_train 2 \
    --tabicl_finetune_n_estimators_validation 2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-smoke
