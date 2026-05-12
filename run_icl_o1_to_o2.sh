#!/bin/bash
# TabICL evaluation using the o1-tth-lora-v3 checkpoint as Griffin backbone.
# For each task in others-2:
#   1. Extract Griffin + ICLProjection embeddings for train/valid/test splits
#   2. Subsample training context to <=10000 examples (stratified for clf,
#      random for regression)
#   3. Fit TabICL v2 on the context (no gradient training — just stores it)
#   4. Predict on valid/test embeddings via TabICL's internal ICL transformer
#   5. Report per-task valid/test metrics
#
# Two stages:
#   A) Probe training: 5 epochs of (Griffin + ICLProjection + LinearProbe)
#      on others-1 tasks so the projection learns useful 128-d features
#   B) Per-task ICL eval on others-2 using the trained projection
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_icl_o1_to_o2.sh \
#     2>&1 | tee logs/icl-o1-to-o2.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-o1-to-o2 icl-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 10000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-o1-to-o2
