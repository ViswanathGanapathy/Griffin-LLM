#!/bin/bash
# Joint multi-task training across all 4 splits (24 tasks).
#
# Differences from the o1-tth-lora-v3 transfer setup:
#   - --tasks joint-all-4 (24 tasks union of c1+c2+o1+o2)
#   - --eval_tasks joint-all-4 (in-distribution eval per epoch)
#   - --downsample_num bumped from 5000 -> 10000 (let larger datasets contribute more)
#   - --maxepoch 12 (down from 15) since each epoch is now ~4x larger
#   - Other hyperparameters identical to o1-tth-lora-v3
#
# Per-task multi-class head finally trains on real signal (5 multi-class
# tasks across the 4 splits: airbnb K=12, talkingdata K=12, telstra K=3,
# seznam-charge K=8, seznam-prepay K=8).
#
# Runtime estimate: ~24-30h on H100. Use tmux.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_joint_all4.sh \
#     2>&1 | tee logs/joint-all4-tth-lora.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/joint-all4-tth-lora joint-all4-tth-lora \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks joint-all-4 --eval_tasks joint-all-4 \
    --task_type_heads \
    --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 12 --patience 4 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 10000 \
    --pool_mode entity \
    --eval_per_epoch 1 \
    --savepath checkpoints/joint-all4-tth-lora
