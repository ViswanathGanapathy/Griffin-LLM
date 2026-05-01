#!/bin/bash
# Stage 1 with pool_layers=4 + attention pooling on others-1 -> others-2.
# Mirrors the o1-tth-lora-v3 setup but adds two architectural levers:
#
#   --pool_layers 4    Learnable softmax-weighted sum of last 4 transformer
#                      layers' hidden states (vs default: just last layer).
#                      Adds 4 trainable parameters and ~3-5x more layer
#                      activation storage during forward.
#
#   --pool_mode attention   Token-axis pooling via a learned query vector
#                           over all sequence positions (vs default 'entity'
#                           which concats entity-pos hidden + last-pos).
#                           Output is [B, D] instead of [B, 2D] — head input
#                           dimension halves.
#
# Memory: bs dropped from 8 to 6 to accommodate the 4-layer activation store.
# Other hyperparameters identical to o1-tth-lora-v3 for clean comparison.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_o1_pool_attn.sh \
#     2>&1 | tee logs/o1-tth-lora-pa.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/o1-tth-lora-pa o1-tth-lora-pa \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads \
    --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --pool_mode attention --pool_layers 4 \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 6 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 \
    --eval_per_epoch 1 \
    --savepath checkpoints/o1-tth-lora-pa
