#!/bin/bash
# Stage 1 with chain-of-thought scaffolding (--cot_prompt). Mirrors the
# o1-tth-lora-v3 setup but adds per-task reasoning hints between the
# question and 'Answer:' position.
#
# The LLM does NOT autoregressively generate reasoning text — the head is
# llm_mlp, so it just gets extra prompt positions to attend over before
# the MLP reads the answer-position hidden state.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_train_o1_cot.sh \
#     2>&1 | tee logs/o1-tth-lora-cot.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/o1-tth-lora-cot o1-tth-lora-cot \
    --head llm_mlp --llm_model Qwen/Qwen3-1.7B \
    --tasks others-1 --eval_tasks others-2 \
    --task_type_heads \
    --use_lora --lora_r 8 --lora_alpha 16 \
    --neighbor_tokens 8 \
    --cot_prompt \
    --loadpath checkpoints/downloaded/single-sft \
    --warmup_epochs 2 --maxepoch 15 --patience 5 \
    --batchsize 8 --lr 1e-4 --projector_lr 1e-3 \
    --downsample_num 5000 --pool_mode entity --eval_per_epoch 1 \
    --savepath checkpoints/o1-tth-lora-cot
