#!/bin/bash
# TabICL FT, o1->o2, GPU 0 — half of the others-2 eval tasks.
#
# Pairs with run_icl_ft_gpu1_o1_to_o2.sh. Each script runs its own
# probe phase + half the eval set, so total wall time is ~halved vs
# the single-GPU FT script.
#
# Why split tasks instead of running multi-GPU FT on one task at a
# time: tabicl's FT loop is not single-process multi-GPU friendly
# (validation/predict crash on bare "cuda"). True multi-GPU FT would
# require `accelerate launch --multi_gpu`, which is a bigger refactor.
#
# Task split rationale (mix of FT and ZS-fallback to balance load):
#   GPU 0: rel-trial-site-success, rel-trial-study-adverse, talkingdata-demo-pred
#   GPU 1: airbnb-destination, rel-trial-study-outcome, telstra-severity
#
# REQUIRES: pip install 'tabicl[finetune]'
#
# Usage:
#   ./run_icl_ft_gpu0_o1_to_o2.sh 2>&1 | tee logs/icl-ft-gpu0-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=0

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-gpu0-o1-to-o2 icl-ft-gpu0-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks rel-trial-site-success rel-trial-study-adverse talkingdata-demo-pred \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 2000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_finetune_epochs 50 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 10 \
    --tabicl_finetune_n_estimators_train 2 \
    --tabicl_finetune_n_estimators_validation 2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-gpu0-o1-to-o2
