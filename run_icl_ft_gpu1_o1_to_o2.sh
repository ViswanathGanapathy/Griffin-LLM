#!/bin/bash
# TabICL FT, o1->o2, GPU 1 — other half of the others-2 eval tasks.
#
# Pairs with run_icl_ft_gpu0_o1_to_o2.sh. See that file's header for
# the rationale.
#
# Task split:
#   GPU 0: rel-trial-site-success, rel-trial-study-adverse, talkingdata-demo-pred
#   GPU 1: airbnb-destination, rel-trial-study-outcome, telstra-severity
#
# REQUIRES: pip install 'tabicl[finetune]'
#
# Usage (run on the second terminal, in parallel with the GPU 0 script):
#   ./run_icl_ft_gpu1_o1_to_o2.sh 2>&1 | tee logs/icl-ft-gpu1-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=1

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-gpu1-o1-to-o2 icl-ft-gpu1-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks airbnb-destination rel-trial-study-outcome telstra-severity \
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
    --savepath checkpoints/icl-ft-gpu1-o1-to-o2
