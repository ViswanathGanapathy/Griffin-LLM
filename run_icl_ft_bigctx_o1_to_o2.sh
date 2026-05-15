#!/bin/bash
# TabICL FT, o1->o2, Path A: single-GPU + freeze the column embedder so a
# much larger context fits. tabicl already runs AMP (mixed precision) by
# default — that's NOT a knob we control. The real lever for memory is
# freezing the early heavy stage.
#
# Knobs we use (verified via probe_tabicl_ft.py):
#   1. --tabicl_finetune_freeze_col  -> no backward through the 12-layer col
#                                       embedder; biggest single memory win
#   2. --tabicl_finetune_n_estimators_train 1   (was 2)  -> ~half activations
#   3. --tabicl_finetune_n_estimators_validation 1
#
# Knobs from the previous version of this script that were DROPPED:
#   --tabicl_finetune_mixed_precision  (tabicl auto-enables AMP)
#   --tabicl_finetune_grad_checkpointing  (not exposed by tabicl)
#   --tabicl_finetune_batch_size  (not exposed by tabicl)
#
# Context: 6000 — freezing col_embedder gives us a lot of headroom over the
# 2K cap of the previous run.
#
# REQUIRES: pip install 'tabicl[finetune]'
#
# Usage:
#   ./run_icl_ft_bigctx_o1_to_o2.sh 2>&1 | tee logs/icl-ft-bigctx-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-bigctx-o1-to-o2 icl-ft-bigctx-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 6000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_finetune_epochs 50 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 10 \
    --tabicl_finetune_n_estimators_train 1 \
    --tabicl_finetune_n_estimators_validation 1 \
    --tabicl_finetune_freeze_col \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-bigctx-o1-to-o2
