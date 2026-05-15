#!/bin/bash
# TabICL FT, o1->o2, Path A: single-GPU + memory optimisations to fit a
# larger context (target 4-5K, up from the 2K cap on the previous run).
#
# Optimisations applied:
#   1. --tabicl_finetune_n_estimators_train 1   (was 2)  -> ~half activations
#   2. --tabicl_finetune_n_estimators_validation 1       -> smaller val pass
#   3. --tabicl_finetune_mixed_precision bf16            -> ~half memory
#   4. --tabicl_finetune_grad_checkpointing              -> trades compute for memory
#
# All four are filtered against the installed tabicl signature; any that
# the installed release doesn't accept are silently dropped (with a warning
# in the log). Run probe_tabicl_ft.py first to see which are actually
# supported.
#
# Context is initially 4000 — bump to 5000-6000 if you see plenty of
# headroom in nvidia-smi after the first task. Lower to 3000 if it OOMs.
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
    --icl_max_context 4000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_finetune_epochs 50 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 10 \
    --tabicl_finetune_n_estimators_train 1 \
    --tabicl_finetune_n_estimators_validation 1 \
    --tabicl_finetune_mixed_precision bf16 \
    --tabicl_finetune_grad_checkpointing \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-bigctx-o1-to-o2
