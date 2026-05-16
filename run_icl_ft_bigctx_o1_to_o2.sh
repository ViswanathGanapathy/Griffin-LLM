#!/bin/bash
# TabICL FT, o1->o2.
# Strategy: small FT-step context (cheap gradient steps) + large inference
# context (rich support set at predict time). Uses tabicl's max_data_size
# knob to bound FT-step memory while passing the full 10K-row context to
# .fit() so predict uses the full context.
#
# Why not freeze_col: the freeze_col=True path inside tabicl appears to
# reset the inference_mgr's exe_device to bare "cuda" mid-FT, breaking
# subsequent mem_get_info calls. Without freeze_col, FT runs cleanly.
#
# Knobs (verified via probe_tabicl_ft.py):
#   --icl_max_context 10000                        -> X_ctx passed to fit()
#   --tabicl_finetune_max_data_size 2000           -> FT-step subsample cap
#   --tabicl_finetune_n_estimators_train 1         -> half activations per step
#   --tabicl_finetune_n_estimators_validation 1
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
    --icl_max_context 10000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_finetune_epochs 50 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 10 \
    --tabicl_finetune_n_estimators_train 1 \
    --tabicl_finetune_n_estimators_validation 1 \
    --tabicl_finetune_max_data_size 2000 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-bigctx-o1-to-o2
