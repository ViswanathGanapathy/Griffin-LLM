#!/bin/bash
# TabPFN v3 fine-tune on the o1->o2 transfer.
#
# Important note about TabPFN's FT class (verified via probe_tabpfn.py
# on tabpfn 8.0.3): the FT class has DIFFERENT knob names than the
# base TabPFN class. Notably:
#   - n_finetune_ctx_plus_query_samples = 10000  (default; FT context size)
#   - n_inference_subsample_samples     = 50000  (default; inference context)
#   - n_estimators_finetune = 2, n_estimators_validation = 2,
#     n_estimators_final_inference = 8
# This already implements the "small FT, big inference" pattern natively.
#
# Our TabPFNHead wraps FinetunedTabPFN{Classifier,Regressor} via the
# existing --tabpfn_finetune path; the new fit_mode / inference_precision
# CLI flags are no-ops for the FT class (they're filtered out — FT class
# doesn't accept them) but kept here for documentation.
#
# REQUIRES:
#   pip install --upgrade tabpfn
#
# Usage:
#   ./run_tabpfn_v3_ft_o1_to_o2.sh 2>&1 | tee logs/tabpfn-v3-ft-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-v3-ft-o1-to-o2 tabpfn-v3-ft-o1-to-o2 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 30000 \
    --icl_n_estimators 8 \
    --tabpfn_version v3 \
    --tabpfn_finetune \
    --tabpfn_finetune_epochs 30 \
    --tabpfn_finetune_lr 1e-5 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-ft-o1-to-o2
