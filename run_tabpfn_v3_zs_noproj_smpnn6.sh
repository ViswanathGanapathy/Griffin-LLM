#!/bin/bash
# TabPFN v3 ZS, NO ICLProjection, SMPNN-6 Griffin backbone, o1->o2.
#
# Pairs with run_tabpfn_v3_zs_noproj_vanilla.sh. Tests whether the
# SMPNN-6 backbone's overfitting (seen in the previous projection-based
# test) was due to the projection+probe combo, or to the deeper backbone
# itself. By skipping projection AND probe, we test the SMPNN-6 features
# raw — no per-task adaptation, no compression.
#
# Usage:
#   ./run_tabpfn_v3_zs_noproj_smpnn6.sh 2>&1 | tee logs/tabpfn-zs-noproj-smpnn6.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-zs-noproj-smpnn6 tabpfn-zs-noproj-smpnn6 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/smpnn-depth-smpnn-6/best_checkpoint \
    --use_smpnn --num_mp 6 \
    --no_icl_projection \
    --probe_epochs 0 \
    --icl_n_estimators 8 \
    --icl_max_context 30000 \
    --tabpfn_version v3 \
    --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES": 50000, "MAX_NUMBER_OF_FEATURES": 600}' \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --use_rev True --use_gate True \
    --output_mlp_dim 1 --no_target_normalize \
    --savepath checkpoints/tabpfn-zs-noproj-smpnn6
