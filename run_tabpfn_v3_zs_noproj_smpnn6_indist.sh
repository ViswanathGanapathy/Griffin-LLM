#!/bin/bash
# TabPFN v3 ZS, NO projection, SMPNN-6 Griffin backbone, IN-DISTRIBUTION
# evaluation on others-1.
#
# Closes the question "is SMPNN-6 actually useful as a backbone?" by
# testing it in its native setting (the same task family it was trained
# on), with the no-proj recipe that worked best for cross-task transfer.
#
# Compare against:
#   - SMPNN-6 native head, others-1 in-distribution (from depth experiment)
#     Average test metric: 0.5572
#   - Vanilla-4 native head, others-1 in-distribution (from depth experiment)
#     Average test metric: 0.5476
#
# Usage:
#   ./run_tabpfn_v3_zs_noproj_smpnn6_indist.sh \
#     2>&1 | tee logs/tabpfn-zs-noproj-smpnn6-indist.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-zs-noproj-smpnn6-indist tabpfn-zs-noproj-smpnn6-indist \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-1 \
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
    --savepath checkpoints/tabpfn-zs-noproj-smpnn6-indist
