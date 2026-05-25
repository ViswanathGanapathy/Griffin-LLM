#!/bin/bash
# TabPFN v3 ZS, NO ICLProjection, vanilla-4 Griffin backbone, o1->o2.
#
# Tests whether the 512->128 projection is helping or hurting cross-task
# transfer. This run uses the raw 512-d Griffin embeddings as TabPFN's
# "tabular features" directly.
#
# Differences from the existing run_tabpfn_v3_zs_o1_to_o2.sh:
#   --no_icl_projection                : bypass the projection
#   --probe_epochs 0                    : skip probe (nothing to train)
#   --tabpfn_inference_config          : bump MAX_NUMBER_OF_FEATURES
#                                          to accommodate 512-d input
#
# Usage:
#   ./run_tabpfn_v3_zs_noproj_vanilla.sh 2>&1 | tee logs/tabpfn-zs-noproj-vanilla.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-zs-noproj-vanilla tabpfn-zs-noproj-vanilla \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
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
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 --no_target_normalize \
    --savepath checkpoints/tabpfn-zs-noproj-vanilla
