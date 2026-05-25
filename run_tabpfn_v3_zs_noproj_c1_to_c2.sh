#!/bin/bash
# TabPFN v3 ZS, NO projection, vanilla-4 Griffin backbone (commerce-1 pretrained),
# eval on commerce-2. Validates the no-proj win in the c1->c2 direction.
#
# Usage:
#   ./run_tabpfn_v3_zs_noproj_c1_to_c2.sh 2>&1 | tee logs/tabpfn-zs-noproj-c1-to-c2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-zs-noproj-c1-to-c2 tabpfn-zs-noproj-c1-to-c2 \
    --head tabpfn \
    --tasks rel-hm-item-sales \
    --eval_tasks commerce-2 \
    --loadpath checkpoints/c1-to-c2-norm/best_checkpoint \
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
    --output_mlp_dim 1 --target_normalize --target_stats_samples 200 \
    --savepath checkpoints/tabpfn-zs-noproj-c1-to-c2
