#!/bin/bash
# TabPFN v3 zero-shot ICL, NO KV cache (baseline for the cache A/B).
# Identical to run_tabpfn_v3_zs_o1_to_o2.sh except fit_mode is the
# default 'fit_preprocessors' (no cache). Each predict() call recomputes
# the full forward pass over the context.
#
# Compare the [TIMING] lines in this log vs the cached log to measure
# the KV-cache speedup at large context.
#
# Activate the dedicated TabPFN conda env first:
#   conda activate griffin-tabpfn
#
# Usage:
#   ./run_tabpfn_v3_zs_nocache_o1_to_o2.sh \
#     2>&1 | tee logs/tabpfn-v3-zs-nocache-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-v3-zs-nocache-o1-to-o2 tabpfn-v3-zs-nocache-o1-to-o2 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 30000 \
    --icl_n_estimators 8 \
    --tabpfn_version v3 \
    --tabpfn_fit_mode fit_preprocessors \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES": 50000}' \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-zs-nocache-o1-to-o2
