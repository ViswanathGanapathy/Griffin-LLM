#!/bin/bash
# TabPFN v3 zero-shot ICL on the o1->o2 transfer.
# Uses fit_mode=fit_with_cache (the v3 KV-cache path) to make predict()
# fast at large context. inference_precision=bfloat16 halves memory.
#
# Confirmed via probe_tabpfn.py: tabpfn 8.0.3 has ModelVersion.V3 and
# accepts fit_mode + memory_saving_mode + inference_precision +
# inference_config kwargs.
#
# REQUIRES on the runner:
#   pip install --upgrade tabpfn    (already 8.0.3)
#
# Usage:
#   ./run_tabpfn_v3_zs_o1_to_o2.sh 2>&1 | tee logs/tabpfn-v3-zs-o1-to-o2.log
#
# If fit_with_cache isn't a valid fit_mode in your tabpfn release, you'll
# see a clear error from tabpfn; check `python3 probe_tabpfn_inference.py`
# for the literal-typed valid values.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-v3-zs-o1-to-o2 tabpfn-v3-zs-o1-to-o2 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 30000 \
    --icl_n_estimators 8 \
    --tabpfn_version v3 \
    --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision bfloat16 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-zs-o1-to-o2
