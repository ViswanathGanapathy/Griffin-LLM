#!/bin/bash
# TabPFN v3 zero-shot ICL on the o1->o2 transfer.
# Uses KV caching to support a large inference context (30K) without
# recomputing the context K/V on every predict call.
#
# Same probe pipeline as the tabicl scripts: 5 epochs of Griffin +
# ICLProjection + LinearProbe on rel-f1-driver-position, then ZS
# eval on others-2.
#
# REQUIRES on the runner:
#   pip install --upgrade tabpfn         # latest stable
#   (or `pip install tabpfn --pre` if v3 is in pre-release)
#
# Usage:
#   ./run_tabpfn_v3_zs_o1_to_o2.sh 2>&1 | tee logs/tabpfn-v3-zs-o1-to-o2.log

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
    --tabpfn_kv_cache \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-zs-o1-to-o2
