#!/bin/bash
# TabPFN v3 fine-tune on the o1->o2 transfer.
# Same probe + extract pipeline as the ZS script; predicts via the
# KV-cached path after FT.
#
# Note: FinetunedTabPFNClassifier/Regressor traditionally hardcoded
# v2.5 internally — confirm whether v3 is supported via the FT API
# by running probe_tabpfn.py first. If FT is still v2.5-only, the
# fair comparison vs the ZS-v3 script becomes "v3 ZS vs v2.5 FT",
# which is still informative.
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
    --tabpfn_kv_cache \
    --tabpfn_finetune \
    --tabpfn_finetune_epochs 30 \
    --tabpfn_finetune_lr 2e-5 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-ft-o1-to-o2
