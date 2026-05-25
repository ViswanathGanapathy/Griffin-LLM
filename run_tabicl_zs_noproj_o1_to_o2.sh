#!/bin/bash
# TabICL v2 ZS, NO projection, vanilla-4 Griffin backbone, o1->o2.
# Tests whether the no-projection win we saw for TabPFN also applies
# to TabICL. TabICL caps inference context at 10K (its practical limit),
# so the comparison is at lower context than the TabPFN runs but still
# isolates the projection's effect.
#
# Usage:
#   ./run_tabicl_zs_noproj_o1_to_o2.sh 2>&1 | tee logs/tabicl-zs-noproj-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabicl-zs-noproj-o1-to-o2 tabicl-zs-noproj-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --no_icl_projection \
    --probe_epochs 0 \
    --icl_n_estimators 8 \
    --icl_max_context 10000 \
    --tabicl_checkpoint_version v2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 --no_target_normalize \
    --savepath checkpoints/tabicl-zs-noproj-o1-to-o2
