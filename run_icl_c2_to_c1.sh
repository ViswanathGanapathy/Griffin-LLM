#!/bin/bash
# TabICL evaluation in the c2→c1 direction.
#   Backbone: checkpoints/c2-to-c1-norm/best_checkpoint (trained on commerce-2
#             with --target_normalize).
#   Probe task: rel-avito-ad-ctr (regression task from commerce-2 — keeps
#               LinearProbe single-output-dim consistent with MSE).
#   Eval tasks: commerce-1 (diginetica-downsample-ctr, rel-hm-item-sales,
#               rel-hm-user-churn, retailrocket-cvr, seznam-charge,
#               seznam-prepay).
#
# Mirror of run_icl_c1_to_c2.sh in the reverse commerce direction.
#   CUDA_VISIBLE_DEVICES=3 ./run_icl_c2_to_c1.sh \
#     2>&1 | tee logs/icl-c2-to-c1.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-c2-to-c1 icl-c2-to-c1 \
    --head tabicl \
    --tasks rel-avito-ad-ctr \
    --eval_tasks commerce-1 \
    --loadpath checkpoints/c2-to-c1-norm/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 10000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --target_normalize --target_stats_samples 200 \
    --savepath checkpoints/icl-c2-to-c1
