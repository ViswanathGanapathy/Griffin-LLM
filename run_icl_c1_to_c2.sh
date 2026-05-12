#!/bin/bash
# TabICL evaluation in the c1→c2 direction.
#   Backbone: checkpoints/c1-to-c2-norm/best_checkpoint (trained on commerce-1
#             with --target_normalize).
#   Probe task: rel-hm-item-sales (the one regression task in commerce-1 —
#               keeps LinearProbe single-output-dim consistent with MSE).
#   Eval tasks: commerce-2 (amazon-churn, amazon-rating, outbrain-small-ctr,
#               rel-avito-ad-ctr, rel-avito-user-clicks, rel-avito-user-visits).
#
# Mirror of run_icl_o1_to_o2.sh in the commerce direction. Run on a free GPU:
#   CUDA_VISIBLE_DEVICES=2 ./run_icl_c1_to_c2.sh \
#     2>&1 | tee logs/icl-c1-to-c2.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-c1-to-c2 icl-c1-to-c2 \
    --head tabicl \
    --tasks rel-hm-item-sales \
    --eval_tasks commerce-2 \
    --loadpath checkpoints/c1-to-c2-norm/best_checkpoint \
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
    --savepath checkpoints/icl-c1-to-c2
