#!/bin/bash
# TabICL evaluation in the o2→o1 direction.
#   Backbone: checkpoints/o2-to-o1-norm/best_checkpoint (trained on others-2
#             with --target_normalize).
#   Probe task: rel-trial-site-success (one regression task from others-2 —
#               keeps LinearProbe single-output-dim consistent with MSE).
#   Eval tasks: others-1 (all 6 tasks).
#
# Mirror of run_icl_o1_to_o2.sh in the other direction. Run on a separate
# GPU from the o1→o2 job:
#   CUDA_VISIBLE_DEVICES=1 ./run_icl_o2_to_o1.sh \
#     2>&1 | tee logs/icl-o2-to-o1.log

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-o2-to-o1 icl-o2-to-o1 \
    --head tabicl \
    --tasks rel-trial-site-success \
    --eval_tasks others-1 \
    --loadpath checkpoints/o2-to-o1-norm/best_checkpoint \
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
    --savepath checkpoints/icl-o2-to-o1
