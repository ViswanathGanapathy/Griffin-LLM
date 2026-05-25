#!/bin/bash
# TabPFN v3 ZS, projection KEPT but probe_epochs=1 (instead of 5), o1->o2.
# Tests whether a short probe is benign — between the projection-overfitting
# (5 epochs, hurt cross-task transfer) and no-projection (best result so far).
# If probe_epochs=1 ties or beats no-projection, the issue isn't the
# projection per se but the duration of probe training.
#
# Usage:
#   ./run_tabpfn_v3_zs_proj_probe1_o1_to_o2.sh \
#     2>&1 | tee logs/tabpfn-zs-proj-probe1-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-zs-proj-probe1-o1-to-o2 tabpfn-zs-proj-probe1-o1-to-o2 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 1 \
    --icl_proj_dim 128 \
    --icl_max_context 30000 \
    --icl_n_estimators 8 \
    --tabpfn_version v3 \
    --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES": 50000}' \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 --no_target_normalize \
    --savepath checkpoints/tabpfn-zs-proj-probe1-o1-to-o2
