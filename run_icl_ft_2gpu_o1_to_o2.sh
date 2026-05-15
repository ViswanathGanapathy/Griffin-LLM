#!/bin/bash
# TabICL FT, o1->o2 direction, 2-GPU setup.
#   - Uses BOTH GPUs visible to the process. tabicl's FT loop shards
#     attention activations across GPUs so we can run a larger context.
#   - Probe phase (Griffin + ICLProjection + LinearProbe) still runs on
#     a single GPU via accelerator's default device pinning; only the
#     FT call into tabicl uses both GPUs.
#   - Context bumped to 8000 (was 2000 on the single-GPU script that
#     barely beat ZS).
#
# REQUIRES on the runner:
#   pip install 'tabicl[finetune]'
#
# Usage (no arg needed; CUDA_VISIBLE_DEVICES is set inside the script):
#   ./run_icl_ft_2gpu_o1_to_o2.sh 2>&1 | tee logs/icl-ft-2gpu-o1-to-o2.log

# Expose both GPUs to the process. Inside the process they'll be
# cuda:0 and cuda:1.
export CUDA_VISIBLE_DEVICES=0,1

# Synchronous CUDA so the first failing batch (if any) gives a clean
# traceback. Drop this for production runs.
export CUDA_LAUNCH_BLOCKING=1

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-2gpu-o1-to-o2 icl-ft-2gpu-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 8000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
    --tabicl_multi_gpu \
    --tabicl_finetune_epochs 50 \
    --tabicl_finetune_lr 1e-5 \
    --tabicl_finetune_patience 10 \
    --tabicl_finetune_n_estimators_train 2 \
    --tabicl_finetune_n_estimators_validation 2 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/icl-ft-2gpu-o1-to-o2
