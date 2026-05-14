#!/bin/bash
# TabICL native fine-tuning, o1->o2 direction.
# Mirrors run_icl_o1_to_o2.sh but turns on the new soda-inria FinetunedTabICL*
# path. The task's valid split is used for early stopping inside tabicl,
# so the reported `test_metric/<task>/<metric>` is the clean held-out score.
#
# REQUIRES on the runner:  pip install 'tabicl[finetune]'
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_icl_ft_o1_to_o2.sh \
#     2>&1 | tee logs/icl-ft-o1-to-o2.log
#
# Cost note: each eval task now triggers a full PyTorch fine-tune of TabICL
# (up to 50 epochs, gradient steps with AdamW). Expect ~minutes per task vs
# ~seconds for the zero-shot ICL path.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/icl-ft-o1-to-o2 icl-ft-o1-to-o2 \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 2000 \
    --icl_n_estimators 8 \
    --tabicl_checkpoint_version v2 \
    --tabicl_finetune \
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
    --savepath checkpoints/icl-ft-o1-to-o2
