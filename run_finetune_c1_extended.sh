#!/bin/bash
# Extended per-task FT sweep on commerce-1 from c2-tth-lora-v3 checkpoint.
# Sweeps N x LR to recover stuck/regressing tasks from the basic sweep:
#   - diginetica-downsample-ctr (best at N=0; FT hurt) -> try low LR
#   - retailrocket-cvr (collapsed at N=4096) -> try low LR + N=2048
#   - seznam-charge / seznam-prepay (still climbing) -> try N=8192
#   - rel-hm-item-sales (climbing regression) -> try N=8192
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_c1_extended.sh \
#     2>&1 | tee logs/c1-from-c2-extended.log
#   python parse_sweep_results.py logs/c1-from-c2-extended.log

CHECKPOINT="checkpoints/c2-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Extended FT: c2 -> c1 (N x LR grid)"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-3; do
  for N in 2048 4096 8192; do
    TAG="c1-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks commerce-2 --eval_tasks commerce-1 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "Done. Parse: python parse_sweep_results.py logs/c1-from-c2-extended.log"
