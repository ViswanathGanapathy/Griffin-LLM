#!/bin/bash
# Extended per-task FT sweep on others-2 from the o1-tth-lora-v3 checkpoint.
# Sweeps N ∈ {1024, 2048, 4096, 8192} × LR ∈ {3e-4, 1e-3} = 8 cells.
#
# Targets two findings from the basic sweep (avg -0.858, oracle -0.847):
#   - 3 tasks still climbing at N=4096 (site-success, study-adverse, telstra)
#     → push to N=8192
#   - study-outcome regressed with N>1024 → test lower LR=3e-4
#
# Per-task isolation: each task fine-tuned on its own N samples, tested
# on its own test split. All 6 others-2 tasks evaluated per cell.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_finetune_o2_extended.sh \
#     2>&1 | tee logs/o2-from-o1-extended.log
#   python parse_sweep_results.py logs/o2-from-o1-extended.log

CHECKPOINT="checkpoints/o1-tth-lora-v3/best_checkpoint"
MODEL="Qwen/Qwen3-1.7B"
DATASET="datasets/joint-v65"

echo "================================================"
echo "Extended FT sweep: N x LR on others-2"
echo "Checkpoint: $CHECKPOINT"
echo "================================================"

for LR in 3e-4 1e-3; do
  for N in 1024 2048 4096 8192; do
    TAG="o2-N${N}-lr${LR}"
    echo ""
    echo "=== N=${N} LR=${LR} ==="
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        $DATASET logs/$TAG $TAG \
        --mode test \
        --head llm_mlp --llm_model $MODEL \
        --tasks others-1 --eval_tasks others-2 \
        --task_type_heads --use_lora --lora_r 8 --lora_alpha 16 \
        --neighbor_tokens 8 \
        --loadpath $CHECKPOINT \
        --batchsize 4 --pool_mode entity \
        --finetune_samples $N --finetune_epochs 5 --finetune_lr $LR \
        --finetune_projector --finetune_lora
  done
done

echo ""
echo "================================================"
echo "Extended sweep complete!"
echo "Parse: python parse_sweep_results.py logs/o2-from-o1-extended.log"
echo "================================================"
