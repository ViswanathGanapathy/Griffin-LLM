#!/bin/bash
# Cross-task transfer: train the 3 anchor backbones (C1, D1, D3) on each
# of the 3 source families we don't yet have backbones for: others-2,
# commerce-1, commerce-2. Together with the existing others-1 backbones,
# this gives us the full source-family coverage for the 4-direction
# transfer matrix (o1->o2, o2->o1, c1->c2, c2->c1).
#
# Total: 3 backbones x 3 source families = 9 trainings.
# Wall time: ~3-4 hours per training on A100 -> ~30 hours total at 1 GPU.
# Trivially parallelisable across GPUs via CUDA_VISIBLE_DEVICES + RUN_ONLY.
#
# Output: checkpoints/smpnn-xtask-<source>-<backbone>/best_checkpoint
#         e.g. checkpoints/smpnn-xtask-o2-d1-smpnn-6/best_checkpoint
#
# Usage:
#   ./run_smpnn_xtask_backbones.sh 2>&1 | tee logs/smpnn-xtask-backbones.log
#
# Or split across GPUs:
#   RUN_ONLY="o2-c1-vanilla-4 o2-d1-smpnn-6 o2-d3-alpha-1e-2" \
#     CUDA_VISIBLE_DEVICES=0 ./run_smpnn_xtask_backbones.sh ...
#   RUN_ONLY="c1-c1-vanilla-4 c1-d1-smpnn-6 c1-d3-alpha-1e-2" \
#     CUDA_VISIBLE_DEVICES=1 ./run_smpnn_xtask_backbones.sh ...

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
RUN_ONLY=${RUN_ONLY:-all}

run_one() {
    local TAG=$1
    local SRC_FAMILY=$2
    local FLAGS=$3
    if [ "$RUN_ONLY" != "all" ]; then
        local found=0
        for t in $RUN_ONLY; do
            if [ "$t" = "$TAG" ]; then found=1; break; fi
        done
        if [ "$found" = "0" ]; then
            echo ">>> Skipping $TAG (not in RUN_ONLY='$RUN_ONLY')"
            return
        fi
    fi
    echo ""
    echo "================================================"
    echo ">>> Training smpnn-xtask-${TAG}"
    echo "    source family: $SRC_FAMILY"
    echo "    flags:         $FLAGS"
    echo "================================================"
    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-xtask-${TAG} smpnn-xtask-${TAG} \
        --tasks ${SRC_FAMILY} \
        $FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath checkpoints/smpnn-xtask-${TAG}
}

# others-2 source family
run_one "o2-c1-vanilla-4"   "others-2" "--num_mp 4"
run_one "o2-d1-smpnn-6"     "others-2" "--num_mp 6 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"
run_one "o2-d3-alpha-1e-2"  "others-2" "--num_mp 6 --use_smpnn --alpha_init 1e-2 --log_alpha_every 2"

# commerce-1 source family
run_one "c1-c1-vanilla-4"   "commerce-1" "--num_mp 4"
run_one "c1-d1-smpnn-6"     "commerce-1" "--num_mp 6 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"
run_one "c1-d3-alpha-1e-2"  "commerce-1" "--num_mp 6 --use_smpnn --alpha_init 1e-2 --log_alpha_every 2"

# commerce-2 source family
run_one "c2-c1-vanilla-4"   "commerce-2" "--num_mp 4"
run_one "c2-d1-smpnn-6"     "commerce-2" "--num_mp 6 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"
run_one "c2-d3-alpha-1e-2"  "commerce-2" "--num_mp 6 --use_smpnn --alpha_init 1e-2 --log_alpha_every 2"

echo ""
echo "================================================"
echo "All 9 cross-task source backbones trained."
echo "Next step: ./run_smpnn_xtask_eval.sh"
echo "================================================"
