#!/bin/bash
# SMPNN-Griffin depth-scaling experiment with NATIVE head.
#
# Compares vanilla Griffin (num_mp=4) vs SMPNN-Griffin at num_mp = {4, 6, 8}
# on the others-1 task group, training from scratch (no warm-start).
# Tests the core SMPNN claim: deeper GNNs without oversmoothing.
#
# Each run produces a checkpoint that can be re-used downstream
# (probe + TabPFN ZS) to test whether deeper backbones produce better
# embeddings for ICL.
#
# Usage:
#   ./run_smpnn_depth_native.sh 2>&1 | tee logs/smpnn-depth.log
#
# Run-time estimate per variant: ~2-4 hours on A100 for 20 epochs on
# all 6 others-1 tasks. Total: ~12 hours on a single GPU. Split across
# two GPUs by running variants in parallel.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_one() {
    local TAG=$1
    local FLAGS=$2
    echo ""
    echo "================================================"
    echo ">>> Training $TAG"
    echo "================================================"
    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-depth-${TAG} smpnn-depth-${TAG} \
        --tasks others-1 \
        $FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath checkpoints/smpnn-depth-${TAG}
}

# 1. Vanilla baseline at num_mp=4 (matches existing Griffin)
run_one "vanilla-4" "--num_mp 4"

# 2. SMPNN parity at num_mp=4 (should match #1)
run_one "smpnn-4" "--num_mp 4 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"

# 3. SMPNN deeper: num_mp=6 (the main experiment)
run_one "smpnn-6" "--num_mp 6 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"

# 4. SMPNN depth ceiling: num_mp=8
run_one "smpnn-8" "--num_mp 8 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"

echo ""
echo "================================================"
echo "Done. Compare final val/test metrics across:"
echo "  logs/smpnn-depth-vanilla-4/"
echo "  logs/smpnn-depth-smpnn-4/"
echo "  logs/smpnn-depth-smpnn-6/"
echo "  logs/smpnn-depth-smpnn-8/"
echo "================================================"
