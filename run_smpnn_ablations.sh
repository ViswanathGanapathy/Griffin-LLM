#!/bin/bash
# SMPNN-Griffin ablation suite (Studies A, B, C, D from the FM4SD plan).
#
# All runs:
#   - in-distribution: train on others-1, eval on others-1 temporal splits
#   - hiddim 512, hop 2, fanout 20, fewshotfanout 3, batchsize 256
#   - 20 epochs, AdamW lr 3e-4 wd 4e-4, no warm-start (from scratch)
#   - log alpha every 2 epochs (where applicable) for Figure 2 data
#
# Skipped (already in hand from run_smpnn_depth_native.sh):
#   - C1 vanilla-4, C2 SMPNN-4, C4 SMPNN-6, C5 SMPNN-8
#
# To parallelise across GPUs, comment out the runs you want to defer and
# invoke with CUDA_VISIBLE_DEVICES=N. ~30-60 min per run on a single A100.
#
# Usage:
#   ./run_smpnn_ablations.sh 2>&1 | tee logs/smpnn-ablations.log
# Or split across 2 GPUs (rough split: first 4 on GPU0, last 3 on GPU1):
#   # comment B1/D2/D3 here, then:
#   CUDA_VISIBLE_DEVICES=0 ./run_smpnn_ablations.sh 2>&1 | tee logs/smpnn-ablations-gpu0.log &
#   # in a second shell with C3/A2/A3/A4 commented out:
#   CUDA_VISIBLE_DEVICES=1 ./run_smpnn_ablations.sh 2>&1 | tee logs/smpnn-ablations-gpu1.log

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_one() {
    local TAG=$1
    local FLAGS=$2
    echo ""
    echo "================================================"
    echo ">>> Training smpnn-ablation-${TAG}"
    echo "    flags: $FLAGS"
    echo "================================================"
    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-ablation-${TAG} smpnn-ablation-${TAG} \
        --tasks others-1 \
        $FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath checkpoints/smpnn-ablation-${TAG}
}

# ─── Study C: depth-scaling control ────────────────────────────────────────
# C3: the critical control showing oversmoothing at num_mp=6 without SMPNN.
# This is the headline figure's missing data point.
run_one "c3-vanilla-6"     "--num_mp 6"

# ─── Study A: SMPNN component ablation (all at num_mp=6) ───────────────────
# A2: is the learnable alpha scaling needed?
run_one "a2-no-alpha"      "--num_mp 6 --use_smpnn --use_alpha False --log_alpha_every 2"

# A3: is the pointwise feedforward sub-block needed?
run_one "a3-no-ff"         "--num_mp 6 --use_smpnn --use_ff False --log_alpha_every 2"

# A4: is the Pre-LN before the GNN sub-block needed?
run_one "a4-no-gnn-ln"     "--num_mp 6 --use_smpnn --use_gnn_ln False --log_alpha_every 2"

# ─── Study D: alpha sensitivity (all SMPNN-6) ──────────────────────────────
# D2: faster ramp-up. Layers contribute earlier in training.
run_one "d2-alpha-1e-4"    "--num_mp 6 --use_smpnn --alpha_init 1e-4 --log_alpha_every 2"

# D3: aggressive init. May destabilise if GNN outputs are noisy early.
run_one "d3-alpha-1e-2"    "--num_mp 6 --use_smpnn --alpha_init 1e-2 --log_alpha_every 2"

# ─── Study B: attention augmentation (paper Appendix A) ────────────────────
# B1: SMPNN-6 + 1-head linear global attention. Cheapest probe of whether
# global context helps on heterogeneous RDB graphs. Adds ~4.7M params.
# Only run B2 (4 heads) and B3 (attn-only) if B1 shows >1% gain.
run_one "b1-attn-1h"       "--num_mp 6 --use_smpnn --use_attention True --num_heads 1 --log_alpha_every 2"

echo ""
echo "================================================"
echo "All ablations complete. Compare via:"
echo "  ls -la logs/smpnn-ablation-*/"
echo "  Key checkpoints under: checkpoints/smpnn-ablation-*/"
echo "================================================"
