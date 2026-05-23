#!/bin/bash
# SMPNN-Griffin parity smoke test using NATIVE head (no LLM, no TabICL/PFN).
#
# Trains Griffin (SMPNN variant, num_mp=4) for a short run on a single
# task (rel-f1-driver-position) and reports val + test metrics. Compare
# to a vanilla Griffin run with the same config — they should be close
# at num_mp=4 with alpha_init=1e-6 (near-identity init means SMPNN
# behaves like vanilla initially, then diverges as alphas ramp up).
#
# Usage:
#   ./run_smpnn_parity_native.sh           # SMPNN variant
#   USE_SMPNN=0 ./run_smpnn_parity_native.sh   # vanilla baseline
#
# Or run both back-to-back:
#   USE_SMPNN=0 ./run_smpnn_parity_native.sh 2>&1 | tee logs/smpnn-parity-vanilla.log
#   USE_SMPNN=1 ./run_smpnn_parity_native.sh 2>&1 | tee logs/smpnn-parity-smpnn.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
USE_SMPNN=${USE_SMPNN:-1}

if [ "$USE_SMPNN" = "1" ]; then
    SMPNN_FLAGS="--use_smpnn --alpha_init 1e-6 --log_alpha_every 1"
    TAG="smpnn"
else
    SMPNN_FLAGS=""
    TAG="vanilla"
fi

echo "================================================"
echo "Griffin parity test: ${TAG} backbone, native head"
echo "================================================"

PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
    datasets/joint-v65 logs/smpnn-parity-${TAG} smpnn-parity-${TAG} \
    --tasks rel-f1-driver-position \
    --num_mp 4 --hiddim 512 \
    --use_rev True --use_gate True \
    ${SMPNN_FLAGS} \
    --maxepoch 10 \
    --batchsize 256 \
    --lr 3e-4 --wd 4e-4 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --eval_per_epoch 1 \
    --savepath checkpoints/smpnn-parity-${TAG}
