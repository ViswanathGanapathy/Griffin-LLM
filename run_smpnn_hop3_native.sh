#!/bin/bash
# SMPNN-Griffin depth-scaling experiment at HOP=3 (longer relational reach).
#
# Companion to run_smpnn_depth_native.sh (which used hop=2). Tests whether
# SMPNN's depth advantage compounds with genuinely longer message-passing
# reach by extracting 3-hop subgraphs at training time.
#
# Four variants, all on others-1:
#   vanilla-4       @ hop=3 -- does longer reach alone help vanilla?
#   SMPNN-6         @ hop=3 -- SMPNN + longer reach (no attention)
#   SMPNN-8         @ hop=3 -- max depth + max reach
#   SMPNN-6 + attn  @ hop=3 -- SMPNN + longer reach + linear global attn (1h)
#
# Compared against the existing hop=2 results from run_smpnn_depth_native.sh.
# The two SMPNN-6 hop=3 variants (with vs without attention) are the
# "with/without transformer at extended reach" ablation pair.
#
# Memory / cost:
#   hop=3 + fanout=10 -> per-seed subgraph ~3x larger than hop=2 + fanout=20
#   batchsize dropped 256 -> 128 to compensate
#   ~6-8 hours per variant on a single A100
#
# Usage:
#   ./run_smpnn_hop3_native.sh 2>&1 | tee logs/smpnn-hop3.log
#
# Or split across two GPUs for ~halved wall time:
#   CUDA_VISIBLE_DEVICES=0 RUN_ONLY=vanilla-4-hop3 ./run_smpnn_hop3_native.sh ...
#   CUDA_VISIBLE_DEVICES=1 RUN_ONLY=smpnn-6-hop3   ./run_smpnn_hop3_native.sh ...
# (then later: CUDA_VISIBLE_DEVICES=0 RUN_ONLY=smpnn-8-hop3 ./run_smpnn_hop3_native.sh)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_ONLY=${RUN_ONLY:-all}

run_one() {
    local TAG=$1
    local FLAGS=$2
    if [ "$RUN_ONLY" != "all" ] && [ "$RUN_ONLY" != "$TAG" ]; then
        echo ">>> Skipping $TAG (RUN_ONLY=$RUN_ONLY)"
        return
    fi
    echo ""
    echo "================================================"
    echo ">>> Training $TAG"
    echo "================================================"
    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-hop3-${TAG} smpnn-hop3-${TAG} \
        --tasks others-1 \
        $FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 128 \
        --lr 3e-4 --wd 4e-4 \
        --hop 3 --fanout 10 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath checkpoints/smpnn-hop3-${TAG}
}

# 1. Vanilla-4 at hop=3 -- does longer reach alone help vanilla?
run_one "vanilla-4-hop3" "--num_mp 4"

# 2. SMPNN-6 at hop=3 -- the headline new variant
run_one "smpnn-6-hop3" "--num_mp 6 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"

# 3. SMPNN-8 at hop=3 -- depth ceiling + longer reach
run_one "smpnn-8-hop3" "--num_mp 8 --use_smpnn --alpha_init 1e-6 --log_alpha_every 2"

# 4. SMPNN-6 + attention at hop=3 -- with-transformer companion to #2.
#    Adds ~4.7M params (3 * 512^2 * num_mp via WQ/WK/WV in LinearGlobalAttention).
#    Compare metric AND param count vs smpnn-6-hop3 for the paper's Table 3.
run_one "smpnn-6-hop3-attn-1h" \
    "--num_mp 6 --use_smpnn --use_attention True --num_heads 1 --alpha_init 1e-6 --log_alpha_every 2"

echo ""
echo "================================================"
echo "Done. Compare final 'Average test metric' lines across:"
echo "  logs/smpnn-hop3.log               (this run)"
echo "  logs/smpnn-depth.log              (the hop=2 baselines)"
echo ""
echo "Headline with/without-transformer pair at hop=3:"
echo "  no  transformer: logs/smpnn-hop3-smpnn-6-hop3/"
echo "  yes transformer: logs/smpnn-hop3-smpnn-6-hop3-attn-1h/"
echo "================================================"
