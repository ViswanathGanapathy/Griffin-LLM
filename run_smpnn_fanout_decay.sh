#!/bin/bash
# Demonstration: SMPNN-6 with hop=num_mp=6 and geometric per-hop fanout decay.
#
# Motivation.
# When num_mp = L, a subgraph sampled with hop = 2 is smaller than the
# effective receptive field of an L=6 SMPNN — the outer 4 layers see
# nothing new because they run over a graph that has been fully
# aggregated after 2 hops. Setting hop = num_mp = 6 matches the two,
# but at the vanilla constant-fanout of 20 per (source, edge_type) per
# hop, the sampled subgraph explodes exponentially. This is why the
# original Griffin defaults ship with hop = 2.
#
# The --fanout_decay flag (added to hdataset.py and all training/eval
# entrypoints) applies a geometric shrink to the fanout per hop:
#
#   hop_fanout(h) = max(1, ceil(fanout * decay ** h))
#
# So with fanout=20, decay=0.5, hop=6:
#   hop 0: 20 neighbours per (src, edge_type)
#   hop 1: 10
#   hop 2:  5
#   hop 3:  3
#   hop 4:  2
#   hop 5:  1
#
# The inner rings preserve the local receptive field; the outer rings
# stay tractable in memory. decay = 1.0 (default) reproduces the
# original constant-fanout behaviour exactly.
#
# This script trains SMPNN-6 on others-1 in three configurations for
# comparison (each with 3 seeds for cost control — the paper's headline
# numbers still need 5 seeds):
#
#   A. Baseline:    hop=2,  fanout=20, decay=1.0   (default; matches paper)
#   B. Deep-const:  hop=6,  fanout=20, decay=1.0   (WARNING: likely OOM)
#   C. Deep-decay:  hop=6,  fanout=20, decay=0.5   (recommended new config)
#   D. Deep-agg:    hop=6,  fanout=20, decay=0.25  (aggressive shrink)
#
# Only run B if you want to demonstrate the OOM problem. C and D are
# the interesting cells.
#
# Total: 3 configs (A, C, D) × 3 seeds = 9 trainings. ~30-40 GPU-h.
# Add config B to also demonstrate OOM (fails fast, ~10 min per attempt).
#
# Usage:
#   ./run_smpnn_fanout_decay.sh                       # runs A + C + D
#   INCLUDE_OOM=1 ./run_smpnn_fanout_decay.sh         # also runs B
#   SEEDS="42 43 44" ./run_smpnn_fanout_decay.sh      # override seeds
#   FORCE=1 ./run_smpnn_fanout_decay.sh               # re-train completed cells

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"42 43 44"}
FORCE=${FORCE:-0}
INCLUDE_OOM=${INCLUDE_OOM:-0}

LOGDIR="logs/smpnn-fanout-decay"
mkdir -p "$LOGDIR"

# Each entry: TAG|HOP|FANOUT|DECAY
CONFIGS=(
    "A-baseline-h2-f20-d1p0|2|20|1.0"
    "C-decay-h6-f20-d0p5|6|20|0.5"
    "D-decay-h6-f20-d0p25|6|20|0.25"
)
if [ "$INCLUDE_OOM" = "1" ]; then
    CONFIGS+=("B-const-h6-f20-d1p0|6|20|1.0")
fi

run_one() {
    local TAG=$1
    local HOP=$2
    local FANOUT=$3
    local DECAY=$4
    local SEED=$5

    local FULL_TAG="s${SEED}-${TAG}"
    local SAVE_PATH="checkpoints/smpnn-fanout-decay-${FULL_TAG}"
    local CELL_LOG="${LOGDIR}/${FULL_TAG}.log"

    if [ "$FORCE" = "0" ] && [ -d "${SAVE_PATH}/best_checkpoint" ]; then
        echo ">>> Skipping ${FULL_TAG} (best_checkpoint exists)"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> SMPNN-6 ${FULL_TAG}"
    echo "    hop: $HOP, fanout: $FANOUT, decay: $DECAY, seed: $SEED"
    echo "    Per-hop fanouts: $(python3 -c "
import math
d = float($DECAY)
f = int($FANOUT)
h = int($HOP)
print(', '.join(str(max(1, math.ceil(f * d**i))) for i in range(h)))
")"
    echo "    save: $SAVE_PATH"
    echo "    log:  $CELL_LOG"
    echo "================================================"

    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-fanout-decay-${FULL_TAG} smpnn-fanout-decay-${FULL_TAG} \
        --tasks others-1 \
        --seed ${SEED} \
        --num_mp 6 --use_smpnn --alpha_init 1e-2 \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop ${HOP} --fanout ${FANOUT} --fanout_decay ${DECAY} \
        --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath ${SAVE_PATH} \
        --log_alpha_every 2 \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "SMPNN-6 fanout-decay demonstration"
echo "  Configs: ${#CONFIGS[@]}  (add INCLUDE_OOM=1 to also test the const-h6 OOM case)"
echo "  SEEDS:   $SEEDS"
echo "  TOTAL:   ${#CONFIGS[@]} × $(echo $SEEDS | wc -w) = $((${#CONFIGS[@]} * $(echo $SEEDS | wc -w))) trainings"
echo "=========================================="

for SEED in $SEEDS; do
    for entry in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG HOP FANOUT DECAY <<< "$entry"
        run_one "$TAG" "$HOP" "$FANOUT" "$DECAY" "$SEED"
    done
done

echo ""
echo "=========================================="
echo "Done. Aggregate + compare via:"
echo ""
echo "  python collect_smpnn_results.py \\"
echo "      --logs logs/smpnn-fanout-decay/*.log \\"
echo "      --out smpnn_fanout_decay_results.csv"
echo ""
echo "Interesting comparisons:"
echo "  * Config A vs C: does hop=6 with decay=0.5 outperform hop=2 (proves"
echo "    the receptive-field motivation was correct)?"
echo "  * Config C vs D: does aggressive decay (0.25) hurt (finds the"
echo "    sweet spot for the shrink factor)?"
echo "  * Config B (if run): compare wall time / OOM behaviour against C."
echo "=========================================="
