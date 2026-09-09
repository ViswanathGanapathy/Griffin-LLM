#!/bin/bash
# Multi-seed backbone training for the SMPNN-Griffin ablation study.
#
# The existing checkpoints (smpnn-depth-*, smpnn-ablation-*, smpnn-xtask-*)
# all used seed=42. This script trains the same 3 anchor variants
# (C1 vanilla-4, D1 SMPNN-6 default, D3 SMPNN-6 alpha=1e-2) with seeds
# 43 and 44, so the headline numbers can be reported as mean +/- std
# over n=3 seeds.
#
# Scope: 3 anchors x 4 source families x 2 new seeds = 24 trainings.
# Wall time: ~3-4h per training on A100 = ~72-96 GPU-hours total at one GPU.
# Trivially parallelisable across GPUs via CUDA_VISIBLE_DEVICES + RUN_ONLY.
#
# Recommended split: run only the cells needed for the 3 decisive
# cross-task wins (Δ > 0.02), via the SCOPE env var.
#
# Output: checkpoints/smpnn-multiseed-s<seed>-<src>-<anchor>/best_checkpoint
#
# Usage:
#   ./run_smpnn_multiseed_backbones.sh
#       Runs everything in scope (default scope: decisive-only, 14 trainings).
#
#   SCOPE=full ./run_smpnn_multiseed_backbones.sh
#       Runs all 24 (3 anchors x 4 source families x 2 seeds).
#
#   SCOPE=decisive ./run_smpnn_multiseed_backbones.sh
#       (default) Only the source families needed for the 3 decisive cells:
#         - others-1: C1 + D3 (for o1->o2 TabPFN +0.026)
#         - commerce-1: C1 + D1 (for c1->c2 TabICL +0.050)
#         - commerce-2: C1 + D3 (for c2->c1 TabICL +0.018)
#         Plus D1 on others-1 for in-dist V5-vs-D1 comparison.
#         Total: 14 trainings.
#
#   SEEDS="43 44" ./run_smpnn_multiseed_backbones.sh
#       Override which seeds to add. Default: 43 44 (i.e. 2 extra seeds).
#
#   RUN_ONLY="s43-o2-d1-smpnn-6 s44-o2-d1-smpnn-6" ./...
#       Run only the named tags.
#
#   FORCE=1 ./...
#       Re-run even if best_checkpoint already exists. Default: skip.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SCOPE=${SCOPE:-decisive}
SEEDS=${SEEDS:-"43 44"}
RUN_ONLY=${RUN_ONLY:-all}
FORCE=${FORCE:-0}

MULTISEED_LOGDIR="logs/smpnn-multiseed"
mkdir -p "$MULTISEED_LOGDIR"

# Each entry: SOURCE_FAMILY|ANCHOR_TAG|TRAINING_FLAGS
# (seed is sliced in by the loop)
ALL_ENTRIES=(
    # ─── others-1 source ───
    "others-1|c1-vanilla-4|--num_mp 4"
    "others-1|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "others-1|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # ─── others-2 source ───
    "others-2|c1-vanilla-4|--num_mp 4"
    "others-2|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "others-2|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # ─── commerce-1 source ───
    "commerce-1|c1-vanilla-4|--num_mp 4"
    "commerce-1|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "commerce-1|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # ─── commerce-2 source ───
    "commerce-2|c1-vanilla-4|--num_mp 4"
    "commerce-2|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "commerce-2|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
)

# Decisive-cell scope: only the anchors needed for the 3 winning cells,
# plus D1 on others-1 for in-dist V5-vs-D1 (since we want to confirm
# V5 > V2 in-distribution too).
DECISIVE_ENTRIES=(
    "others-1|c1-vanilla-4|--num_mp 4"
    "others-1|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "others-1|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    "commerce-1|c1-vanilla-4|--num_mp 4"
    "commerce-1|d1-smpnn-6|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    "commerce-2|c1-vanilla-4|--num_mp 4"
    "commerce-2|d3-alpha-1e-2|--num_mp 6 --use_smpnn --alpha_init 1e-2"
)

if [ "$SCOPE" = "full" ]; then
    ENTRIES=("${ALL_ENTRIES[@]}")
else
    ENTRIES=("${DECISIVE_ENTRIES[@]}")
fi

ckpt_dir() {
    local SEED=$1
    local SRC=$2
    local ANCHOR=$3
    echo "checkpoints/smpnn-multiseed-s${SEED}-${SRC}-${ANCHOR}"
}

ckpt_exists() {
    local SEED=$1; local SRC=$2; local ANCHOR=$3
    [ -d "$(ckpt_dir $SEED $SRC $ANCHOR)/best_checkpoint" ]
}

run_one() {
    local SEED=$1
    local SRC=$2
    local ANCHOR=$3
    local FLAGS=$4

    local TAG="s${SEED}-${SRC}-${ANCHOR}"

    # RUN_ONLY filter
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

    # Skip-if-complete check
    local SAVE_PATH="$(ckpt_dir $SEED $SRC $ANCHOR)"
    if [ "$FORCE" = "0" ] && ckpt_exists "$SEED" "$SRC" "$ANCHOR"; then
        echo ">>> Skipping $TAG (best_checkpoint exists at $SAVE_PATH/best_checkpoint)"
        return
    fi

    local CELL_LOG="${MULTISEED_LOGDIR}/${TAG}.log"

    echo ""
    echo "================================================"
    echo ">>> Training smpnn-multiseed-${TAG}"
    echo "    seed:      $SEED"
    echo "    source:    $SRC"
    echo "    anchor:    $ANCHOR"
    echo "    flags:     $FLAGS"
    echo "    save:      $SAVE_PATH"
    echo "    log:       $CELL_LOG"
    echo "================================================"

    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-multiseed-${TAG} smpnn-multiseed-${TAG} \
        --tasks ${SRC} \
        --seed ${SEED} \
        $FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath ${SAVE_PATH} \
        --log_alpha_every 2 \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "SMPNN multi-seed backbone training"
echo "  SCOPE   = $SCOPE ($(echo $SCOPE | wc -c) chars)"
echo "  SEEDS   = $SEEDS"
echo "  ENTRIES = ${#ENTRIES[@]} source-anchor combos"
echo "  TOTAL   = ${#ENTRIES[@]} entries x $(echo $SEEDS | wc -w) seeds = $((${#ENTRIES[@]} * $(echo $SEEDS | wc -w))) trainings"
echo "  Per-cell logs: ${MULTISEED_LOGDIR}/"
echo "=========================================="

for SEED in $SEEDS; do
    for entry in "${ENTRIES[@]}"; do
        IFS='|' read -r SRC ANCHOR FLAGS <<< "$entry"
        run_one "$SEED" "$SRC" "$ANCHOR" "$FLAGS"
    done
done

echo ""
echo "=========================================="
echo "Multi-seed backbone training complete."
echo "Next: ./run_smpnn_multiseed_eval.sh to evaluate the 3 decisive"
echo "cross-task cells across all 3 seeds (42 + the new ones)."
echo "=========================================="
