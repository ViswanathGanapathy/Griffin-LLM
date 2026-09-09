#!/bin/bash
# Second anchor task for o1→o2 transfer: rel-trial-study-outcome.
#
# Purpose: strengthen the o1→o2 headline claim in the paper draft.
# Currently the 7× variance-reduction result rests on airbnb-destination
# alone. Adding rel-trial-study-outcome as a second binary-classification
# AUROC anchor doubles the evidence: if variance reduction reproduces on
# a second task, the claim generalises within the target family; if it
# does not, the airbnb-destination result may be task-specific.
#
# Both tasks are in the others-2 family — same source (others-1) backbones,
# same eval direction. We just point --eval_tasks at the specific task
# name (not the family) so the eval focuses on rel-trial-study-outcome
# instead of averaging over the whole family.
#
# Backbones (all from existing multiseed checkpoints):
#   Vanilla-4       → smpnn-multiseed-s{42,43,44,45,46}-others-1-c1-vanilla-4
#   Vanilla-6       → smpnn-multiseed-s{42,43,44,45,46}-others-1-c3-vanilla-6
#   SMPNN-6 α=1e-6  → smpnn-multiseed-s{42,43,44,45,46}-others-1-d1-smpnn-6
#   SMPNN-6 α=1e-2  → smpnn-multiseed-s{42,43,44,45,46}-others-1-d3-alpha-1e-2
#
# Heads: TabPFN v3 (matches Table 5.1 anchor) + TabICL v2 (bonus).
# Total: 4 backbones × 5 seeds × 2 heads = 40 eval runs.
# Wall time: ~5 min/eval = ~3.3 GPU-hours.
#
# Pre-req: run_smpnn_multiseed_backbones.sh must have completed for all
# 5 seeds.
#
# Each cell tees to its own log at
# logs/smpnn-o1-o2-second-anchor/<TAG>.log.
#
# Usage:
#   ./run_smpnn_o1_o2_second_anchor.sh
#       Runs all 40 evals; skips completed ones.
#
#   FORCE=1 ./run_smpnn_o1_o2_second_anchor.sh
#       Re-runs completed evals.
#
#   HEAD=tabpfn ./run_smpnn_o1_o2_second_anchor.sh
#       Only run TabPFN (or 'tabicl' for TabICL only). Default: both.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
FORCE=${FORCE:-0}
HEAD=${HEAD:-both}
RUN_ONLY=${RUN_ONLY:-all}

EVAL_LOGDIR="logs/smpnn-o1-o2-second-anchor"
mkdir -p "$EVAL_LOGDIR"

# Target task (single binary AUROC task within others-2)
TARGET_TASK="rel-trial-study-outcome"

# Backbone entries — reuse the multiseed checkpoints
# Each entry: SEED|BACKBONE_TAG|CKPT|BACKBONE_FLAGS
CASES=()
for SEED in 42 43 44 45 46; do
    CASES+=(
        "${SEED}|c1-vanilla-4|checkpoints/smpnn-multiseed-s${SEED}-others-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
        "${SEED}|c3-vanilla-6|checkpoints/smpnn-multiseed-s${SEED}-others-1-c3-vanilla-6/best_checkpoint|--num_mp 6"
        "${SEED}|d1-smpnn-6|checkpoints/smpnn-multiseed-s${SEED}-others-1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-6"
        "${SEED}|d3-alpha-1e-2|checkpoints/smpnn-multiseed-s${SEED}-others-1-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    )
done

# Heads to evaluate
if [ "$HEAD" = "tabpfn" ]; then
    HEADS=("tabpfn")
elif [ "$HEAD" = "tabicl" ]; then
    HEADS=("tabicl")
else
    HEADS=("tabpfn" "tabicl")
fi

DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

cell_is_complete() {
    local cell_log=$1
    [ -f "$cell_log" ] || return 1
    # Expect exactly 1 test_metric line for a single-task eval
    local n
    n=$(grep -c "test_metric/" "$cell_log" 2>/dev/null || echo 0)
    [ "$n" -ge 1 ]
}

eval_one() {
    local SEED=$1
    local BACKBONE_TAG=$2
    local CKPT=$3
    local BACKBONE_FLAGS=$4
    local HEAD_KIND=$5

    local RUN_TAG="s${SEED}-${BACKBONE_TAG}-${HEAD_KIND}-trial"
    local CELL_LOG="${EVAL_LOGDIR}/${RUN_TAG}.log"

    if [ "$RUN_ONLY" != "all" ]; then
        local match_tag="s${SEED}-${BACKBONE_TAG}"
        local found=0
        for t in $RUN_ONLY; do
            if [ "$t" = "$match_tag" ] || [ "$t" = "$RUN_TAG" ]; then
                found=1; break
            fi
        done
        if [ "$found" = "0" ]; then
            echo ">>> Skipping $RUN_TAG (not in RUN_ONLY='$RUN_ONLY')"
            return
        fi
    fi

    if [ "$FORCE" = "0" ] && cell_is_complete "$CELL_LOG"; then
        local nlines
        nlines=$(grep -c "test_metric/" "$CELL_LOG")
        echo ">>> Skipping $RUN_TAG (already complete: $nlines test_metric lines)"
        return
    fi

    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT (cell: $RUN_TAG)"
        echo "       run run_smpnn_multiseed_backbones.sh first"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    seed:        $SEED"
    echo "    backbone:    $BACKBONE_TAG"
    echo "    head:        $HEAD_KIND"
    echo "    target task: $TARGET_TASK"
    echo "    ckpt:        $CKPT"
    echo "    log:         $CELL_LOG"
    echo "================================================"

    local EXTRA_FLAGS
    if [ "$HEAD_KIND" = "tabpfn" ]; then
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 30000 --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache --tabpfn_inference_precision autocast --tabpfn_inference_config {\"MAX_NUMBER_OF_SAMPLES\":50000,\"MAX_NUMBER_OF_FEATURES\":600}"
    else
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 10000 --tabicl_checkpoint_version v2"
    fi

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-o1-o2-second-anchor-${RUN_TAG} smpnn-o1-o2-second-anchor-${RUN_TAG} \
        --head $HEAD_KIND \
        --tasks rel-f1-driver-position \
        --eval_tasks ${TARGET_TASK} \
        --loadpath $CKPT \
        --seed $SEED \
        $BACKBONE_FLAGS \
        $DATA_FLAGS \
        --no_icl_projection \
        --probe_epochs 0 \
        --hiddim 512 --use_rev True --use_gate True \
        --batchsize 256 \
        --output_mlp_dim 1 --no_target_normalize \
        $EXTRA_FLAGS \
        --savepath checkpoints/smpnn-o1-o2-second-anchor-${RUN_TAG} \
        2>&1 | tee "$CELL_LOG"

    local nlines
    nlines=$(grep -c "test_metric/" "$CELL_LOG" 2>/dev/null || echo 0)
    if [ "$nlines" -ge 1 ]; then
        echo ">>> $RUN_TAG: OK ($nlines test_metric lines captured)"
    else
        echo ">>> $RUN_TAG: WARNING only $nlines test_metric lines"
    fi
}

echo "=========================================="
echo "o1→o2 second-anchor eval on ${TARGET_TASK}"
echo "  Backbones: Vanilla-4, Vanilla-6, SMPNN-6 α=1e-6, SMPNN-6 α=1e-2"
echo "  Seeds:     42, 43, 44, 45, 46"
echo "  Heads:     ${HEADS[*]}"
echo "  Total:     4 × 5 × ${#HEADS[@]} = $((4 * 5 * ${#HEADS[@]})) eval runs"
echo "  FORCE=$FORCE  RUN_ONLY=$RUN_ONLY  HEAD=$HEAD"
echo "  Per-cell logs: ${EVAL_LOGDIR}/"
echo "=========================================="

for HEAD_KIND in "${HEADS[@]}"; do
    for entry in "${CASES[@]}"; do
        IFS='|' read -r SEED BACKBONE_TAG CKPT BACKBONE_FLAGS <<< "$entry"
        eval_one "$SEED" "$BACKBONE_TAG" "$CKPT" "$BACKBONE_FLAGS" "$HEAD_KIND"
    done
done

echo ""
echo "=========================================="
echo "Second-anchor eval complete. Aggregate:"
echo ""
echo "  python collect_smpnn_results.py \\"
echo "      --logs ${EVAL_LOGDIR}/*.log \\"
echo "      --out smpnn_o1_o2_second_anchor_results.csv"
echo ""
echo "Compare mean ± std of ${TARGET_TASK} AUROC across the 5 seeds"
echo "against Table 5.1 (airbnb-destination) in the paper draft."
echo "If SMPNN-6 α=1e-2 still has ≥3× lower std than Vanilla-6 here,"
echo "the variance-reduction claim is confirmed across two anchors."
echo "=========================================="
