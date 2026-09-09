#!/bin/bash
# Native Griffin head evaluation on o1→o2 transfer.
#
# Purpose: sanity-check the variance-reduction claim from the paper draft.
# Multi-seed multiseed_eval.sh used only ICL heads (TabPFN, TabICL). This
# script re-runs the same 4 backbones × 5 seeds on the same direction
# (others-1 → others-2) with Griffin's NATIVE head — a linear decoder on
# top of the frozen encoder embeddings.
#
# If the 7× variance-reduction result holds under the native head too,
# it is not an artifact of TabPFN's ensembling or subsampling; it is a
# property of the SMPNN backbone's representation. This is the strongest
# possible confirmation of the headline claim.
#
# Backbones (all from existing multiseed checkpoints):
#   Vanilla-4       → smpnn-multiseed-s{42,43,44,45,46}-others-1-c1-vanilla-4
#   Vanilla-6       → smpnn-multiseed-s{42,43,44,45,46}-others-1-c3-vanilla-6
#   SMPNN-6 α=1e-6  → smpnn-multiseed-s{42,43,44,45,46}-others-1-d1-smpnn-6
#   SMPNN-6 α=1e-2  → smpnn-multiseed-s{42,43,44,45,46}-others-1-d3-alpha-1e-2
#
# Total: 4 backbones × 5 seeds = 20 eval runs.
# Wall time: ~5 min/eval = ~1.7 GPU-hours.
#
# Pre-req: run_smpnn_multiseed_backbones.sh must have completed for all 5
# seeds. If your seed-42 checkpoints live at the older paths
# (smpnn-depth-vanilla-4, smpnn-ablation-d3-alpha-1e-2), edit the CASES
# block below.
#
# Each cell tees to its own log at logs/smpnn-native-head-o1-o2/<TAG>.log
# so output is durable without a wrapping tee.
#
# Usage:
#   ./run_smpnn_native_head_o1_o2.sh
#       Runs all 20 evals; skips ones whose log already has ≥2
#       test_metric lines.
#
#   FORCE=1 ./run_smpnn_native_head_o1_o2.sh
#       Re-runs completed evals.
#
#   RUN_ONLY="s43-vanilla-4 s44-d3-alpha-1e-2" ./...
#       Only run these tag suffixes.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
FORCE=${FORCE:-0}
RUN_ONLY=${RUN_ONLY:-all}

EVAL_LOGDIR="logs/smpnn-native-head-o1-o2"
mkdir -p "$EVAL_LOGDIR"

# Each entry: SEED|BACKBONE_TAG|CKPT|BACKBONE_FLAGS
# Backbone flags mirror the training flags for the corresponding architecture.
CASES=()
for SEED in 42 43 44 45 46; do
    CASES+=(
        "${SEED}|c1-vanilla-4|checkpoints/smpnn-multiseed-s${SEED}-others-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
        "${SEED}|c3-vanilla-6|checkpoints/smpnn-multiseed-s${SEED}-others-1-c3-vanilla-6/best_checkpoint|--num_mp 6"
        "${SEED}|d1-smpnn-6|checkpoints/smpnn-multiseed-s${SEED}-others-1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-6"
        "${SEED}|d3-alpha-1e-2|checkpoints/smpnn-multiseed-s${SEED}-others-1-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    )
done

DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

cell_is_complete() {
    local cell_log=$1
    [ -f "$cell_log" ] || return 1
    local n
    n=$(grep -c "test_metric/" "$cell_log" 2>/dev/null || echo 0)
    [ "$n" -ge 2 ]
}

eval_one() {
    local SEED=$1
    local BACKBONE_TAG=$2
    local CKPT=$3
    local BACKBONE_FLAGS=$4

    local RUN_TAG="s${SEED}-${BACKBONE_TAG}-native"
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
    echo ">>> $RUN_TAG (native head)"
    echo "    seed:      $SEED"
    echo "    backbone:  $BACKBONE_TAG"
    echo "    ckpt:      $CKPT"
    echo "    log:       $CELL_LOG"
    echo "================================================"

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-native-head-o1-o2-${RUN_TAG} smpnn-native-head-o1-o2-${RUN_TAG} \
        --head default \
        --tasks rel-f1-driver-position \
        --eval_tasks others-2 \
        --loadpath $CKPT \
        --seed $SEED \
        $BACKBONE_FLAGS \
        $DATA_FLAGS \
        --hiddim 512 --use_rev True --use_gate True \
        --batchsize 256 \
        --output_mlp_dim 1 --no_target_normalize \
        --savepath checkpoints/smpnn-native-head-o1-o2-${RUN_TAG} \
        2>&1 | tee "$CELL_LOG"

    local nlines
    nlines=$(grep -c "test_metric/" "$CELL_LOG" 2>/dev/null || echo 0)
    if [ "$nlines" -ge 2 ]; then
        echo ">>> $RUN_TAG: OK ($nlines test_metric lines captured)"
    else
        echo ">>> $RUN_TAG: WARNING only $nlines test_metric lines"
    fi
}

echo "=========================================="
echo "SMPNN native-head o1→o2 confirmation eval"
echo "  Backbones: Vanilla-4, Vanilla-6, SMPNN-6 α=1e-6, SMPNN-6 α=1e-2"
echo "  Seeds:     42, 43, 44, 45, 46"
echo "  Total:     4 × 5 = 20 eval runs"
echo "  FORCE=$FORCE  RUN_ONLY=$RUN_ONLY"
echo "  Per-cell logs: ${EVAL_LOGDIR}/"
echo "=========================================="

for entry in "${CASES[@]}"; do
    IFS='|' read -r SEED BACKBONE_TAG CKPT BACKBONE_FLAGS <<< "$entry"
    eval_one "$SEED" "$BACKBONE_TAG" "$CKPT" "$BACKBONE_FLAGS"
done

echo ""
echo "=========================================="
echo "Native-head eval complete. Aggregate:"
echo ""
echo "  python collect_smpnn_results.py \\"
echo "      --logs ${EVAL_LOGDIR}/*.log \\"
echo "      --out smpnn_native_head_o1_o2_results.csv"
echo ""
echo "Then compute mean ± std per backbone across the 5 seeds and"
echo "compare against Table 5.1 in GRIFFIN_SMPNN_PAPER_DRAFT.md."
echo "=========================================="
