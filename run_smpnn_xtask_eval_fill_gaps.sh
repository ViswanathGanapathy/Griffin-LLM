#!/bin/bash
# Fill in cross-task eval cells that were lost (terminal output not saved)
# or never completed (script crashed / interrupted before reaching them).
#
# Differences vs run_smpnn_xtask_eval.sh:
#   - Each cell tees its OWN stdout to logs/smpnn-xtask-eval-fill/<TAG>.log
#     so a dropped session NEVER loses output -- the per-cell log is the
#     durable record (independent of the user's tee on launch).
#   - Each cell is SKIPPED if its log file already has 6 'test_metric/'
#     lines (=fully captured). Override with FORCE=1.
#   - RUN_ONLY env var filters to specific cells. Default: all gap cells.
#
# Usage:
#   ./run_smpnn_xtask_eval_fill_gaps.sh
#       Auto-skips cells already captured; runs only the gaps.
#
#   FORCE=1 ./run_smpnn_xtask_eval_fill_gaps.sh
#       Re-runs even cells whose log files exist.
#
#   RUN_ONLY="o1-to-o2-c1-vanilla-4-tabpfn o1-to-o2-d1-smpnn-6-tabpfn" \
#       ./run_smpnn_xtask_eval_fill_gaps.sh
#       Runs only the named cells.
#
# Wall time: ~8 min/cell average. 12 gap cells -> ~1.5-2 GPU-hours total.
# Less if many were already captured in your original run.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
FORCE=${FORCE:-0}
RUN_ONLY=${RUN_ONLY:-all}

FILL_LOGDIR="logs/smpnn-xtask-eval-fill"
mkdir -p "$FILL_LOGDIR"

# Same CASES structure as run_smpnn_xtask_eval.sh:
# DIRECTION|SRC_TAG|TGT_FAMILY|CKPT|BACKBONE_FLAGS
ALL_GAP_CASES=(
    # ─── o1->o2 cells (lost in original run, no tee was used) ───
    "o1-to-o2|c1-vanilla-4|others-2|checkpoints/smpnn-depth-vanilla-4/best_checkpoint|--num_mp 4"
    "o1-to-o2|d1-smpnn-6|others-2|checkpoints/smpnn-depth-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "o1-to-o2|d3-alpha-1e-2|others-2|checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"

    # ─── o2->o1 cells with partial loss (cells 7-8) ───
    "o2-to-o1|c1-vanilla-4|others-1|checkpoints/smpnn-xtask-o2-c1-vanilla-4/best_checkpoint|--num_mp 4"

    # ─── c2->c1 cells (potentially incomplete from original run) ───
    "c2-to-c1|c1-vanilla-4|commerce-1|checkpoints/smpnn-xtask-c2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "c2-to-c1|d1-smpnn-6|commerce-1|checkpoints/smpnn-xtask-c2-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "c2-to-c1|d3-alpha-1e-2|commerce-1|checkpoints/smpnn-xtask-c2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
)

DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

cell_is_complete() {
    # A cell is "complete" if its per-cell log has at least 6 test_metric/ lines.
    local cell_log=$1
    [ -f "$cell_log" ] || return 1
    local n
    n=$(grep -c "test_metric/" "$cell_log" 2>/dev/null || echo 0)
    [ "$n" -ge 6 ]
}

eval_one() {
    local DIR=$1
    local SRC_TAG=$2
    local HEAD=$3
    local TGT_FAMILY=$4
    local CKPT=$5
    local BACKBONE_FLAGS=$6
    local EXTRA_FLAGS=$7

    local RUN_TAG="${DIR}-${SRC_TAG}-${HEAD}-noproj"
    local CELL_LOG="${FILL_LOGDIR}/${RUN_TAG}.log"

    # RUN_ONLY filter
    if [ "$RUN_ONLY" != "all" ]; then
        local found=0
        for t in $RUN_ONLY; do
            if [ "$t" = "$RUN_TAG" ]; then found=1; break; fi
        done
        if [ "$found" = "0" ]; then
            echo ">>> Skipping $RUN_TAG (not in RUN_ONLY='$RUN_ONLY')"
            return
        fi
    fi

    # Skip-if-complete check
    if [ "$FORCE" = "0" ] && cell_is_complete "$CELL_LOG"; then
        local nlines
        nlines=$(grep -c "test_metric/" "$CELL_LOG")
        echo ">>> Skipping $RUN_TAG (already complete: $nlines test_metric lines in $CELL_LOG)"
        return
    fi

    # Checkpoint exists?
    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT (cell: $RUN_TAG)"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    direction: $DIR    backbone: $SRC_TAG    head: $HEAD"
    echo "    target:    $TGT_FAMILY"
    echo "    ckpt:      $CKPT"
    echo "    log:       $CELL_LOG"
    echo "================================================"

    # Run, tee to per-cell log AND propagate to master stdout
    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-xtask-${RUN_TAG} smpnn-xtask-${RUN_TAG} \
        --head $HEAD \
        --tasks rel-f1-driver-position \
        --eval_tasks $TGT_FAMILY \
        --loadpath $CKPT \
        $BACKBONE_FLAGS \
        $DATA_FLAGS \
        --no_icl_projection \
        --probe_epochs 0 \
        --hiddim 512 --use_rev True --use_gate True \
        --batchsize 256 \
        --output_mlp_dim 1 --no_target_normalize \
        $EXTRA_FLAGS \
        --savepath checkpoints/smpnn-xtask-${RUN_TAG} \
        2>&1 | tee "$CELL_LOG"

    # Post-run verification: did we get 6 metrics?
    local nlines
    nlines=$(grep -c "test_metric/" "$CELL_LOG" 2>/dev/null || echo 0)
    if [ "$nlines" -ge 6 ]; then
        echo ">>> $RUN_TAG: OK ($nlines test_metric lines captured)"
    else
        echo ">>> $RUN_TAG: WARNING only $nlines test_metric lines captured (expected 6)"
    fi
}

TABPFN_FLAGS=(
    --icl_n_estimators 8
    --icl_max_context 30000
    --tabpfn_version v3
    --tabpfn_fit_mode fit_with_cache
    --tabpfn_inference_precision autocast
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES":50000,"MAX_NUMBER_OF_FEATURES":600}'
)

TABICL_FLAGS=(
    --icl_n_estimators 8
    --icl_max_context 10000
    --tabicl_checkpoint_version v2
)

echo "=========================================="
echo "SMPNN xtask eval gap-filler"
echo "  FORCE    = $FORCE"
echo "  RUN_ONLY = $RUN_ONLY"
echo "  Per-cell logs go to: $FILL_LOGDIR/"
echo "=========================================="

for entry in "${ALL_GAP_CASES[@]}"; do
    IFS='|' read -r DIR SRC_TAG TGT_FAMILY CKPT BACKBONE_FLAGS <<< "$entry"
    eval_one "$DIR" "$SRC_TAG" "tabpfn" "$TGT_FAMILY" "$CKPT" "$BACKBONE_FLAGS" "${TABPFN_FLAGS[*]}"
    eval_one "$DIR" "$SRC_TAG" "tabicl" "$TGT_FAMILY" "$CKPT" "$BACKBONE_FLAGS" "${TABICL_FLAGS[*]}"
done

echo ""
echo "=========================================="
echo "Gap-fill run complete."
echo "Per-cell logs:"
ls -la "$FILL_LOGDIR"/ 2>/dev/null | head
echo ""
echo "Aggregate results across ALL eval logs:"
echo "  python collect_smpnn_results.py \\"
echo "    --logs $FILL_LOGDIR/*.log \\"
echo "    --out smpnn_xtask_combined_results.csv"
echo "=========================================="
