#!/bin/bash
# Multi-seed evaluation for the 3 decisive cross-task wins (Δ > 0.02).
# Runs the ICL eval (TabPFN ZS no-proj or TabICL ZS no-proj) against each
# of the 3 seeds (42 + 43 + 44) of the relevant anchor backbones.
#
# Decisive cells being multi-seeded:
#   1. o1→o2 TabPFN: C1 vs D3 (D3 wins by +0.026)
#   2. c1→c2 TabICL: C1 vs D1 (D1 wins by +0.050)
#   3. c2→c1 TabICL: C1 vs D3 (D3 wins by +0.018)
#
# Total: 3 cells x 2 backbones (winner + runner-up) x 3 seeds = 18 eval runs.
# Wall time: ~5 min/eval = ~1.5 GPU-hours.
#
# Pre-req: run_smpnn_multiseed_backbones.sh must have completed (or at
# least the decisive-scope subset). Skip-if-complete handles partial state.
#
# Each cell tees to its own log at logs/smpnn-multiseed-eval/<TAG>.log so
# output is durable even without a wrapping tee.
#
# Usage:
#   ./run_smpnn_multiseed_eval.sh
#       Runs all 18 evals; skips ones whose log already has 6 test_metric lines.
#
#   FORCE=1 ./run_smpnn_multiseed_eval.sh
#       Re-runs even completed evals.
#
#   RUN_ONLY="..." ./...
#       Filter to specific cell tags.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
FORCE=${FORCE:-0}
RUN_ONLY=${RUN_ONLY:-all}

EVAL_LOGDIR="logs/smpnn-multiseed-eval"
mkdir -p "$EVAL_LOGDIR"

# Each entry: SEED|DIRECTION|SRC_TAG|HEAD|TGT_FAMILY|CKPT|BACKBONE_FLAGS
# Seed 42 entries point to the existing checkpoints (smpnn-depth-*,
# smpnn-ablation-*, smpnn-xtask-*). Seed 43/44 entries point to the new
# smpnn-multiseed-* checkpoints from run_smpnn_multiseed_backbones.sh.
CASES=(
    # ─── DECISIVE CELL 1: o1->o2 TabPFN (D3 vs C1) ───
    # Seed 42 (existing): trained on others-1
    "42|o1-to-o2|c1-vanilla-4|tabpfn|others-2|checkpoints/smpnn-depth-vanilla-4/best_checkpoint|--num_mp 4"
    "42|o1-to-o2|d3-alpha-1e-2|tabpfn|others-2|checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # Seed 43 (new)
    "43|o1-to-o2|c1-vanilla-4|tabpfn|others-2|checkpoints/smpnn-multiseed-s43-others-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "43|o1-to-o2|d3-alpha-1e-2|tabpfn|others-2|checkpoints/smpnn-multiseed-s43-others-1-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # Seed 44 (new)
    "44|o1-to-o2|c1-vanilla-4|tabpfn|others-2|checkpoints/smpnn-multiseed-s44-others-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "44|o1-to-o2|d3-alpha-1e-2|tabpfn|others-2|checkpoints/smpnn-multiseed-s44-others-1-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"

    # ─── DECISIVE CELL 2: c1->c2 TabICL (D1 vs C1) ───
    # Seed 42 (existing): trained on commerce-1
    "42|c1-to-c2|c1-vanilla-4|tabicl|commerce-2|checkpoints/smpnn-xtask-c1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "42|c1-to-c2|d1-smpnn-6|tabicl|commerce-2|checkpoints/smpnn-xtask-c1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    # Seed 43 (new)
    "43|c1-to-c2|c1-vanilla-4|tabicl|commerce-2|checkpoints/smpnn-multiseed-s43-commerce-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "43|c1-to-c2|d1-smpnn-6|tabicl|commerce-2|checkpoints/smpnn-multiseed-s43-commerce-1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-6"
    # Seed 44 (new)
    "44|c1-to-c2|c1-vanilla-4|tabicl|commerce-2|checkpoints/smpnn-multiseed-s44-commerce-1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "44|c1-to-c2|d1-smpnn-6|tabicl|commerce-2|checkpoints/smpnn-multiseed-s44-commerce-1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-6"

    # ─── DECISIVE CELL 3: c2->c1 TabICL (D3 vs C1) ───
    # Seed 42 (existing): trained on commerce-2
    "42|c2-to-c1|c1-vanilla-4|tabicl|commerce-1|checkpoints/smpnn-xtask-c2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "42|c2-to-c1|d3-alpha-1e-2|tabicl|commerce-1|checkpoints/smpnn-xtask-c2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # Seed 43 (new)
    "43|c2-to-c1|c1-vanilla-4|tabicl|commerce-1|checkpoints/smpnn-multiseed-s43-commerce-2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "43|c2-to-c1|d3-alpha-1e-2|tabicl|commerce-1|checkpoints/smpnn-multiseed-s43-commerce-2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    # Seed 44 (new)
    "44|c2-to-c1|c1-vanilla-4|tabicl|commerce-1|checkpoints/smpnn-multiseed-s44-commerce-2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "44|c2-to-c1|d3-alpha-1e-2|tabicl|commerce-1|checkpoints/smpnn-multiseed-s44-commerce-2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
)

DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

cell_is_complete() {
    local cell_log=$1
    [ -f "$cell_log" ] || return 1
    local n
    n=$(grep -c "test_metric/" "$cell_log" 2>/dev/null || echo 0)
    [ "$n" -ge 6 ]
}

eval_one() {
    local SEED=$1
    local DIR=$2
    local SRC_TAG=$3
    local HEAD=$4
    local TGT_FAMILY=$5
    local CKPT=$6
    local BACKBONE_FLAGS=$7

    local RUN_TAG="s${SEED}-${DIR}-${SRC_TAG}-${HEAD}-noproj"
    local CELL_LOG="${EVAL_LOGDIR}/${RUN_TAG}.log"

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

    if [ "$FORCE" = "0" ] && cell_is_complete "$CELL_LOG"; then
        local nlines
        nlines=$(grep -c "test_metric/" "$CELL_LOG")
        echo ">>> Skipping $RUN_TAG (already complete: $nlines test_metric lines)"
        return
    fi

    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT (cell: $RUN_TAG)"
        echo "       run run_smpnn_multiseed_backbones.sh first to train seed-${SEED} backbones"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    seed:      $SEED"
    echo "    direction: $DIR    backbone: $SRC_TAG    head: $HEAD"
    echo "    target:    $TGT_FAMILY"
    echo "    ckpt:      $CKPT"
    echo "    log:       $CELL_LOG"
    echo "================================================"

    local EXTRA_FLAGS
    if [ "$HEAD" = "tabpfn" ]; then
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 30000 --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache --tabpfn_inference_precision autocast --tabpfn_inference_config {\"MAX_NUMBER_OF_SAMPLES\":50000,\"MAX_NUMBER_OF_FEATURES\":600}"
    else
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 10000 --tabicl_checkpoint_version v2"
    fi

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-multiseed-eval-${RUN_TAG} smpnn-multiseed-eval-${RUN_TAG} \
        --head $HEAD \
        --tasks rel-f1-driver-position \
        --eval_tasks $TGT_FAMILY \
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
        --savepath checkpoints/smpnn-multiseed-eval-${RUN_TAG} \
        2>&1 | tee "$CELL_LOG"

    local nlines
    nlines=$(grep -c "test_metric/" "$CELL_LOG" 2>/dev/null || echo 0)
    if [ "$nlines" -ge 6 ]; then
        echo ">>> $RUN_TAG: OK ($nlines test_metric lines captured)"
    else
        echo ">>> $RUN_TAG: WARNING only $nlines test_metric lines (expected 6)"
    fi
}

echo "=========================================="
echo "SMPNN multi-seed cross-task eval"
echo "  Cells: 3 decisive (o1->o2 TabPFN, c1->c2 TabICL, c2->c1 TabICL)"
echo "  Backbones per cell: winner + vanilla runner-up = 2"
echo "  Seeds: 42 (existing), 43, 44 (new)"
echo "  Total: 3 x 2 x 3 = 18 eval runs"
echo "  FORCE = $FORCE    RUN_ONLY = $RUN_ONLY"
echo "  Per-cell logs: ${EVAL_LOGDIR}/"
echo "=========================================="

for entry in "${CASES[@]}"; do
    IFS='|' read -r SEED DIR SRC_TAG HEAD TGT_FAMILY CKPT BACKBONE_FLAGS <<< "$entry"
    eval_one "$SEED" "$DIR" "$SRC_TAG" "$HEAD" "$TGT_FAMILY" "$CKPT" "$BACKBONE_FLAGS"
done

echo ""
echo "=========================================="
echo "Multi-seed eval complete. Summarise the 3 decisive cells:"
echo ""
echo "  python collect_smpnn_results.py \\"
echo "      --logs ${EVAL_LOGDIR}/*.log \\"
echo "      --out smpnn_multiseed_results.csv"
echo ""
echo "Then compute mean ± std per (direction, head, backbone) across seeds."
echo "=========================================="
