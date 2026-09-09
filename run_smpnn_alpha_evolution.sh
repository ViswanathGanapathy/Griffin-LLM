#!/bin/bash
# α (learnable residual scale) evolution tracking during SMPNN training.
#
# Purpose: strengthen the mechanistic story in §3.3 and §6.4 of the paper
# draft by observing how α_gnn[i] and α_ff[i] evolve during 20 epochs of
# training. The claim we want to support:
#
#   "α = 1e-2 gives the spectral residual a non-trivial contribution
#    from step 1, which under a warm-started backbone means the FFN
#    sub-block starts refining the already-learned representation
#    immediately. α = 1e-6 must grow α substantially during training."
#
# If we observe:
#   - α = 1e-6 grows substantially during training (e.g., → 1e-3 or higher)
#   - α = 1e-2 remains stable or grows only modestly
# → mechanistic story confirmed.
#
# If we observe both α values converging to similar final values,
# → the story needs revision: the init only affects the warm-up trajectory,
#   not the final behavior.
#
# Design: re-train SMPNN-6 on others-1 with BOTH α initializations
# (1e-6 and 1e-2) across 3 seeds (42, 43, 44). Enable --log_alpha_every 1
# so per-epoch α values are printed to stdout. Post-process the logs
# to extract a CSV of (seed, alpha_init, epoch, layer_i, alpha_gnn,
# alpha_ff) rows and plot the trajectories.
#
# Backbones trained: 2 α values × 3 seeds = 6 trainings.
# Wall time: ~3-4 GPU-hours × 6 = ~20-24 GPU-hours on a single A100.
#
# NOTE: If the existing multiseed checkpoints were trained WITH
# --log_alpha_every set, you can skip re-training and just parse the
# existing logs. Check first:
#   grep -l '\[alpha\]' logs/smpnn-multiseed/*.log | head
#
# Usage:
#   ./run_smpnn_alpha_evolution.sh
#       Runs all 6 trainings; skips ones whose checkpoint already exists.
#
#   FORCE=1 ./run_smpnn_alpha_evolution.sh
#       Re-trains even if checkpoint exists.
#
#   SEEDS="42 43 44" ./run_smpnn_alpha_evolution.sh
#       Override seed set (default: 42 43 44 — 3 seeds is enough for the
#       trajectory claim; the transfer conclusions still need n=5).

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"42 43 44"}
FORCE=${FORCE:-0}

LOGDIR="logs/smpnn-alpha-evolution"
mkdir -p "$LOGDIR"

# (α_INIT_TAG, α_VALUE) pairs
CASES=(
    "d1|1e-6"
    "d3|1e-2"
)

run_one() {
    local SEED=$1
    local ALPHA_TAG=$2
    local ALPHA_VALUE=$3

    local TAG="s${SEED}-${ALPHA_TAG}-alpha-evo"
    local SAVE_PATH="checkpoints/smpnn-alpha-evolution-${TAG}"
    local CELL_LOG="${LOGDIR}/${TAG}.log"

    if [ "$FORCE" = "0" ] && [ -d "${SAVE_PATH}/best_checkpoint" ]; then
        echo ">>> Skipping $TAG (best_checkpoint exists at ${SAVE_PATH}/best_checkpoint)"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> Training α-evolution SMPNN-6 ${TAG}"
    echo "    seed:        $SEED"
    echo "    alpha_init:  $ALPHA_VALUE"
    echo "    save:        $SAVE_PATH"
    echo "    log:         $CELL_LOG"
    echo "================================================"

    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-alpha-evolution-${TAG} smpnn-alpha-evolution-${TAG} \
        --tasks others-1 \
        --seed ${SEED} \
        --num_mp 6 --use_smpnn --alpha_init ${ALPHA_VALUE} \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath ${SAVE_PATH} \
        --log_alpha_every 1 \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "SMPNN α-evolution training"
echo "  α inits: 1e-6 (D1, paper default), 1e-2 (D3, warm-start)"
echo "  SEEDS:   $SEEDS"
echo "  TOTAL:   ${#CASES[@]} × $(echo $SEEDS | wc -w) = $((${#CASES[@]} * $(echo $SEEDS | wc -w))) trainings"
echo "  Per-cell logs: ${LOGDIR}/"
echo "=========================================="

for SEED in $SEEDS; do
    for entry in "${CASES[@]}"; do
        IFS='|' read -r ALPHA_TAG ALPHA_VALUE <<< "$entry"
        run_one "$SEED" "$ALPHA_TAG" "$ALPHA_VALUE"
    done
done

echo ""
echo "=========================================="
echo "Training complete. Extract α trajectories:"
echo ""
echo "  python parse_alpha_evolution.py \\"
echo "      --logs ${LOGDIR}/*.log \\"
echo "      --out smpnn_alpha_evolution.csv"
echo ""
echo "Then plot with:"
echo ""
echo "  python plot_alpha_evolution.py \\"
echo "      --csv smpnn_alpha_evolution.csv \\"
echo "      --out figs/alpha_evolution.png"
echo "=========================================="
