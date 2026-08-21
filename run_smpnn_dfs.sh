#!/bin/bash
# DFS + MPNN hybrid experiments (RDBLearn-style Deep Feature Synthesis
# injected into Griffin). See sprints/v2/DFS_GRIFFIN_DESIGN.md.
#
# Pre-req (one-time, CPU-ok, ~minutes for small families / ~hours for full):
#   python dataconverterdfs.py datasets/joint-v65
#
# Matrix (per seed):
#   E0 baseline      dfs_fewshot=0 dfs_root=0   reproduces existing numbers
#   E1 fewshot-dfs1  dfs_fewshot=1 dfs_root=0   Step 1: leaf enrichment alone
#   E2 root-dfs2     dfs_fewshot=0 dfs_root=2   Step 2: root enrichment alone
#   E3 both          dfs_fewshot=1 dfs_root=2   additivity
#
# Backbones: SMPNN-6 alpha=1e-2 (current o1->o2 best) and Vanilla-4
# (baseline). Source family: others-1. 3 seeds to start.
# Total: 4 cells x 2 backbones x 3 seeds = 24 trainings (~90 GPU-h).
# Trim with BACKBONES / CELLS / SEEDS env vars.
#
# Registered predictions (from the design doc, judge honestly afterwards):
#   * E1 should mainly help label-context-driven tasks (churn-style).
#   * E2's clearest signature is REDUCED SEED-STD vs E0 (DFS features are
#     deterministic — they attack sampling variance at the source).
#   * If E3 ~= E2, root aggregates subsume leaf enrichment.
#
# Usage:
#   ./run_smpnn_dfs.sh
#   SEEDS="42" CELLS="E0 E2" BACKBONES="smpnn" ./run_smpnn_dfs.sh
#   FORCE=1 ./run_smpnn_dfs.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"42 43 44"}
CELLS=${CELLS:-"E0 E1 E2 E3"}
BACKBONES=${BACKBONES:-"smpnn vanilla"}
FORCE=${FORCE:-0}

LOGDIR="logs/smpnn-dfs"
mkdir -p "$LOGDIR"

if [ ! -f "datasets/joint-v65/dfs/metadfs.yaml" ]; then
    echo "ERROR: DFS artifacts not found. Run first:"
    echo "    python dataconverterdfs.py datasets/joint-v65"
    exit 1
fi

# CELL|DFS_FEWSHOT|DFS_ROOT
CELL_DEFS=(
    "E0|0|0"
    "E1|1|0"
    "E2|0|2"
    "E3|1|2"
)

backbone_flags() {
    if [ "$1" = "smpnn" ]; then
        echo "--num_mp 6 --use_smpnn --alpha_init 1e-2"
    else
        echo "--num_mp 4"
    fi
}

run_one() {
    local CELL=$1 DFS_FS=$2 DFS_ROOT=$3 BACKBONE=$4 SEED=$5
    local TAG="s${SEED}-${CELL}-${BACKBONE}"
    local SAVE_PATH="checkpoints/smpnn-dfs-${TAG}"
    local CELL_LOG="${LOGDIR}/${TAG}.log"

    local found=0
    for c in $CELLS; do [ "$c" = "$CELL" ] && found=1; done
    [ "$found" = "0" ] && return
    found=0
    for b in $BACKBONES; do [ "$b" = "$BACKBONE" ] && found=1; done
    [ "$found" = "0" ] && return

    if [ "$FORCE" = "0" ] && [ -d "${SAVE_PATH}/best_checkpoint" ]; then
        echo ">>> Skipping $TAG (best_checkpoint exists)"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> DFS cell $TAG"
    echo "    dfs_fewshot_depth: $DFS_FS   dfs_root_depth: $DFS_ROOT"
    echo "    backbone: $BACKBONE   seed: $SEED"
    echo "    save: $SAVE_PATH    log: $CELL_LOG"
    echo "================================================"

    PYTHONUNBUFFERED=1 accelerate launch hmaintask_combine.py \
        datasets/joint-v65 logs/smpnn-dfs-${TAG} smpnn-dfs-${TAG} \
        --tasks others-1 \
        --seed ${SEED} \
        $(backbone_flags $BACKBONE) \
        --dfs_fewshot_depth ${DFS_FS} --dfs_root_depth ${DFS_ROOT} \
        --hiddim 512 --use_rev True --use_gate True \
        --maxepoch 20 --batchsize 256 \
        --lr 3e-4 --wd 4e-4 \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --eval_per_epoch 1 \
        --savepath ${SAVE_PATH} \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "DFS + MPNN hybrid training"
echo "  CELLS=$CELLS  BACKBONES=$BACKBONES  SEEDS=$SEEDS"
echo "  Per-cell logs: ${LOGDIR}/"
echo "=========================================="

for SEED in $SEEDS; do
    for BACKBONE in $BACKBONES; do
        for entry in "${CELL_DEFS[@]}"; do
            IFS='|' read -r CELL DFS_FS DFS_ROOT <<< "$entry"
            run_one "$CELL" "$DFS_FS" "$DFS_ROOT" "$BACKBONE" "$SEED"
        done
    done
done

echo ""
echo "=========================================="
echo "Done. IMPORTANT: eval each checkpoint with the SAME dfs flags it"
echo "was trained with, e.g. for E2:"
echo "  python hmaintask_combine_llm.py ... --head tabpfn \\"
echo "      --loadpath checkpoints/smpnn-dfs-s42-E2-smpnn/best_checkpoint \\"
echo "      --dfs_root_depth 2 --dfs_fewshot_depth 0 ..."
echo "Compare per-cell mean AND seed-std against E0 (variance is the"
echo "registered primary signature for E2 — see DFS_GRIFFIN_DESIGN.md §6)."
echo "=========================================="
