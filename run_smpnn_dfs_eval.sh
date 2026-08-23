#!/bin/bash
# Evaluate DFS-cell checkpoints on others-2 (o1->o2 transfer) + optional
# in-distribution others-1, with TabPFN v3. Companion to run_smpnn_dfs.sh.
#
# CRITICAL: each checkpoint is evaluated with the SAME dfs flags it was
# trained with — the column set is part of the input contract.
#
# Baseline note: for E0/vanilla you can skip re-training and reuse the
# existing multiseed checkpoints (same config: vanilla-4, others-1):
#   checkpoints/smpnn-multiseed-s{42,43,44}-others-1-c1-vanilla-4/best_checkpoint
# Set REUSE_E0=1 (default) to use those for the vanilla E0 rows.
#
# Usage:
#   ./run_smpnn_dfs_eval.sh                     # all cells x 3 seeds, o1->o2
#   EVAL_FAMILY=others-1 ./run_smpnn_dfs_eval.sh  # in-dist instead
#   SEEDS="42" CELLS="E0 E2" BACKBONES="vanilla" ./run_smpnn_dfs_eval.sh
#   FORCE=1 ./run_smpnn_dfs_eval.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"42 43 44"}
CELLS=${CELLS:-"E0 E1 E2 E3"}
BACKBONES=${BACKBONES:-"smpnn vanilla"}
EVAL_FAMILY=${EVAL_FAMILY:-others-2}
# Family the checkpoints were TRAINED on (must match run_smpnn_dfs.sh's
# TRAIN_FAMILY; others-1 keeps the historical unprefixed tag).
TRAIN_FAMILY=${TRAIN_FAMILY:-others-1}
CKPT_PREFIX=""
[ "$TRAIN_FAMILY" != "others-1" ] && CKPT_PREFIX="${TRAIN_FAMILY}-"
TABPFN_VERSION=${TABPFN_VERSION:-v3}   # v2.5 | v2.6 | v3 (see DFS_ABLATION_PLAN.md)
REUSE_E0=${REUSE_E0:-1}
FORCE=${FORCE:-0}

EVAL_LOGDIR="logs/smpnn-dfs-eval"
mkdir -p "$EVAL_LOGDIR"

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

cell_is_complete() {
    local n
    n=$(grep -c "test_metric/" "$1" 2>/dev/null || echo 0)
    [ "$n" -ge 6 ]
}

eval_one() {
    local CELL=$1 DFS_FS=$2 DFS_ROOT=$3 BACKBONE=$4 SEED=$5
    local TAG="${CKPT_PREFIX}s${SEED}-${CELL}-${BACKBONE}-$(echo $EVAL_FAMILY | tr -d '-')-tabpfn$(echo $TABPFN_VERSION | tr -d '.')"
    local CELL_LOG="${EVAL_LOGDIR}/${TAG}.log"

    local found=0
    for c in $CELLS; do [ "$c" = "$CELL" ] && found=1; done
    [ "$found" = "0" ] && return
    found=0
    for b in $BACKBONES; do [ "$b" = "$BACKBONE" ] && found=1; done
    [ "$found" = "0" ] && return

    # Checkpoint resolution (E0/vanilla may reuse the multiseed baseline)
    local CKPT="checkpoints/smpnn-dfs-${CKPT_PREFIX}s${SEED}-${CELL}-${BACKBONE}/best_checkpoint"
    if [ "$CELL" = "E0" ] && [ "$BACKBONE" = "vanilla" ] && [ "$REUSE_E0" = "1" ] && [ ! -d "$CKPT" ]; then
        CKPT="checkpoints/smpnn-multiseed-s${SEED}-others-1-c1-vanilla-4/best_checkpoint"
    fi
    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT (cell $TAG) — train via run_smpnn_dfs.sh"
        return
    fi

    if [ "$FORCE" = "0" ] && cell_is_complete "$CELL_LOG"; then
        echo ">>> Skipping $TAG (already complete)"
        return
    fi

    echo ""
    echo "================================================"
    echo ">>> $TAG   ckpt: $CKPT"
    echo "    dfs_fewshot=$DFS_FS dfs_root=$DFS_ROOT  eval on: $EVAL_FAMILY (test split)"
    echo "================================================"

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-dfs-eval-${TAG} smpnn-dfs-eval-${TAG} \
        --head tabpfn \
        --tasks rel-f1-driver-position \
        --eval_tasks ${EVAL_FAMILY} \
        --loadpath $CKPT \
        --seed $SEED \
        $(backbone_flags $BACKBONE) \
        --dfs_fewshot_depth ${DFS_FS} --dfs_root_depth ${DFS_ROOT} \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --no_icl_projection --probe_epochs 0 \
        --hiddim 512 --use_rev True --use_gate True \
        --batchsize 256 \
        --output_mlp_dim 1 --no_target_normalize \
        --icl_n_estimators 8 --icl_max_context 30000 \
        --tabpfn_version ${TABPFN_VERSION} --tabpfn_fit_mode fit_with_cache \
        --tabpfn_inference_precision autocast \
        --tabpfn_inference_config {\"MAX_NUMBER_OF_SAMPLES\":50000,\"MAX_NUMBER_OF_FEATURES\":600} \
        --savepath checkpoints/smpnn-dfs-eval-${TAG} \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "DFS eval: cells [$CELLS] x backbones [$BACKBONES] x seeds [$SEEDS]"
echo "  eval family: $EVAL_FAMILY (test split)   REUSE_E0=$REUSE_E0"
echo "  TabPFN version: $TABPFN_VERSION"
echo "  logs: ${EVAL_LOGDIR}/"
echo "=========================================="

for SEED in $SEEDS; do
    for BACKBONE in $BACKBONES; do
        for entry in "${CELL_DEFS[@]}"; do
            IFS='|' read -r CELL DFS_FS DFS_ROOT <<< "$entry"
            eval_one "$CELL" "$DFS_FS" "$DFS_ROOT" "$BACKBONE" "$SEED"
        done
    done
done

echo ""
echo "Aggregate:  python collect_smpnn_results.py --logs ${EVAL_LOGDIR}/*.log --out smpnn_dfs_eval_results.csv"
echo "Compare each cell vs E0 on BOTH mean and seed-std per task."
