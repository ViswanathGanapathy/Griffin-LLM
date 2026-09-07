#!/bin/bash
# LoG 2026 rebuttal: the two missing Table-1 cells (the two N/A entries).
#
#   ABLATION 1 — V6 control on o2->o1 (both heads).
#     The reviewer-named gap (Appendix E, exp. 1). Trains vanilla --num_mp 6
#     on others-2 (anchor c3-vanilla-6, seeds 42-46), then evaluates on
#     others-1 under TabPFN v3 AND TabICL v2. Decides Section 4.3: if V6 is
#     at parity with V4 here, the depth-tracking account is wrong and must
#     be replaced by a directional one.
#
#   ABLATION 2 — SMPNN-6 alpha_init=1e-6 on o1->o2 (TabPFN).
#     The second Table-1 N/A (Appendix E, exp. 5 / Appendix D). Trains
#     d1-smpnn-6 on others-1 where a seed's checkpoint is missing, then
#     evaluates on others-2 under TabPFN v3. Completes the alpha ablation
#     on the one direction where SMPNN wins.
#
# Consistency: training block copied from run_smpnn_multiseed_backbones.sh
# and eval block from run_smpnn_multiseed_eval.sh VERBATIM (same recipe,
# same head versions: TabPFN v3 ctx 30000, TabICL v2 ctx 10000, noproj,
# probe_epochs 0), so the new cells are comparable with the paper's
# existing n=5 cells. Do not change flags here without changing them there.
#
# Checkpoint reuse: every (train) step is skip-if-exists, and known legacy
# paths are probed first — d1-smpnn-6 on others-1 may already exist for
# some seeds (multiseed decisive scope trained 43/44; seed 42 may exist as
# checkpoints/smpnn-ablation-d1-smpnn-6). V6 on others-2 may exist from
# the in-progress o2->o1 depth sweep. Whatever exists is reused; only the
# gaps are trained.
#
# Cost envelope: worst case (nothing exists) 10 trainings x ~3-4 GPU-h
# + 15 evals x ~5-10 min. Best case (all checkpoints exist) ~2 GPU-h of
# evals only — the Appendix E estimate.
#
# Usage:
#   ./run_log_rebuttal_ablations.sh                      # everything
#   PHASE=train ./run_log_rebuttal_ablations.sh          # trainings only
#   PHASE=eval  ./run_log_rebuttal_ablations.sh          # evals only
#   SEEDS="45 46" ./run_log_rebuttal_ablations.sh        # subset of seeds
#   ABLATION=1 ./run_log_rebuttal_ablations.sh           # V6 o2->o1 only
#   ABLATION=2 ./run_log_rebuttal_ablations.sh           # alpha arm only
#   FORCE=1 ...                                          # re-run complete cells
#
# Afterwards:
#   python analyze_rebuttal_ablations.py                 # Table-1/B.1/B.4 numbers

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SEEDS=${SEEDS:-"42 43 44 45 46"}
PHASE=${PHASE:-both}
ABLATION=${ABLATION:-both}
FORCE=${FORCE:-0}

TRAIN_LOGDIR="logs/smpnn-multiseed"
EVAL_LOGDIR="logs/log-rebuttal-eval"
mkdir -p "$TRAIN_LOGDIR" "$EVAL_LOGDIR"

# ── checkpoint resolution ────────────────────────────────────────────────
# Echo the first existing best_checkpoint among candidates; else echo the
# canonical multiseed path (the one training would create).
resolve_ckpt() {
    local CANONICAL=$1; shift
    for c in "$@" "$CANONICAL"; do
        if [ -d "${c}/best_checkpoint" ]; then echo "$c"; return; fi
    done
    echo "$CANONICAL"
}

# ── training (verbatim recipe from run_smpnn_multiseed_backbones.sh) ────
train_one() {
    local SEED=$1 SRC=$2 ANCHOR=$3 FLAGS=$4
    local TAG="s${SEED}-${SRC}-${ANCHOR}"
    local SAVE_PATH="checkpoints/smpnn-multiseed-${TAG}"
    local CELL_LOG="${TRAIN_LOGDIR}/${TAG}.log"

    if [ "$FORCE" = "0" ] && [ -d "${SAVE_PATH}/best_checkpoint" ]; then
        echo ">>> Skipping train $TAG (best_checkpoint exists)"
        return
    fi
    echo ""
    echo "================================================"
    echo ">>> Training smpnn-multiseed-${TAG}"
    echo "    seed: $SEED  source: $SRC  anchor: $ANCHOR"
    echo "    flags: $FLAGS"
    echo "    save:  $SAVE_PATH   log: $CELL_LOG"
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

# ── eval (verbatim from run_smpnn_multiseed_eval.sh) ────────────────────
DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

cell_is_complete() {
    local cell_log=$1
    [ -f "$cell_log" ] || return 1
    local n
    n=$(grep -c "test_metric/" "$cell_log" 2>/dev/null || echo 0)
    [ "$n" -ge 6 ]
}

eval_one() {
    local SEED=$1 DIR=$2 SRC_TAG=$3 HEAD=$4 TGT_FAMILY=$5 CKPT_DIR=$6 BACKBONE_FLAGS=$7
    local CKPT="${CKPT_DIR}/best_checkpoint"
    local RUN_TAG="s${SEED}-${DIR}-${SRC_TAG}-${HEAD}-noproj"
    local CELL_LOG="${EVAL_LOGDIR}/${RUN_TAG}.log"

    if [ "$FORCE" = "0" ] && cell_is_complete "$CELL_LOG"; then
        echo ">>> Skipping eval $RUN_TAG (already complete)"
        return
    fi
    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT (cell: $RUN_TAG)"
        echo "       run PHASE=train first for this seed"
        return
    fi
    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    ckpt: $CKPT   eval on: $TGT_FAMILY"
    echo "================================================"

    local EXTRA_FLAGS
    if [ "$HEAD" = "tabpfn" ]; then
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 30000 --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache --tabpfn_inference_precision autocast --tabpfn_inference_config {\"MAX_NUMBER_OF_SAMPLES\":50000,\"MAX_NUMBER_OF_FEATURES\":600}"
    else
        EXTRA_FLAGS="--icl_n_estimators 8 --icl_max_context 10000 --tabicl_checkpoint_version v2"
    fi

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/log-rebuttal-eval-${RUN_TAG} log-rebuttal-eval-${RUN_TAG} \
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
        --savepath checkpoints/log-rebuttal-eval-${RUN_TAG} \
        2>&1 | tee "$CELL_LOG"
}

echo "=========================================="
echo "LoG rebuttal ablations   PHASE=$PHASE  ABLATION=$ABLATION"
echo "  seeds: $SEEDS   FORCE=$FORCE"
echo "  A1: V6 (c3-vanilla-6) on o2->o1, TabPFN v3 + TabICL v2"
echo "  A2: SMPNN-6 alpha=1e-6 (d1-smpnn-6) on o1->o2, TabPFN v3"
echo "=========================================="

for SEED in $SEEDS; do
    # ── ABLATION 1: V6 on o2->o1 ──
    if [ "$ABLATION" = "both" ] || [ "$ABLATION" = "1" ]; then
        A1_CKPT=$(resolve_ckpt "checkpoints/smpnn-multiseed-s${SEED}-others-2-c3-vanilla-6" \
                               "checkpoints/smpnn-depth-o2-vanilla-6-s${SEED}")
        if [ "$PHASE" != "eval" ] && [ ! -d "${A1_CKPT}/best_checkpoint" ]; then
            train_one "$SEED" "others-2" "c3-vanilla-6" "--num_mp 6"
            A1_CKPT="checkpoints/smpnn-multiseed-s${SEED}-others-2-c3-vanilla-6"
        fi
        if [ "$PHASE" != "train" ]; then
            eval_one "$SEED" "o2-to-o1" "c3-vanilla-6" "tabpfn" "others-1" "$A1_CKPT" "--num_mp 6"
            eval_one "$SEED" "o2-to-o1" "c3-vanilla-6" "tabicl" "others-1" "$A1_CKPT" "--num_mp 6"
        fi
    fi

    # ── ABLATION 2: SMPNN-6 alpha=1e-6 on o1->o2 ──
    if [ "$ABLATION" = "both" ] || [ "$ABLATION" = "2" ]; then
        A2_CKPT=$(resolve_ckpt "checkpoints/smpnn-multiseed-s${SEED}-others-1-d1-smpnn-6" \
                               "checkpoints/smpnn-ablation-d1-smpnn-6-s${SEED}" \
                               $([ "$SEED" = "42" ] && echo "checkpoints/smpnn-ablation-d1-smpnn-6"))
        if [ "$PHASE" != "eval" ] && [ ! -d "${A2_CKPT}/best_checkpoint" ]; then
            train_one "$SEED" "others-1" "d1-smpnn-6" "--num_mp 6 --use_smpnn --alpha_init 1e-6"
            A2_CKPT="checkpoints/smpnn-multiseed-s${SEED}-others-1-d1-smpnn-6"
        fi
        if [ "$PHASE" != "train" ]; then
            eval_one "$SEED" "o1-to-o2" "d1-smpnn-6" "tabpfn" "others-2" "$A2_CKPT" "--num_mp 6 --use_smpnn --alpha_init 1e-6"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Done. Fill the two Table-1 N/A cells with:"
echo "    python analyze_rebuttal_ablations.py --logs '${EVAL_LOGDIR}/*.log'"
echo "Add existing V4/SMPNN eval logs via extra --logs globs to get the"
echo "paired/Welch comparisons in the same report."
echo "=========================================="
