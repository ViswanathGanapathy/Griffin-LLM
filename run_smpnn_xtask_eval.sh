#!/bin/bash
# Cross-task transfer evaluation: 4 directions x 3 anchor backbones x 2 ICL
# heads (TabPFN v3 ZS, TabICL v2 ZS) all with --no_icl_projection.
#
# Total: 24 eval runs. Each ~3-5 min -> ~1.5-2 GPU-hours total.
#
# Pre-req:
#   - others-1 backbones from run_smpnn_ablations.sh / run_smpnn_depth_native.sh
#     (already trained -- used for o1->o2 direction)
#   - others-2, commerce-1, commerce-2 backbones from run_smpnn_xtask_backbones.sh
#
# Usage:
#   ./run_smpnn_xtask_eval.sh 2>&1 | tee logs/smpnn-xtask-eval.log
#
# Output: logs/smpnn-xtask-<direction>-<backbone>-<head>-noproj/
#         e.g. logs/smpnn-xtask-o1-to-o2-d3-alpha-1e-2-tabpfn-noproj/

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Each entry: DIRECTION | SRC_TAG | TGT_FAMILY | CKPT | BACKBONE_FLAGS
# SRC_TAG is the human-readable label for the source backbone (used in
# the output log dir name); CKPT is where to find the trained weights.
# others-1 source uses the existing smpnn-depth-/smpnn-ablation- ckpts;
# the other 3 sources use the smpnn-xtask- ckpts.
CASES=(
    # o1 -> o2 (use existing others-1 backbones)
    "o1-to-o2|c1-vanilla-4|others-2|checkpoints/smpnn-depth-vanilla-4/best_checkpoint|--num_mp 4"
    "o1-to-o2|d1-smpnn-6|others-2|checkpoints/smpnn-depth-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "o1-to-o2|d3-alpha-1e-2|others-2|checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"

    # o2 -> o1
    "o2-to-o1|c1-vanilla-4|others-1|checkpoints/smpnn-xtask-o2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "o2-to-o1|d1-smpnn-6|others-1|checkpoints/smpnn-xtask-o2-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "o2-to-o1|d3-alpha-1e-2|others-1|checkpoints/smpnn-xtask-o2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"

    # c1 -> c2
    "c1-to-c2|c1-vanilla-4|commerce-2|checkpoints/smpnn-xtask-c1-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "c1-to-c2|d1-smpnn-6|commerce-2|checkpoints/smpnn-xtask-c1-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "c1-to-c2|d3-alpha-1e-2|commerce-2|checkpoints/smpnn-xtask-c1-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"

    # c2 -> c1
    "c2-to-c1|c1-vanilla-4|commerce-1|checkpoints/smpnn-xtask-c2-c1-vanilla-4/best_checkpoint|--num_mp 4"
    "c2-to-c1|d1-smpnn-6|commerce-1|checkpoints/smpnn-xtask-c2-d1-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "c2-to-c1|d3-alpha-1e-2|commerce-1|checkpoints/smpnn-xtask-c2-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
)

DATA_FLAGS="--hop 2 --fanout 20 --fewshotfanout 3"

eval_one() {
    local DIR=$1            # direction tag (o1-to-o2 etc.)
    local SRC_TAG=$2        # backbone tag
    local HEAD=$3           # tabpfn | tabicl
    local TGT_FAMILY=$4     # task family for --eval_tasks
    local CKPT=$5
    local BACKBONE_FLAGS=$6
    local EXTRA_FLAGS=$7

    # We pick a regression task in the source family as the (vestigial)
    # --tasks arg -- it's never actually used because --probe_epochs 0.
    local PROBE_TASK="rel-f1-driver-position"  # exists in joint-v65; harmless placeholder

    local RUN_TAG="${DIR}-${SRC_TAG}-${HEAD}-noproj"
    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    direction: $DIR    backbone: $SRC_TAG    head: $HEAD"
    echo "    target:    $TGT_FAMILY"
    echo "    ckpt:      $CKPT"
    echo "================================================"

    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT"
        return
    fi

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/smpnn-xtask-${RUN_TAG} smpnn-xtask-${RUN_TAG} \
        --head $HEAD \
        --tasks $PROBE_TASK \
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
        --savepath checkpoints/smpnn-xtask-${RUN_TAG}
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

for entry in "${CASES[@]}"; do
    IFS='|' read -r DIR SRC_TAG TGT_FAMILY CKPT BACKBONE_FLAGS <<< "$entry"
    eval_one "$DIR" "$SRC_TAG" "tabpfn" "$TGT_FAMILY" "$CKPT" "$BACKBONE_FLAGS" "${TABPFN_FLAGS[*]}"
    eval_one "$DIR" "$SRC_TAG" "tabicl" "$TGT_FAMILY" "$CKPT" "$BACKBONE_FLAGS" "${TABICL_FLAGS[*]}"
done

echo ""
echo "================================================"
echo "Cross-task eval complete. Summarise via:"
echo "  python collect_smpnn_results.py \\"
echo "      --logs logs/smpnn-xtask-eval.log \\"
echo "      --out smpnn_xtask_results.csv"
echo "================================================"
