#!/bin/bash
# ICL-head evaluation across all SMPNN ablation backbones.
#
# For each backbone produced by run_smpnn_ablations.sh (+ the existing depth-
# scan backbones already in checkpoints/smpnn-depth-*), runs:
#   - TabPFN v3 zero-shot, NO projection (raw 512-d Griffin embeddings)
#   - TabICL v2 zero-shot, NO projection (capped at 10K context)
#
# All in-distribution: --eval_tasks others-1. --tasks is vestigial when
# --probe_epochs 0 + --no_icl_projection (no probe phase runs), so we set
# it to rel-f1-driver-position purely as a placeholder.
#
# Pre-req: backbone checkpoints under checkpoints/smpnn-ablation-<tag>/best_checkpoint
#          and checkpoints/smpnn-depth-<tag>/best_checkpoint must exist.
#
# Wall-time per eval: ~3-5 min (no training, just ICL inference). Total for
# 14 backbone-by-head combinations: ~1 GPU-hour.
#
# Usage:
#   ./run_smpnn_ablations_icl_eval.sh 2>&1 | tee logs/smpnn-ablations-icl-eval.log
# Or split per GPU by commenting out variants below.

set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Each entry: TAG | CKPT_DIR | BACKBONE_FLAGS
# BACKBONE_FLAGS must match the training-time flags exactly so the
# checkpoint state-dict loads cleanly.
BACKBONES=(
    # Existing depth-scan checkpoints (run_smpnn_depth_native.sh).
    "depth-vanilla-4|checkpoints/smpnn-depth-vanilla-4/best_checkpoint|--num_mp 4"
    "depth-smpnn-4|checkpoints/smpnn-depth-smpnn-4/best_checkpoint|--num_mp 4 --use_smpnn"
    "depth-smpnn-6|checkpoints/smpnn-depth-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn"
    "depth-smpnn-8|checkpoints/smpnn-depth-smpnn-8/best_checkpoint|--num_mp 8 --use_smpnn"
    # New ablation checkpoints (run_smpnn_ablations.sh).
    "c3-vanilla-6|checkpoints/smpnn-ablation-c3-vanilla-6/best_checkpoint|--num_mp 6"
    "a2-no-alpha|checkpoints/smpnn-ablation-a2-no-alpha/best_checkpoint|--num_mp 6 --use_smpnn --use_alpha False"
    "a3-no-ff|checkpoints/smpnn-ablation-a3-no-ff/best_checkpoint|--num_mp 6 --use_smpnn --use_ff False"
    "a4-no-gnn-ln|checkpoints/smpnn-ablation-a4-no-gnn-ln/best_checkpoint|--num_mp 6 --use_smpnn --use_gnn_ln False"
    "d2-alpha-1e-4|checkpoints/smpnn-ablation-d2-alpha-1e-4/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-4"
    "d3-alpha-1e-2|checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2"
    "b1-attn-1h|checkpoints/smpnn-ablation-b1-attn-1h/best_checkpoint|--num_mp 6 --use_smpnn --use_attention True --num_heads 1"
)

eval_one() {
    local TAG=$1            # backbone tag
    local HEAD=$2           # tabpfn | tabicl
    local CKPT=$3           # backbone checkpoint path
    local BACKBONE_FLAGS=$4 # flags that recreate the backbone for ckpt loading
    local EXTRA_FLAGS=$5    # head-specific knobs

    local RUN_TAG="${TAG}-${HEAD}-noproj"
    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    ckpt:   $CKPT"
    echo "    head:   $HEAD"
    echo "    bbflags:$BACKBONE_FLAGS"
    echo "================================================"

    if [ ! -d "$CKPT" ]; then
        echo "[skip] checkpoint not found: $CKPT"
        return
    fi

    PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
        datasets/joint-v65 logs/${RUN_TAG} ${RUN_TAG} \
        --head $HEAD \
        --tasks rel-f1-driver-position \
        --eval_tasks others-1 \
        --loadpath $CKPT \
        $BACKBONE_FLAGS \
        --no_icl_projection \
        --probe_epochs 0 \
        --hiddim 512 --use_rev True --use_gate True \
        --hop 2 --fanout 20 --fewshotfanout 3 \
        --batchsize 256 \
        --output_mlp_dim 1 --no_target_normalize \
        $EXTRA_FLAGS \
        --savepath checkpoints/${RUN_TAG}
}

TABPFN_FLAGS=(
    --icl_n_estimators 8
    --icl_max_context 30000
    --tabpfn_version v3
    --tabpfn_fit_mode fit_with_cache
    --tabpfn_inference_precision autocast
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES": 50000, "MAX_NUMBER_OF_FEATURES": 600}'
)

TABICL_FLAGS=(
    --icl_n_estimators 8
    --icl_max_context 10000
    --tabicl_checkpoint_version v2
)

for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r TAG CKPT BACKBONE_FLAGS <<< "$entry"
    eval_one "$TAG" "tabpfn" "$CKPT" "$BACKBONE_FLAGS" "${TABPFN_FLAGS[*]}"
    eval_one "$TAG" "tabicl" "$CKPT" "$BACKBONE_FLAGS" "${TABICL_FLAGS[*]}"
done

echo ""
echo "================================================"
echo "All ICL evals complete. Grep results:"
echo "  grep -E 'test_metric' logs/*-tabpfn-noproj/std* 2>/dev/null | sort"
echo "  grep -E 'test_metric' logs/*-tabicl-noproj/std* 2>/dev/null | sort"
echo "================================================"
