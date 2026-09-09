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

# Each entry: TAG | CKPT_DIR | BACKBONE_FLAGS | DATA_FLAGS
# BACKBONE_FLAGS must match the training-time backbone-construction flags
# exactly so the checkpoint state-dict loads cleanly.
# DATA_FLAGS must match the training-time data-loader config (hop / fanout)
# so the eval-time subgraphs look like what the model saw at training.
DEFAULT_DATA="--hop 2 --fanout 20 --fewshotfanout 3"
HOP3_DATA="--hop 3 --fanout 10 --fewshotfanout 3"

BACKBONES=(
    # Existing depth-scan checkpoints (run_smpnn_depth_native.sh, hop=2).
    "depth-vanilla-4|checkpoints/smpnn-depth-vanilla-4/best_checkpoint|--num_mp 4|$DEFAULT_DATA"
    "depth-smpnn-4|checkpoints/smpnn-depth-smpnn-4/best_checkpoint|--num_mp 4 --use_smpnn|$DEFAULT_DATA"
    "depth-smpnn-6|checkpoints/smpnn-depth-smpnn-6/best_checkpoint|--num_mp 6 --use_smpnn|$DEFAULT_DATA"
    "depth-smpnn-8|checkpoints/smpnn-depth-smpnn-8/best_checkpoint|--num_mp 8 --use_smpnn|$DEFAULT_DATA"
    # New ablation checkpoints (run_smpnn_ablations.sh, hop=2).
    "c3-vanilla-6|checkpoints/smpnn-ablation-c3-vanilla-6/best_checkpoint|--num_mp 6|$DEFAULT_DATA"
    "a2-no-alpha|checkpoints/smpnn-ablation-a2-no-alpha/best_checkpoint|--num_mp 6 --use_smpnn --use_alpha False|$DEFAULT_DATA"
    "a3-no-ff|checkpoints/smpnn-ablation-a3-no-ff/best_checkpoint|--num_mp 6 --use_smpnn --use_ff False|$DEFAULT_DATA"
    "a4-no-gnn-ln|checkpoints/smpnn-ablation-a4-no-gnn-ln/best_checkpoint|--num_mp 6 --use_smpnn --use_gnn_ln False|$DEFAULT_DATA"
    "d2-alpha-1e-4|checkpoints/smpnn-ablation-d2-alpha-1e-4/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-4|$DEFAULT_DATA"
    "d3-alpha-1e-2|checkpoints/smpnn-ablation-d3-alpha-1e-2/best_checkpoint|--num_mp 6 --use_smpnn --alpha_init 1e-2|$DEFAULT_DATA"
    "b1-attn-1h|checkpoints/smpnn-ablation-b1-attn-1h/best_checkpoint|--num_mp 6 --use_smpnn --use_attention True --num_heads 1|$DEFAULT_DATA"
    # Hop=3 backbones (run_smpnn_hop3_native.sh). The with/without-transformer
    # pair at extended reach -- answers whether attention helps when 3-hop
    # subgraphs already give the model longer relational paths.
    "hop3-smpnn-6|checkpoints/smpnn-hop3-smpnn-6-hop3/best_checkpoint|--num_mp 6 --use_smpnn|$HOP3_DATA"
    "hop3-smpnn-6-attn-1h|checkpoints/smpnn-hop3-smpnn-6-hop3-attn-1h/best_checkpoint|--num_mp 6 --use_smpnn --use_attention True --num_heads 1|$HOP3_DATA"
)

eval_one() {
    local TAG=$1            # backbone tag
    local HEAD=$2           # tabpfn | tabicl
    local CKPT=$3           # backbone checkpoint path
    local BACKBONE_FLAGS=$4 # flags that recreate the backbone for ckpt loading
    local DATA_FLAGS=$5     # --hop / --fanout / --fewshotfanout matching training
    local EXTRA_FLAGS=$6    # head-specific knobs

    local RUN_TAG="${TAG}-${HEAD}-noproj"
    echo ""
    echo "================================================"
    echo ">>> $RUN_TAG"
    echo "    ckpt:    $CKPT"
    echo "    head:    $HEAD"
    echo "    bbflags: $BACKBONE_FLAGS"
    echo "    data:    $DATA_FLAGS"
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
        $DATA_FLAGS \
        --no_icl_projection \
        --probe_epochs 0 \
        --hiddim 512 --use_rev True --use_gate True \
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
    # JSON must be whitespace-free AND single-quoted: this array is expanded
    # via ${TABPFN_FLAGS[*]} and re-split on whitespace inside eval_one.
    # Whitespace inside the JSON shreds it across argv tokens; missing single
    # quotes would let bash strip the inner double quotes from the array element.
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES":50000,"MAX_NUMBER_OF_FEATURES":600}'
)

TABICL_FLAGS=(
    --icl_n_estimators 8
    --icl_max_context 10000
    --tabicl_checkpoint_version v2
)

for entry in "${BACKBONES[@]}"; do
    IFS='|' read -r TAG CKPT BACKBONE_FLAGS DATA_FLAGS <<< "$entry"
    eval_one "$TAG" "tabpfn" "$CKPT" "$BACKBONE_FLAGS" "$DATA_FLAGS" "${TABPFN_FLAGS[*]}"
    eval_one "$TAG" "tabicl" "$CKPT" "$BACKBONE_FLAGS" "$DATA_FLAGS" "${TABICL_FLAGS[*]}"
done

echo ""
echo "================================================"
echo "All ICL evals complete. Grep results:"
echo "  grep -E 'test_metric' logs/*-tabpfn-noproj/std* 2>/dev/null | sort"
echo "  grep -E 'test_metric' logs/*-tabicl-noproj/std* 2>/dev/null | sort"
echo "================================================"
