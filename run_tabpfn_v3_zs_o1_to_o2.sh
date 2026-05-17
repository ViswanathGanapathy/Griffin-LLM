#!/bin/bash
# TabPFN v3 zero-shot ICL on the o1->o2 transfer.
#
# Verified via probe_tabpfn_inference.py (tabpfn 8.0.3):
#   - fit_mode accepts: 'low_memory' | 'fit_preprocessors' |
#                       'fit_with_cache' | 'batched'
#     fit_with_cache is the v3 KV-cache path: precompute context K/V
#     once at fit() time; predict() reuses.
#   - inference_precision: torch.dtype | 'autocast' | 'auto'.
#     'autocast' auto-picks bf16/fp16 on A100/H100; the cleanest CLI
#     choice. Our TabPFNHead also accepts strings like 'bfloat16' and
#     converts them to torch dtypes.
#   - inference_config: dict with UPPERCASE keys (matching the
#     InferenceConfig dataclass fields). MAX_NUMBER_OF_SAMPLES is the
#     hard cap on context rows — bump above the default so 30K context
#     actually goes through.
#
# Usage:
#   ./run_tabpfn_v3_zs_o1_to_o2.sh 2>&1 | tee logs/tabpfn-v3-zs-o1-to-o2.log

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHONUNBUFFERED=1 python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-v3-zs-o1-to-o2 tabpfn-v3-zs-o1-to-o2 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-2 \
    --loadpath checkpoints/o1-tth-lora-v3/best_checkpoint \
    --probe_epochs 5 \
    --icl_proj_dim 128 \
    --icl_max_context 30000 \
    --icl_n_estimators 8 \
    --tabpfn_version v3 \
    --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES": 50000}' \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 --lr 1e-4 --wd 2e-4 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --output_mlp_dim 1 \
    --no_target_normalize \
    --savepath checkpoints/tabpfn-v3-zs-o1-to-o2
