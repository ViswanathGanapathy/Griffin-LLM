# Griffin

This is the official implementation of the paper "[Griffin: Towards a Graph-Centric Relational Database Foundation Model](https://arxiv.org/abs/2505.05568)".

---

## Table of Contents

- [Griffin](#griffin)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
    - [Using Provided Processed Datasets](#using-provided-processed-datasets)
    - [Processing Raw Data](#processing-raw-data)
  - [Pretraining \& Finetuning](#pretraining--finetuning)
    - [Using Provided Pretrained Checkpoints](#using-provided-pretrained-checkpoints)
    - [Pretraining from Scratch](#pretraining-from-scratch)
    - [Finetuning on Specific Datasets](#finetuning-on-specific-datasets)
    - [Transfer Experiments](#transfer-experiments)
      - [Overview](#overview)
      - [Data Splits](#data-splits)
      - [Pretrained Models](#pretrained-models)
      - [Run Command](#run-command)
      - [Notes](#notes)

---

## Getting Started

### Prerequisites

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) first.

Then install the following dependencies:

```bash
pip install datasets pqdm accelerate evaluate sentence_transformers einops torchmetrics seaborn
```

---

## Dataset Preparation

This section describes how to prepare the datasets for use with this project.

### Using Provided Processed Datasets

We provide already processed datasets for your convenience. You can download them from [Hugging Face RDB datasets collection](https://huggingface.co/datasets/yamboo/Griffin_datasets_joint_v65) and [Hugging Face Single datasets collection](https://huggingface.co/datasets/yamboo/Griffin_datasets_single_pretrain_v3).
The datasets are organized as follows:

```bash
./datasets/
├── joint-v65 # Major dataset, including all RDB datasets. Used for main experiments.
└── single-pretrain-v3 # Single-table dataset, including all single-table datasets. Used for pretraining.
```

### Processing Raw Data

If you wish to process the raw data yourself, the scripts and instructions are available in a separate branch: [processing_data](https://github.com/yanxwb/Griffin/tree/processing_data).

---

## Pretraining & Finetuning

This section covers how to use or pretrain your own models, and finetune them on specific datasets.

### Using Provided Pretrained Checkpoints

We provide pretrained model checkpoints to get you started quickly. You can download them from [Hugging Face checkpoints collection](https://huggingface.co/yamboo/Griffin_models).

The checkpoints are organized as follows:

```bash
./checkpoints/
├── single-completion # Pretrained single table completion model.
├── single-sft # Pretrained single table SFT model. Used in main experiments.
└── transfer # Pretrained transfer model. Used in transfer experiments.
    ├── commerce-1 # Split name.
       ├── FULL # Model name.
       ├── MIXED # Model name.
       └── LIMITED # Model name.
    ├── commerce-2 # Same as above.
       ├── FULL
       ├── MIXED
       └── LIMITED
    ├── others-1
       ├── FULL
       ├── MIXED
       └── LIMITED
    └── others-2
       ├── FULL
       ├── MIXED
       └── LIMITED
```

### Pretraining from Scratch

To pretrain the model from scratch, you can use the following command:

```bash
# Step 1: Completion Pretraining
accelerate launch --config_file hconfig.yaml hmaintask_completion.py datasets/single-pretrain-v3 logs/single-completion log --savepath checkpoints/single-completion --hop 0 --fanout 10 --fewshotfanout 0 --maxepoch 5 --batchsize 4096 --lr 0.00010178976613680036 --wd 0.008749895419888909 --num_mp 4 --use_rev True --use_gate True --eval_per_epoch 1 --hiddim 512

# Step 2: SFT Pretraining
accelerate launch --config_file hconfig.yaml hmaintask_combine.py datasets/single-pretrain-v3 logs/single-sft log --loadpath checkpoints/single-completion/best_checkpoint --savepath checkpoints/single-sft --task ALLTASK --hop 0 --fanout 10 --fewshotfanout 0 --maxepoch 40 --batchsize 4096 --lr 0.00042364843314963003 --wd 2.423189169972981e-05 --num_mp 4 --use_rev True --use_gate True --eval_per_epoch 2 --hiddim 512
```

### Finetuning on Specific Datasets

Once you have a pretrained model (either downloaded or trained by yourself), you can finetune it on specific downstream datasets.

To finetune the model, use the following command:

```bash
accelerate launch --config_file hconfig.yaml hmaintask_combine.py $dataset $log_path $log_name --loadpath $load_path --savepath $save_path --tasks $TASK --hop 2 --fanout 20 --maxepoch 50 --patience 15 --eval_per_epoch 2 --batchsize 256 --lr 3e-4 --wd 2e-4 --num_mp 4 --use_rev True --use_gate False --fewshotfanout 3 --hiddim 512 
```


### Transfer Experiments

#### Overview

The experiment involves four data splits and three types of pretrained models trained on each split. Each split contains six tasks, defined in the `task_names.yaml` file. The goal is to evaluate the transferability of models pretrained on one split to another.

#### Data Splits

The data splits are as follows:

- **commerce-1**
- **commerce-2**
- **others-1**
- **others-2**

#### Pretrained Models

The three pretrained models are:

- **FULL**
- **MIXED**
- **LIMITED**

#### Run Command

Use the following format to run the experiment:

```bash
bash transfer.sh $GPU $SPLIT_1 $SPLIT_2 $TASK $MODEL $EVAL_SAMPLE_RATIO $SEED_START $SEED_END
```

#### Notes

1. **Multiple GPUs:** You can use multiple GPUs to parallelize the experiments. For example, to use two GPUs with index 0 and 1:

```bash
bash transfer.sh 0,1 commerce-1 others-1 rel-f1-driver-dnf 1 42 43
```

2. **Model Indexing:**
   - `FULL`: index `1`
   - `MIXED`: index `2`
   - `LIMITED`: index `3`

3. **Evaluation Sample Ratio:**
   Evaluation sample ratio is set to 1 by default. This only applies to the validation phase and does not affect the final testing result. It is okay to set it to 1 for most tasks. If you want to evaluate on a smaller subset of the data to speed up the evaluation, you can set the evaluation sample ratio to a value less than 1 like 0.2.

---

## Extended Heads: LLM, TabPFN v2, TabICL v2

The unified training script `hmaintask_combine_llm.py` supports **five prediction heads** over the same Griffin MPNN backbone. See [CHANGES.md](CHANGES.md) for detailed implementation notes.

### Architecture

```
Griffin MPNN ──→ [N, 512] node embeddings
      │
      ├──→ Default Head (dec / @y.T)        ──→ regression / classification
      ├──→ LLM Head (Projection → LLM)      ──→ text generation
      ├──→ LLM+MLP Head (Proj → LLM → MLP)  ──→ direct prediction
      ├──→ TabPFN v2 (ICL)                  ──→ in-context prediction
      └──→ TabICL v2 (ICL)                  ──→ in-context prediction
```

### Additional Dependencies

```bash
# For --head llm or llm_mlp
pip install transformers
pip install peft  # only if using --use_lora

# For --head tabpfn
pip install tabpfn

# For --head tabicl
pip install tabicl
```

### Pre-training with Extended Heads

Pre-training always uses the original scripts (Stage 1: `hmaintask_completion.py`, Stage 2: `hmaintask_combine.py`). The extended heads are used at fine-tuning and evaluation time.

```bash
# Stage 1: Completion Pretraining (unchanged)
accelerate launch --config_file hconfig.yaml hmaintask_completion.py \
    datasets/single-pretrain-v3 logs/pretrain log \
    --savepath checkpoints/pretrain \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True \
    --hop 0 --fanout 10 --fewshotfanout 0 \
    --batchsize 4096 --lr 1e-4 --maxepoch 5

# Stage 2: SFT Pretraining (unchanged)
accelerate launch --config_file hconfig.yaml hmaintask_combine.py \
    datasets/single-pretrain-v3 logs/sft log \
    --loadpath checkpoints/pretrain/best_checkpoint \
    --savepath checkpoints/sft \
    --tasks ALLTASK --hop 0 --fanout 10 --fewshotfanout 0 \
    --batchsize 4096 --lr 4e-4 --maxepoch 40 --eval_per_epoch 2 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate True
```

### Fine-tuning with Default Head (Griffin Alone)

```bash
# Backward-compatible: identical to using hmaintask_combine.py
accelerate launch --config_file hconfig.yaml hmaintask_combine_llm.py \
    datasets/joint-v65 logs/default-ft log \
    --head default \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/default-ft \
    --hop 2 --fanout 20 --batchsize 256 --lr 3e-4 --wd 2e-4 \
    --maxepoch 50 --patience 15 --eval_per_epoch 2 \
    --num_mp 4 --use_rev True --use_gate False --fewshotfanout 3 --hiddim 512
```

### Fine-tuning with Griffin + LLM

```bash
# Frozen LLM, train projection layer + Griffin
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/llm-ft log \
    --head llm \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/llm-ft \
    --llm_model meta-llama/Llama-3.2-1B --llm_frozen \
    --batchsize 4 --lr 1e-4 --wd 2e-4 \
    --maxepoch 10 --patience 5 --eval_per_epoch 1 \
    --neighbor_tokens 5 --num_demo 3 \
    --hiddim 512 --num_mp 4 --use_rev True --use_gate False \
    --hop 2 --fanout 20 --fewshotfanout 3

# With LoRA fine-tuning on the LLM
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/llm-lora log \
    --head llm \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/llm-lora \
    --llm_model meta-llama/Llama-3.2-1B --use_lora \
    --batchsize 4 --lr 1e-4 --maxepoch 10

# LLM + MLP output head (faster inference)
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/llm-mlp log \
    --head llm_mlp \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/llm-mlp \
    --llm_model meta-llama/Llama-3.2-1B --use_lora \
    --output_mlp_dim 1 \
    --batchsize 4 --lr 1e-4 --maxepoch 10
```

### Fine-tuning with Griffin + TabPFN v2

```bash
# Frozen Griffin, pure ICL evaluation
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn log \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft --freeze_griffin \
    --icl_n_estimators 8 --hiddim 512

# Fine-tune Griffin with linear probe, then evaluate with TabPFN
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-probe log \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/tabpfn-probe \
    --probe_epochs 5 --output_mlp_dim 1 \
    --batchsize 256 --lr 1e-4 --hiddim 512

# Fine-tune TabPFN itself on Griffin embeddings
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabpfn-ft log \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft --freeze_griffin \
    --tabpfn_finetune --tabpfn_finetune_epochs 30 --tabpfn_finetune_lr 2e-5
```

### Fine-tuning with Griffin + TabICL v2

```bash
# Frozen Griffin, pure ICL evaluation
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabicl log \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft --freeze_griffin \
    --icl_n_estimators 8 --hiddim 512

# Fine-tune Griffin with linear probe, then evaluate with TabICL
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/tabicl-probe log \
    --head tabicl \
    --tasks rel-f1-driver-position \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/tabicl-probe \
    --probe_epochs 5 --output_mlp_dim 1 \
    --batchsize 256 --lr 1e-4 --hiddim 512
```

### Comparing All Five Heads on One Task

```bash
DATASET=datasets/joint-v65
CKPT=checkpoints/single-sft
TASK="rel-f1-driver-position"

# 1. Griffin alone
python hmaintask_combine_llm.py $DATASET logs/cmp/default cmp \
    --head default --tasks $TASK --loadpath $CKPT \
    --savepath checkpoints/cmp/default \
    --batchsize 256 --lr 1e-4 --maxepoch 10 --patience 5 --hiddim 512

# 2. Griffin + LLM
python hmaintask_combine_llm.py $DATASET logs/cmp/llm cmp \
    --head llm --tasks $TASK --loadpath $CKPT \
    --savepath checkpoints/cmp/llm \
    --llm_model meta-llama/Llama-3.2-1B --llm_frozen \
    --neighbor_tokens 5 --num_demo 3 \
    --batchsize 4 --lr 1e-4 --maxepoch 10 --hiddim 512

# 3. Griffin + LLM + MLP
python hmaintask_combine_llm.py $DATASET logs/cmp/llm-mlp cmp \
    --head llm_mlp --tasks $TASK --loadpath $CKPT \
    --savepath checkpoints/cmp/llm-mlp \
    --llm_model meta-llama/Llama-3.2-1B --use_lora \
    --output_mlp_dim 1 --batchsize 4 --lr 1e-4 --maxepoch 10 --hiddim 512

# 4. Griffin + TabPFN v2
python hmaintask_combine_llm.py $DATASET logs/cmp/tabpfn cmp \
    --head tabpfn --tasks $TASK --loadpath $CKPT --freeze_griffin \
    --icl_n_estimators 8 --hiddim 512

# 5. Griffin + TabICL v2
python hmaintask_combine_llm.py $DATASET logs/cmp/tabicl cmp \
    --head tabicl --tasks $TASK --loadpath $CKPT --freeze_griffin \
    --icl_n_estimators 8 --hiddim 512
```

### Generalization to Unseen Tasks

Train on one task group, evaluate zero-shot on a different group:

```bash
# Train on commerce-1
accelerate launch --config_file hconfig.yaml hmaintask_combine_llm.py \
    datasets/joint-v65 logs/gen/train gen-train \
    --head default --tasks commerce-1 \
    --loadpath checkpoints/single-sft \
    --savepath checkpoints/gen/train \
    --batchsize 256 --lr 3e-4 --maxepoch 10 --hiddim 512

# Evaluate on unseen commerce-2 tasks with each head
CKPT_GEN=checkpoints/gen/train/best_checkpoint
for HEAD in default tabpfn tabicl; do
    python hmaintask_combine_llm.py datasets/joint-v65 logs/gen/$HEAD gen-$HEAD \
        --head $HEAD --tasks commerce-2 --mode test \
        --loadpath $CKPT_GEN --freeze_griffin --hiddim 512
done
```

### Extended CLI Arguments (New)

| Flag | Default | Heads | Description |
|------|---------|-------|-------------|
| `--head` | `default` | all | `default`, `llm`, `llm_mlp`, `tabpfn`, `tabicl` |
| `--llm_model` | `Llama-3.2-1B` | llm, llm_mlp | HuggingFace model ID |
| `--llm_frozen` | True | llm, llm_mlp | Freeze LLM weights |
| `--use_lora` | False | llm, llm_mlp | Apply LoRA to LLM |
| `--lora_r` | 8 | llm, llm_mlp | LoRA rank |
| `--lora_alpha` | 16 | llm, llm_mlp | LoRA alpha |
| `--freeze_griffin` | False | all non-default | Freeze Griffin MPNN |
| `--max_new_tokens` | 16 | llm | Max generation tokens |
| `--projector_bottleneck` | 1024 | llm, llm_mlp | Projector bottleneck dim |
| `--projector_lr` | 1e-3 | llm, llm_mlp | Projector LR in phase-2 |
| `--neighbor_tokens` | 0 | llm, llm_mlp | Neighbor embeddings in prompt |
| `--num_demo` | 0 | llm, llm_mlp | ICL demo examples in prompt |
| `--warmup_epochs` | 0 | llm, llm_mlp | Projector-only warmup epochs |
| `--output_mlp_dim` | 1 | llm_mlp, tabpfn, tabicl | Output dimension |
| `--tabpfn_finetune` | False | tabpfn | Fine-tune TabPFN |
| `--tabpfn_finetune_epochs` | 30 | tabpfn | TabPFN fine-tuning epochs |
| `--tabpfn_finetune_lr` | 2e-5 | tabpfn | TabPFN fine-tuning LR |
| `--icl_n_estimators` | 8 | tabpfn, tabicl | Ensemble members |
| `--probe_epochs` | 0 | tabpfn, tabicl | Griffin fine-tuning with linear probe |

### New File Structure

```
Griffin/
├── hmodel.py                    # GriffinMod MPNN (unchanged)
├── hdataset.py                  # Graph, Task, Node (unchanged)
├── hloaderwrapper.py            # LoaderWrapperTask + LoaderWrapperTaskLLM (new)
├── hFloatEmb.py                 # Float encoding/decoding (unchanged)
├── metric.py                    # Evaluation metrics (unchanged)
├── task_names.yaml              # Task split definitions (unchanged)
├── hconfig.yaml                 # Accelerate config (unchanged)
│
├── hmaintask_completion.py      # Stage 1: pretraining (unchanged)
├── hmaintask_combine.py         # Stage 2/3: original training (unchanged)
├── hmaintask_combine_llm.py     # Stage 2/3: UNIFIED training (5 heads)
│
├── task_prompts.py              # Task descriptions for LLM prompts (new)
├── tabular_heads.py             # TabPFN v2 / TabICL v2 wrappers (new)
│
├── CHANGES.md                   # Detailed change notes (new)
└── README.md                    # This file (updated)
```

---

## SMPNN Extension (branch `smpnn-ablations`)

This branch adds a **SMPNN-Griffin** variant that ports the DiT-style
depth-scaling recipe to Griffin's heterogeneous relational MPNN,
plus a **decaying-fanout subgraph sampler** that makes deeper stacks
tractable in memory. Everything is fully backward compatible: if you
never set `--use_smpnn` and leave `--fanout_decay 1.0` (the defaults),
behaviour is identical to the original Griffin.

### The SMPNN backbone

A new file `hmodel_smpnn.py` defines `GriffinMod`, a Griffin encoder
with per-layer affine LayerNorms, sequential graph→FFN sub-blocks,
and a learnable α scalar on the FFN sub-block (initialised near
zero for DiT-style identity init).

Training flag: `--use_smpnn` routes to `hmodel_smpnn.py` instead of
the original `hmodel.py`.

| Flag | Values | Meaning |
|---|---|---|
| `--use_smpnn` | (bool) | Enable the SMPNN backbone. |
| `--num_mp` | 4–8 | Number of MP layers. `6` is the sweet spot (Table B, paper). |
| `--alpha_init` | `1e-6`, `1e-4`, `1e-2` | Initial value of the learnable α FFN residual scale. `1e-6` = paper default (D1); `1e-2` = warm-start optimised (D3). |
| `--use_alpha` / `--use_ff` / `--use_gnn_ln` / `--use_attention` | (bool) | Ablation switches — see [ARCHITECTURE_VARIANTS](sprints/v2/ARCHITECTURE_VARIANTS.md). |
| `--log_alpha_every N` | int | Print α values every N epochs (0 = off). |

### Decaying-fanout subgraph sampler (NEW)

When `num_mp ≥ 4`, the effective receptive field of the encoder
exceeds `hop=2`, so a hop-2 subgraph does not fully use the deeper
stack. Setting `hop = num_mp` gives the model its full receptive
field but explodes subgraph size at constant fanout.

`hdataset.py::Graph.subgraph()` now accepts a `fanout_decay` argument
that geometrically shrinks the per-hop fanout:

```
hop_fanout(h) = max(1, ceil(fanout * fanout_decay ** h))
```

- `--fanout_decay 1.0` (default) — constant fanout at every hop.
  **Identical to pre-existing behaviour** — no observable change on
  legacy scripts.
- `--fanout_decay 0.5` with `--fanout 20 --hop 6` — per-hop fanouts
  `20, 10, 5, 3, 2, 1`. Inner rings preserve local context; outer
  rings stay cheap.
- `--fanout_decay 0.25` with `--fanout 20 --hop 6` — per-hop fanouts
  `20, 5, 2, 1, 1, 1`. Aggressive shrink; recommended when GPU
  memory is tight.

Fewshot leaves are unaffected (they are sampled with `hop=0` in
`hloaderwrapper.py::fewshotsubgraph`, so the decay loop never
executes for them).

The flag is wired through **all five** training and evaluation
entrypoints: `hmaintask_combine.py`, `hmaintask_combine_llm.py`,
`hmaintask_completion.py`,
`hmaintask_downsample_absolute_eval_sample.py`, and
`optuna_griffin_llm.py`.

### Recommended combinations for common goals

| Goal | Flags |
|---|---|
| Reproduce original Griffin | *(no new flags)* |
| SMPNN paper default | `--use_smpnn --num_mp 6 --alpha_init 1e-6` |
| SMPNN warm-start (paper headline) | `--use_smpnn --num_mp 6 --alpha_init 1e-2` |
| Deep SMPNN with full receptive field | `--use_smpnn --num_mp 6 --alpha_init 1e-2 --hop 6 --fanout 20 --fanout_decay 0.5` |
| Aggressive memory-saving deep run | `--use_smpnn --num_mp 6 --alpha_init 1e-2 --hop 6 --fanout 20 --fanout_decay 0.25 --batchsize 128` |

### Full experiment recipes (all live on this branch)

Wrapper shell scripts sit at the repo root; every script tees per-cell
logs to `logs/<name>/` and skips completed cells via a checkpoint /
`test_metric` line count check.

| Script | What it does | Cost |
|---|---|---|
| `run_smpnn_multiseed_backbones.sh` | Train 20 backbones = 4 architectures × 5 seeds on `others-1`. Core paper artifact. | ~50 GPU-h |
| `run_smpnn_multiseed_eval.sh` | Cross-task eval of the 20 backbones on the 5 headline transfer directions × 2 ICL heads. | ~2 GPU-h |
| `run_smpnn_depth_native.sh` | Depth sweep L ∈ {2, 4, 6, 8} × Vanilla vs SMPNN, 3 seeds each. | ~20 GPU-h |
| `run_smpnn_native_head_o1_o2.sh` | Native Griffin head on o1→o2 — sanity-check that the 7× variance reduction is a backbone property, not a TabPFN artefact. | ~1.7 GPU-h |
| `run_smpnn_o1_o2_second_anchor.sh` | Second AUROC anchor (rel-trial-study-outcome) for o1→o2 with both TabPFN and TabICL. | ~3.3 GPU-h |
| `run_smpnn_alpha_evolution.sh` + `parse_alpha_evolution.py` + `plot_alpha_evolution.py` | Re-train SMPNN-6 with per-epoch α logging enabled; extract CSV; produce two-panel log-scale plot showing α_gnn / α_ff trajectories per layer. | ~20–24 GPU-h |
| `run_smpnn_fanout_decay.sh` (NEW) | Demonstrate the `--fanout_decay` flag: SMPNN-6 with `hop=2` (baseline) vs `hop=6, decay=0.5` (recommended) vs `hop=6, decay=0.25` (aggressive) — 3 seeds each. | ~30–40 GPU-h |
| `run_smpnn_dfs.sh` (NEW) | DFS + MPNN hybrid matrix: E0 baseline / E1 fewshot-leaf DFS-1 / E2 root DFS-2 / E3 both, × {SMPNN-6, Vanilla-4} × 3 seeds. Requires `python dataconverterdfs.py datasets/joint-v65` first. | ~90 GPU-h |

### DFS + MPNN hybrid (NEW)

Inspired by RDBLearn (arXiv 2602.18495) and fastdfs
(github.com/HKUSHXLab/fastdfs): precomputed Deep-Feature-Synthesis
aggregates (count / mean / max per relation, strict past-only cutoff,
no-backtrack at depth 2) are appended as extra **feature columns**, which
Griffin's column-name-conditioned attention absorbs with **zero model
changes** (existing checkpoints still load).

```bash
# 1. One-time offline computation (CPU-ok):
python dataconverterdfs.py datasets/joint-v65

# 2. Train with DFS-enriched fewshot leaves (one-hop neighborhood
#    context for hop-0 leaves, no graph expansion):
accelerate launch hmaintask_combine.py ... --dfs_fewshot_depth 1

# 3. Train with two-hop DFS on root nodes (exact unsampled aggregates
#    alongside the sampled MPNN):
accelerate launch hmaintask_combine.py ... --dfs_root_depth 2

# Eval MUST use the same dfs flags the checkpoint was trained with.
```

Design + leakage rules + registered predictions:
[sprints/v2/DFS_GRIFFIN_DESIGN.md](sprints/v2/DFS_GRIFFIN_DESIGN.md).
Experiment matrix: `run_smpnn_dfs.sh`.

### Documentation

- DFS + MPNN hybrid design: [sprints/v2/DFS_GRIFFIN_DESIGN.md](sprints/v2/DFS_GRIFFIN_DESIGN.md)
- Extended technical report: [sprints/v2/GRIFFIN_SMPNN_PAPER_DRAFT.md](sprints/v2/GRIFFIN_SMPNN_PAPER_DRAFT.md)
- LoG 2026 4-page draft: [sprints/v2/GRIFFIN_SMPNN_LOG_4PAGE.md](sprints/v2/GRIFFIN_SMPNN_LOG_4PAGE.md) + LaTeX bundle at [sprints/v2/log_2026_submission/](sprints/v2/log_2026_submission/)
- Full multi-seed results readout: [sprints/v2/SMPNN_MASTER_RESULTS.md](sprints/v2/SMPNN_MASTER_RESULTS.md)
- Architecture variant catalog: [sprints/v2/ARCHITECTURE_VARIANTS.md](sprints/v2/ARCHITECTURE_VARIANTS.md)
- Data-conversion handoff for new collaborators: [sprints/v2/NEW_DATASET_HANDOFF.md](sprints/v2/NEW_DATASET_HANDOFF.md)
- Multi-seed collaborator handoff: [sprints/v2/COLLABORATOR_HANDOFF.md](sprints/v2/COLLABORATOR_HANDOFF.md)

