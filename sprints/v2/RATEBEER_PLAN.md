# Rel-Ratebeer on Griffin: Onboarding Plan

Goal: train Griffin (optionally + DFS) on **rel-ratebeer** and evaluate
on its RelBench-defined **test split**.

## 1. Where the data comes from

`rel-ratebeer` is one of the four databases added in **RelBench v2**
(ICLR 2026, [arXiv 2602.12606](https://arxiv.org/abs/2602.12606)) —
two decades of beer/place/user/brewer interactions with 200+ columns
of numeric, categorical, temporal, and **text** (review) features.
It loads through the standard RelBench API
([snap-stanford/relbench](https://github.com/snap-stanford/relbench)):

```python
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names
ds = get_dataset("rel-ratebeer", download=True)     # multi-GB download
db = ds.get_db()                                    # Database of Tables
print(get_task_names("rel-ratebeer"))               # e.g. churn / rating tasks
```

Each task ships train/val/test Tables split at RelBench's
`val_timestamp` / `test_timestamp` cutoffs — the "test split" we
evaluate on is exactly RelBench's official one, carried through
Griffin's task Arrow tables (`metatask.yaml` `split:`).

## 2. Pipeline overview

```
relbench download
   │  build_relbench_src.py            (bridge: tables/fkeys/tasks -> raw src)
   ▼
griffin_src/rel-ratebeer/              metadata.yaml + .npy columns
   │  dataconverter.py                 (node features)
   │  dataconverteredge.py             (adjacency + edge name embs)
   │  dataconvertertask.py             (task tables + metatask.yaml)
   │  dataconverterpost.py             (adds the "fewshot" edge emb)
   ▼
datasets/rel-ratebeer/                 Griffin-format dataset
   │  dataconverterdfs.py              (optional: DFS aggregates)
   ▼
hmaintask_combine.py --tasks ratebeer-… (train; valid split selects ckpt)
hmaintask_combine_llm.py --eval_tasks … (test-split metrics, TabPFN/TabICL)
```

## 3. Step-by-step

### 3.0 Install + inspect (CPU, ~minutes + download time)

```bash
conda activate griffin
pip install relbench

# Schema recon before exporting anything:
python build_relbench_src.py rel-ratebeer griffin_src/rel-ratebeer --dry_run
```

`--dry_run` prints every table (rows, pkey, time_col, fkeys) and the
task list. **Review this output first** — it drives the two decisions
below.

### 3.1 Export the raw src (bridge)

```bash
# v1: numeric + categorical + timestamps only (drop review text)
python build_relbench_src.py rel-ratebeer griffin_src/rel-ratebeer --skip_text
```

Decisions the bridge makes for you (revisit after v1 works):
- **Text columns dropped** under `--skip_text`. Ratebeer's review text
  is a large fraction of its signal; adding it later means Nomic-
  embedding millions of rows (GPU, hours) via the `Griffin_text_`
  convention — do it once the numeric-only pipeline is proven.
- **Rows of temporal tables are sorted by time** and fkeys remapped —
  required by Griffin's past-only fewshot sampler.
- **NaN handling**: numeric NaNs -> 0 plus an `_isnan` indicator
  column when >1% missing.

### 3.2 Classification tasks — one manual step

The scaffold exports **regression-form** tasks directly. RelBench
churn-style tasks are binary classification; joint-v65 encodes those
as "retrieval" with a synthetic label-holder node type (the
`seed_type`, `is_target: true`, one row per class) plus label edges
from entity rows. Before training a classification task, either:

  (a) mirror that construction in `build_relbench_src.py` (extend the
      task loop; use any joint-v65 classification task in
      `metatask.yaml` as the reference shape), or
  (b) start with ratebeer's regression tasks (ratings) only — zero
      extra work, and exercises the whole pipeline.

Recommendation: **(b) first**, then (a).

### 3.3 Run the 4-stage ETL

```bash
export SRC=griffin_src/rel-ratebeer DST=datasets/rel-ratebeer
python dataconverter.py     "$SRC" "$DST" --ncpu 16
python dataconverteredge.py "$SRC" "$DST"          # needs GPU (Nomic)
python dataconvertertask.py "$SRC" "$DST" --ncpu 16
python dataconverterpost.py "$DST"                  # needs GPU (Nomic)

# Sanity:
python -c "
from hdataset import Graph, Task
g, t = Graph('datasets/rel-ratebeer'), Task('datasets/rel-ratebeer')
print('nodes:', list(g.metanode)[:8]); print('tasks:', list(t.metatask))"
```

### 3.4 (Optional) DFS + register the task family

```bash
python dataconverterdfs.py datasets/rel-ratebeer

# task_names.yaml — append:
#   ratebeer:
#     - ratebeer-<task1>
#     - ratebeer-<task2>
```

### 3.5 Train on rel-ratebeer

```bash
accelerate launch hmaintask_combine.py \
    datasets/rel-ratebeer logs/ratebeer-v4-s42 ratebeer-v4-s42 \
    --tasks ratebeer --seed 42 \
    --num_mp 4 \
    --hiddim 512 --use_rev True --use_gate True \
    --maxepoch 20 --batchsize 256 --lr 3e-4 --wd 4e-4 \
    --hop 2 --fanout 20 --fewshotfanout 3 --eval_per_epoch 1 \
    --savepath checkpoints/ratebeer-v4-s42
# SMPNN variant: --num_mp 6 --use_smpnn --alpha_init 1e-2
# DFS variant:   --dfs_fewshot_depth 1 --dfs_root_depth 2
```

Per-epoch validation runs on the RelBench **val** cutoff;
`best_checkpoint` is selected there — test rows are never touched
during training.

### 3.6 Evaluate on the RelBench test split

```bash
python hmaintask_combine_llm.py \
    datasets/rel-ratebeer logs/ratebeer-eval-s42 ratebeer-eval-s42 \
    --head tabpfn \
    --tasks ratebeer --eval_tasks ratebeer \
    --loadpath checkpoints/ratebeer-v4-s42/best_checkpoint \
    --seed 42 --num_mp 4 \
    --hiddim 512 --use_rev True --use_gate True \
    --hop 2 --fanout 20 --fewshotfanout 3 --batchsize 256 \
    --output_mlp_dim 1 --no_target_normalize \
    --no_icl_projection --probe_epochs 0 \
    --icl_n_estimators 8 --icl_max_context 30000 \
    --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES":50000,"MAX_NUMBER_OF_FEATURES":600}' \
    --savepath checkpoints/ratebeer-eval-s42
```

`test_metric/<task>` lines are the RelBench-test-split numbers
(TabICL variant: `--head tabicl --icl_max_context 10000
--tabicl_checkpoint_version v2`). Match any `--dfs_*` / backbone flags
to training. Run ≥3 seeds before comparing variants.

## 4. Known gaps / order of attack

| # | Item | Status |
|---|---|---|
| 1 | Bridge validated against live rel-ratebeer schema | pending download — run `--dry_run` first |
| 2 | Classification (churn) task construction | manual step (§3.2); start with regression tasks |
| 3 | Review-text columns | deferred behind `--skip_text`; biggest quality lever once pipeline is proven |
| 4 | RelBench official metric parity | Griffin computes AUROC/MAE internally; cross-check one task against `task.evaluate()` |
