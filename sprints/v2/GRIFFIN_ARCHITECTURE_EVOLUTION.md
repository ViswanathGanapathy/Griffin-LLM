# Griffin Architecture Evolution: Vanilla → SMPNN → Depth-Aware Sampling → DFS Hybrid

A self-contained technical reference for the four stages of this
branch's architecture work. Each section explains *what the mechanism
is*, *why it exists*, *where it lives in code*, and *how to turn it
on*. Section 5 is a copy-pasteable recipe for a DFS-enabled training
run evaluated on a RelBench split.

All file references are to the `smpnn-ablations` branch of
`ViswanathGanapathy/Griffin-LLM`.

---

## 1. Griffin as-is (the baseline)

Griffin ([hmodel.py](../../hmodel.py), paper arXiv 2505.05568) is a
relational tabular foundation model: it encodes a multi-table database
as a heterogeneous graph — each table a node type, each foreign-key
relation an edge type, each row a node — and trains one encoder jointly
across many databases and tasks.

### 1.1 Data pipeline: per-target subgraph sampling

For each target row (e.g. a driver whose finishing position we
predict), the loader samples a local subgraph
([hdataset.py](../../hdataset.py) `Graph.subgraph`):

- **hop** (default 2): number of expansion rings around the target.
- **fanout** (default 10–20): per (source-node, edge-type) cap on
  sampled neighbors at each ring — random sampling with timestamp
  filtering (`getedge` masks any neighbor whose timestamp is not
  strictly in the past of the source row, preventing temporal leakage).
- Node features come from Arrow tables; numeric values are embedded by
  a frozen pretrained scalar encoder (`floatenc`,
  [hFloatEmb.py](../../hFloatEmb.py)), text values by cached Nomic
  embeddings.

### 1.2 Initial embeddings: column-name-conditioned attention

A node's raw representation is a set of `(column-name-embedding,
column-value-embedding)` pairs — shape `(C, D)` names + `(N, C, D)`
values. Layer 0 merges them with a small attention module
(`nodefeataggr`); between MP layers, an updated *task prompt*
(`taskfeat`, refreshed from the current node states) re-conditions the
column attention. Two properties matter later:

1. The attention is **shape-agnostic in C** — adding feature columns
   changes no parameter shapes (this is what makes the DFS integration
   in §4 nearly free).
2. The task's target column is **masked** on target rows
   (`target_feat_mask` from
   [hdataset.py](../../hdataset.py) `Task.get_retrieval/get_regression`),
   so the model cannot read the label it must predict.

### 1.3 The RMPNN operator

Griffin's message-passing core ([hmodel.py:147-174](../../hmodel.py#L147-L174))
aggregates per **(target-node, edge-type)** group:

```
edge_attr  = Linear(edge_attr)                          # relation embedding
out1       = mean-pool of neighbor states within each (target, edge-type) group
out2       = max-pool over edge types of (out1 * edge_attr)
```

Mean within a relation is noise-robust over large fan-in; max across
relations lets each node pick its most informative relation; the
learned relation embedding gates which dimensions each relation
surfaces. A **reverse RMPNN** on the flipped `edge_index` gives
bidirectional flow across directional FK edges.

### 1.4 The vanilla layer update (Eq. 8)

One residual update with three **parallel** branches sharing a single
non-affine LayerNorm ([hmodel.py:336-355](../../hmodel.py#L336-L355)):

```
x ← x + MLP_ff(LN(x))                                   # FFN branch
      + g_fwd · RMPNN(MLP_pre(LN(x)))                    # forward MP
      + g_rev · RMPNN_rev(MLP_pre(LN(x)))                # reverse MP
```

- `MLP_pre = Linear → SiLU` (nonlinearity *before* aggregation).
- Per-node gates `g_fwd, g_rev` are **zero-initialised** → MP branches
  silent at step 1.
- The FFN branch has **no gate** — always active at coefficient 1.

### 1.5 The fewshot mechanism (in-graph ICL)

With `--fewshotfanout K` (default 3), each target row additionally gets
K past rows of its own type as labeled exemplars
([hloaderwrapper.py](../../hloaderwrapper.py), `Graph.fewshot` in
[hdataset.py](../../hdataset.py)):

- Sampling is **past-only by construction**: a random index modulo the
  target's own row index (tables are time-sorted), optionally re-ranked
  by feature similarity when `prefetch_factor > 1`.
- Each exemplar is a **bare leaf** — `fewshotsubgraph` forces `hop=0`,
  so it carries only its own columns *including its label* (its target
  column is *not* masked).
- Leaves attach to the target via a synthetic **"fewshot" edge type**
  with its own learned relation embedding; message passing then mixes
  labeled context into the target through the ordinary RMPNN operator.

### 1.6 Known limitation

Stacking vanilla Griffin past L=4 degrades (over-smoothing): the
ungated FFN branch passes noise from the still-training MP branches
straight through, compounding with depth, with no mechanism to switch a
layer off. This motivates §2.

---

## 2. The SMPNN changes

SMPNN-Griffin ([hmodel_smpnn.py](../../hmodel_smpnn.py), enabled with
`--use_smpnn`) restructures each layer using three Transformer-era
tricks — per-sub-block LayerNorm, sequential residual composition, and
DiT-style near-identity initialisation via a learnable α.

### 2.1 The two-sub-block layer (Eqs. 9–15)

[hmodel_smpnn.py:295-327](../../hmodel_smpnn.py#L295-L327) (`_gnn_ff_step`):

```
# Sub-block 1: gated bidirectional graph convolution
lnx  = LN_1(x)                       # per-layer AFFINE LayerNorm
mlpx = Linear(lnx)                   # bare Linear — SiLU moved below
x ← x + α_gnn · SiLU(g_fwd·RMPNN(mlpx) + g_rev·RMPNN_rev(mlpx))

# Sub-block 2: gated feed-forward on the POST-aggregation state
x ← x + α_ff · MLP_ff(LN_2(x))
```

The three deltas vs vanilla:

1. **Sequential, not parallel**: the FFN sees the post-MP state and
   learns corrections instead of competing contributions.
2. **Per-sub-block affine LNs** (`ln_gnn[i]`, `ln_ff[i]`) replace the
   single shared non-affine LN.
3. **Identity at initialisation**: gates are zero-init so
   `SiLU(0)=0` kills sub-block 1 at step 1, and the learnable
   `α` scalars ([hmodel_smpnn.py:236-244](../../hmodel_smpnn.py#L236-L244))
   are initialised tiny (`--alpha_init`, 1e-6 or 1e-2) so sub-block 2
   starts near-zero. **The whole layer begins as ≈identity**, which is
   what lets deeper stacks train.

Cost: 2 affine LNs + 2 α scalars per layer ≈ 12K parameters on a 40M
model (<0.03%). No FLOPs change worth measuring.

### 2.2 Ablation switches and attention

The extended module carries per-component flags (all in
[hmodel_smpnn.py:179-187](../../hmodel_smpnn.py#L179-L187)):
`--use_alpha`, `--use_ff`, `--use_gnn_ln`, `--use_attention
--num_heads` (the O(N) virtual-node `LinearGlobalAttention` of the
SMPNN paper's Appendix A,
[hmodel_smpnn.py:129-166](../../hmodel_smpnn.py#L129-L166)), and
`--alpha_init`. These generate the 11-variant catalog in
[ARCHITECTURE_VARIANTS.md](ARCHITECTURE_VARIANTS.md).

### 2.3 What the 5-seed study found

([SMPNN_MASTER_RESULTS.md](SMPNN_MASTER_RESULTS.md),
[GRIFFIN_SMPNN_PAPER_DRAFT.md](GRIFFIN_SMPNN_PAPER_DRAFT.md))

- **Domain-specific, not universal**: SMPNN helps others-domain
  transfer (o1→o2), hurts commerce transfer in both directions.
- **Variance is the real win**: on o1→o2 TabPFN, SMPNN-6 α=1e-2
  matches Vanilla-6's mean (+0.003) with **7× lower seed-std**
  (±0.0102 → ±0.0015).
- **Depth sweet spot at L=6**: SMPNN needs L≥4, peaks at 6, degrades
  at 8 alongside vanilla — the near-identity init delays over-smoothing
  by ~2 layers, it does not eliminate it.
- A +0.050 single-seed commerce win **reversed** to −0.030 at n=5:
  report n≥5 for architectural claims in this regime.

Training flags: `--use_smpnn --num_mp 6 --alpha_init 1e-2` (warm-start
recommended) or `--alpha_init 1e-6` (from-scratch default).

---

## 3. Depth-aware graph sampling (`--fanout_decay`)

### 3.1 The problem: receptive field vs subgraph radius

An L-layer MPNN has an L-hop receptive field, but the sampler defaults
to `hop=2`. With `num_mp=6` and `hop=2`, the outer four layers re-mix
already-seen information — the depth buys refinement, not reach.
Setting `hop = num_mp` fixes the mismatch but explodes memory: at
constant fanout 20, ring size grows ~20^h per relation chain.

### 3.2 The change: geometric per-hop fanout decay

`Graph.subgraph` now takes `fanout_decay`
([hdataset.py](../../hdataset.py), inside the hop loop):

```
hop_fanout(h) = max(1, ceil(fanout · fanout_decay^h))
```

- `--fanout_decay 1.0` (default) → constant fanout, **bit-identical to
  the original behaviour**.
- `--fanout 20 --fanout_decay 0.5 --hop 6` → per-ring fanouts
  `20, 10, 5, 3, 2, 1`: inner rings keep full local context, outer
  rings stay affordable.
- `--fanout_decay 0.25` → `20, 5, 2, 1, 1, 1` (aggressive, for tight
  memory).
- Uncapped mode (`fanout = INF`) bypasses the decay entirely.

The fewshot path is untouched (leaves are `hop=0`; the decay loop never
executes for them).

### 3.3 Wiring and usage

The flag is plumbed through **all five** entrypoints
(`hmaintask_combine.py`, `hmaintask_combine_llm.py`,
`hmaintask_completion.py`,
`hmaintask_downsample_absolute_eval_sample.py`,
`optuna_griffin_llm.py`) via `subgraphargs`, with
`getattr(args, "fanout_decay", 1.0)` guards for safety.

Recommended pairing for deep SMPNN:

```bash
--use_smpnn --num_mp 6 --alpha_init 1e-2 \
--hop 6 --fanout 20 --fanout_decay 0.5
```

Experiment matrix: [run_smpnn_fanout_decay.sh](../../run_smpnn_fanout_decay.sh)
(baseline hop-2 vs hop-6 decay-0.5 vs decay-0.25; optional
`INCLUDE_OOM=1` demonstrates why constant-fanout hop-6 is infeasible).

---

## 4. The DFS changes (Deep Feature Synthesis hybrid)

Full design: [DFS_GRIFFIN_DESIGN.md](DFS_GRIFFIN_DESIGN.md).
Background: RDBLearn (arXiv 2602.18495) shows DFS-flattened relational
features + a frozen ICL head rival GNN encoders; fastdfs
(HKUSHXLab/fastdfs) is the reference engine.

### 4.1 Why: DFS and the MPNN are duals

| | DFS aggregates | Griffin RMPNN |
|---|---|---|
| Coverage | Exact, all neighbors | Fanout-sampled |
| Weights | Fixed primitives | Learned, gated |
| Variance | **Deterministic** | Seed-noisy (the §2.3 problem) |

Injecting deterministic exact aggregates alongside the learned sampled
ones directly targets the seed-variance weakness the multi-seed study
exposed.

### 4.2 Offline computation: `dataconverterdfs.py`

One-time pass over a processed Griffin dataset (CPU-ok):

- **Depth 1**, per node n, per relation r:
  `count` (log1p'd), and `mean`/`max` of each *float* column of the
  neighbor type — over **strictly-past** neighbors only (reuses
  `getedge`'s timestamp mask, so the leakage rule is identical to
  subgraph sampling; non-temporal types skip the cutoff).
- **Depth 2**: mean over strictly-past neighbors of *their* depth-1
  features, with the **no-backtrack rule** — B-features derived from
  the reverse of the traversed relation are excluded, blocking the
  A→B→A path that would leak a row's own label back into itself.
- Explosion control: float columns only, `--max_cols_per_rel 8`,
  `--max_d2_feats 32` (kept by variance), depth-2 mean-only.
- All features **z-normalized** (the frozen `floatenc` was trained on
  ~N(0,1) inputs).
- Column-**name embeddings** are composed from existing
  edge/feature-name embeddings (deterministic, no GPU; `--nomic`
  switches to true Nomic encoding), so the column attention gets
  semantically meaningful names like
  `dfs1 mean <relation> <column>`.

Artifacts land under `datasets/<name>/dfs/`:
`<nodetype>/d1.pt`, `<nodetype>/d2.pt`, `metadfs.yaml`,
`dfsfeatnameemb.pt`.

### 4.3 Runtime integration: extra columns, zero model changes

`Graph.subgraph(..., dfs_depth=0)` ([hdataset.py](../../hdataset.py))
appends the precomputed DFS columns (embedded through the same
`floatenc` path as native numerics) plus their name embeddings to the
**root node type** of that call. Depth is cumulative: 1 → d1; 2 → d1+d2.

Two loader-level switches route it:

- **`--dfs_fewshot_depth 1` (proposal 1)** — `fewshotsubgraph`
  overrides `dfs_depth` for the hop-0 leaf call
  ([hloaderwrapper.py](../../hloaderwrapper.py)). Each fewshot leaf
  arrives with a one-hop aggregate summary of its own neighborhood —
  the neighborhood context that `hop=0` denies it, at zero graph
  expansion. Verified on joint-v65: a rel-f1-drivers leaf goes from
  2 → 35 columns.
- **`--dfs_root_depth 2` (proposal 2)** — the main subgraph call
  enriches root rows with exact two-hop aggregates; the sampled MPNN
  runs on top. Verified: root rows 2 → 43 columns (2 native + 33 d1 +
  8 d2), and the merged fewshot graph + SMPNN forward run unchanged.

Mask correctness: `pad_feat_mask` pads `target_feat_mask` with True
(visible) for appended columns at all four use sites — DFS columns
aggregate *neighbor* values, never the row's own masked target column.
The fewshot similarity path receives the native-width mask slice.

Backward compatibility: defaults `0/0` are bit-identical to the
pre-DFS pipeline; enabling flags without artifacts raises a clear
error naming the converter command; checkpoints are architecture-
compatible in both directions (column attention is shape-agnostic) —
but **evaluate every checkpoint with the same DFS flags it was trained
with**, since the column set is part of the input contract.

### 4.4 Experiment matrix

[run_smpnn_dfs.sh](../../run_smpnn_dfs.sh): E0 (off/off), E1
(fewshot-d1), E2 (root-d2), E3 (both) × {SMPNN-6 α=1e-2, Vanilla-4} ×
3 seeds on others-1. Registered predictions in
[DFS_GRIFFIN_DESIGN.md §6](DFS_GRIFFIN_DESIGN.md): E1 should help
label-context tasks; E2's signature is *reduced seed-std*; E3 ≈ E2
would mean root aggregates subsume leaf enrichment.

---

## 5. Recipe: train with DFS and evaluate on a RelBench split

End-to-end on `joint-v65` (RelBench-derived). Train on the `others-1`
family; evaluate on the held-out **test split** of its tasks
(in-distribution) and on `others-2` (cross-task transfer, the paper's
headline direction).

### Step 0 — one-time DFS artifact computation

```bash
conda activate griffin
cd ~/Griffin

# All node types (CPU-ok; minutes for small families, longer for amazon-scale):
python dataconverterdfs.py datasets/joint-v65

# Sanity check:
ls datasets/joint-v65/dfs/          # metadfs.yaml, dfsfeatnameemb.pt, <type>/d1.pt ...
```

### Step 1 — train SMPNN-Griffin with both DFS integrations

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch hmaintask_combine.py \
    datasets/joint-v65 logs/dfs-e3-s42 dfs-e3-s42 \
    --tasks others-1 \
    --seed 42 \
    --num_mp 6 --use_smpnn --alpha_init 1e-2 \
    --dfs_fewshot_depth 1 --dfs_root_depth 2 \
    --hiddim 512 --use_rev True --use_gate True \
    --maxepoch 20 --batchsize 256 \
    --lr 3e-4 --wd 4e-4 \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --eval_per_epoch 1 \
    --savepath checkpoints/dfs-e3-s42 \
    --log_alpha_every 2 \
    2>&1 | tee logs/dfs-e3-s42.log
```

Variants: drop `--dfs_root_depth` for E1, drop `--dfs_fewshot_depth`
for E2, drop both for the E0 control. Or run the whole matrix with
`./run_smpnn_dfs.sh`. Per-epoch validation on the others-1 tasks'
**valid split** selects `best_checkpoint`.

### Step 2 — evaluate on the RelBench test split (in-distribution)

The task Arrow tables carry RelBench's train/valid/test boundaries
(`metatask.yaml` `split:`); `--eval_tasks` reports
`test_metric/<task>` on the held-out test rows. **The DFS flags must
match training.**

```bash
python hmaintask_combine_llm.py \
    datasets/joint-v65 logs/dfs-e3-s42-eval-o1 dfs-e3-s42-eval-o1 \
    --head tabpfn \
    --tasks rel-f1-driver-position \
    --eval_tasks others-1 \
    --loadpath checkpoints/dfs-e3-s42/best_checkpoint \
    --seed 42 \
    --num_mp 6 --use_smpnn --alpha_init 1e-2 \
    --dfs_fewshot_depth 1 --dfs_root_depth 2 \
    --hiddim 512 --use_rev True --use_gate True \
    --hop 2 --fanout 20 --fewshotfanout 3 \
    --batchsize 256 \
    --output_mlp_dim 1 --no_target_normalize \
    --no_icl_projection --probe_epochs 0 \
    --icl_n_estimators 8 --icl_max_context 30000 \
    --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache \
    --tabpfn_inference_precision autocast \
    --tabpfn_inference_config '{"MAX_NUMBER_OF_SAMPLES":50000,"MAX_NUMBER_OF_FEATURES":600}' \
    --savepath checkpoints/dfs-e3-s42-eval-o1 \
    2>&1 | tee logs/dfs-e3-s42-eval-o1.log
```

For the TabICL head instead: `--head tabicl --icl_max_context 10000
--tabicl_checkpoint_version v2` (classification; the regressor default
`tabicl-regressor-v2-20260212.ckpt` is auto-selected for regression
tasks).

### Step 3 — cross-task transfer (the headline comparison)

Same command with `--eval_tasks others-2` — reports test metrics on
airbnb-destination, rel-trial-*, etc., none seen in training. This is
the o1→o2 cell where SMPNN-6 α=1e-2 holds the 7× variance reduction;
the DFS hypothesis (E2/E3) predicts a further std reduction here.

### Step 4 — aggregate and compare

```bash
python collect_smpnn_results.py \
    --logs logs/dfs-e3-*-eval-*.log \
    --out smpnn_dfs_results.csv
```

Compare each DFS cell against E0 on **both mean and seed-std** per
(direction, head). Repeat Steps 1–3 for seeds 43, 44 (or run
`run_smpnn_dfs.sh`, which loops them) before drawing conclusions —
per §2.3, single-seed deltas in this regime are noise.

### Common pitfalls

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ... dfs/metadfs.yaml` | Run Step 0 (`dataconverterdfs.py`) first. |
| Eval numbers wildly off vs training-time valid | DFS flags at eval don't match training — the column set is part of the input contract. |
| OOM at train time | Drop `--batchsize` 256→128; DFS adds ~30–40 columns per enriched row, a modest but real activation increase. |
| A node type shows no DFS columns | It has no relations (or is a label-holder `is_target` type) — the converter skips it by design; loaders handle the absence gracefully. |

---

## Appendix: flag cheat-sheet across all four stages

| Stage | Flags | Default (= legacy behaviour) |
|---|---|---|
| Griffin baseline | `--num_mp 4 --hop 2 --fanout 20 --fewshotfanout 3` | — |
| SMPNN | `--use_smpnn --num_mp 6 --alpha_init {1e-6,1e-2}` (+ ablations `--use_alpha/--use_ff/--use_gnn_ln/--use_attention`) | off |
| Depth-aware sampling | `--fanout_decay 0.5` (pair with `--hop == --num_mp`) | 1.0 (constant) |
| DFS hybrid | `--dfs_fewshot_depth 1`, `--dfs_root_depth 2` (after `python dataconverterdfs.py <dataset>`) | 0 / 0 (off) |
