# DFS × Griffin Ablation Plan (Collaborator Edition)

Six-configuration study: {vanilla Griffin, SMPNN-Griffin} × {no DFS,
DFS on fewshot leaves, DFS on fewshot leaves + root nodes}. All
configurations **train with Griffin's native decoder** and are then
**evaluated with TabPFN v2.5 as a frozen in-context (ICL) head** over
the trained encoder's embeddings.

Everything needed is on the `smpnn-ablations` branch of
`ViswanathGanapathy/Griffin-LLM`. Companion reading:
[DFS_INTEGRATION_WALKTHROUGH.md](DFS_INTEGRATION_WALKTHROUGH.md) (code
review guide), [DFS_GRIFFIN_DESIGN.md](DFS_GRIFFIN_DESIGN.md) (design +
leakage rules), [GRIFFIN_ARCHITECTURE_EVOLUTION.md](GRIFFIN_ARCHITECTURE_EVOLUTION.md)
(all four architecture stages), and
[COMMERCE_DFS_RUNBOOK.md](COMMERCE_DFS_RUNBOOK.md) (concrete
commerce-1 → commerce-2 instructions: Step-0 artifact conversion,
vanilla-Griffin training, TabICL v2 evaluation).

---

> **⚠ Cutoff-time fix (2026-08-23) — earlier DFS results are invalid as
> evidence.** DFS artifacts generated before format v2 were evaluated
> at each row's own timestamp, with *no* cutoff for non-temporal root
> types — aggregating the evaluation window (and, potentially, label
> relations) into the features. Any E1/E2/E3 checkpoint trained on v1
> artifacts measures a partially leaked signal; treat those numbers as
> an "E-leaky" row only. Before running any DFS cell:
>
> ```bash
> python dataconverterdfs.py datasets/joint-v65 \
>     --nodetypes <types...> --cutoff_stats --tasks <tasks...>
> ```
>
> `--cutoff_stats` is a **prerequisite** for supervised-task training:
> it fits the cutoff-mode normalization stats (`d1_at`/`d2_at`) the
> online path needs; the loader refuses to serve task cutoffs without
> them. The measured leakage (E-leaky − E-fixed, same seeds) is itself
> a result worth reporting. See DFS_INTEGRATION_WALKTHROUGH.md §0a.

## 1. The configuration matrix

| # | Tag | Backbone | `--dfs_fewshot_depth` | `--dfs_root_depth` | Extra backbone flags |
|---|---|---|---|---|---|
| 1 | `E0-vanilla` | Vanilla Griffin, L=4 | 0 | 0 | `--num_mp 4` |
| 2 | `E1-vanilla` | Vanilla Griffin, L=4 | **1** | 0 | `--num_mp 4` |
| 3 | `E3-vanilla` | Vanilla Griffin, L=4 | **1** | **2** | `--num_mp 4` |
| 4 | `E0-smpnn` | SMPNN, L=6, α=1e-2 | 0 | 0 | `--num_mp 6 --use_smpnn --alpha_init 1e-2` |
| 5 | `E1-smpnn` | SMPNN, L=6, α=1e-2 | **1** | 0 | `--num_mp 6 --use_smpnn --alpha_init 1e-2` |
| 6 | `E3-smpnn` | SMPNN, L=6, α=1e-2 | **1** | **2** | `--num_mp 6 --use_smpnn --alpha_init 1e-2` |

Notes:
- "DFS for fewshot fanout nodes alone" = `E1` (`dfs_fewshot_depth=1`):
  each hop-0 fewshot leaf carries a one-hop aggregate summary of its
  own neighborhood as extra feature columns.
- "DFS for fewshot + root" = `E3` (`dfs_fewshot_depth=1,
  dfs_root_depth=2`): root nodes additionally carry exact two-hop
  aggregates alongside the sampled MPNN.
- The root-only cell (`E2`) is not in this plan's scope but the
  tooling supports it (`CELLS="E2"`) if a follow-up needs to separate
  the two effects.
- **Training uses the native Griffin decoder automatically** — that is
  what `hmaintask_combine.py` trains and validates with; no extra flag
  is needed. TabPFN enters only at evaluation time and is frozen (no
  gradient into it or from it).

## 2. Hypotheses being tested (register before running)

| Comparison | Question | Expected signature |
|---|---|---|
| 1 vs 2 (and 4 vs 5) | Does leaf enrichment help? | Gains concentrated on label-context-driven tasks (churn-style); leaf batches are the same size, so any gain is "free" |
| 2 vs 3 (and 5 vs 6) | Does root enrichment add on top? | E3's deterministic root aggregates should **reduce seed-std** relative to E1 even at equal mean; if E3 ≈ E1, root aggregates are subsumed by leaf info flowing over the fewshot edge |
| 1 vs 4 (E0 pair) | Reproduces the known SMPNN result | SMPNN ≈ vanilla mean, lower variance (o1→o2); commerce would reverse it — this plan stays in the others domain |
| 2 vs 5, 3 vs 6 | Do DFS and SMPNN compose? | If both independently reduce variance, the combination should be the most stable cell of all six |

Primary metric: per-task AUROC on the held-out **test split**, reported
as **mean ± std over 3 seeds**. Judge variance as seriously as means —
the project's 5-seed history shows single-seed deltas in this regime
are noise ([SMPNN_MASTER_RESULTS.md](SMPNN_MASTER_RESULTS.md)).

## 3. Prerequisites (one-time)

```bash
# Environment (see COLLABORATOR_HANDOFF.md for the full conda recipe)
conda activate griffin
git clone https://github.com/ViswanathGanapathy/Griffin-LLM.git Griffin
cd Griffin && git checkout smpnn-ablations

# Data: datasets/joint-v65 (~5 GB, not in git — get from the project lead)
ls datasets/joint-v65/metanode.yaml   # must exist

# DFS artifacts (CPU-ok; deterministic incl. name embeddings as of ba7a993).
# Either all types:
python dataconverterdfs.py datasets/joint-v65
# ...or the o1/o2 scope used here (40 types; command in logs/dfs-convert-o1o2.log)

# Sanity gate — DO NOT proceed unless everything passes:
python test_dfs_vanilla.py            # 6-test battery + T3b
```

TabPFN v2.5 weights download on first use (the `tabpfn` package
handles it; a free license token may be prompted — priorlabs.ai).

## 4. Phase 1 — training (18 runs)

6 configs × 3 seeds (42, 43, 44) on **others-1**, 20 epochs each,
native decoder, per-epoch validation on the RelBench valid split
selecting `best_checkpoint`.

Everything is driven by [run_smpnn_dfs.sh](../../run_smpnn_dfs.sh):

```bash
# All 6 configs, 3 seeds (skip-if-checkpoint-exists; FORCE=1 overrides):
CELLS="E0 E1 E3" BACKBONES="vanilla smpnn" SEEDS="42 43 44" ./run_smpnn_dfs.sh

# Or split across 2 GPUs:
CUDA_VISIBLE_DEVICES=0 CELLS="E0 E1 E3" BACKBONES="vanilla" ./run_smpnn_dfs.sh &
CUDA_VISIBLE_DEVICES=1 CELLS="E0 E1 E3" BACKBONES="smpnn"   ./run_smpnn_dfs.sh &
wait
```

Outputs: `checkpoints/smpnn-dfs-s<seed>-<cell>-<backbone>/best_checkpoint`,
logs under `logs/smpnn-dfs/`.

Budget: ~3.5–4.5 h/run on A100/H100 → **~65–80 GPU-h total** (halved
per-GPU when split).

Cost saver: if the lead's existing `smpnn-multiseed-*` vanilla-4 and
SMPNN-6 α=1e-2 others-1 checkpoints (seeds 42–44) are available, the
two E0 rows need no retraining — the eval script picks the vanilla
ones up automatically (`REUSE_E0=1`), reducing Phase 1 to 12 runs.

## 5. Phase 2 — evaluation with TabPFN v2.5

**Iron rule:** every checkpoint is evaluated with the **same
`--dfs_*` flags it was trained with** — the DFS column set is part of
the encoder's input contract, and a mismatch produces plausible but
wrong numbers with no error. The script enforces the pairing.

```bash
# In-distribution: others-1 test split
TABPFN_VERSION=v2.5 EVAL_FAMILY=others-1 \
  CELLS="E0 E1 E3" BACKBONES="vanilla smpnn" ./run_smpnn_dfs_eval.sh

# Transfer: others-2 test split (o1→o2, the project's headline direction)
TABPFN_VERSION=v2.5 EVAL_FAMILY=others-2 \
  CELLS="E0 E1 E3" BACKBONES="vanilla smpnn" ./run_smpnn_dfs_eval.sh
```

36 eval runs (6 configs × 3 seeds × 2 eval families), ~5 min each →
**~3 GPU-h**. Per-cell logs land in `logs/smpnn-dfs-eval/`, tagged
with the TabPFN version.

What the script does per cell: extract frozen-encoder embeddings for
train/test rows → feed **raw 512-d** embeddings (no projection,
`--no_icl_projection --probe_epochs 0`) to TabPFN v2.5 (context
subsampled to 30K, 8-estimator ensemble) → report
`test_metric/<task>` per task.

**v2.5-specific caveat:** TabPFN v2's pretraining nominally covers up
to ~500 features; our raw embeddings are 512-d. The wrapper raises the
cap via `--tabpfn_inference_config` (`MAX_NUMBER_OF_FEATURES: 600`),
which is how the v3 runs already operate — but v2.5 is running
slightly outside its pretraining envelope. Run ONE cell first
(`SEEDS=42 CELLS=E0 BACKBONES=vanilla`) and confirm it completes with
sane AUROCs before launching the full grid. If v2.5 refuses or
degrades badly, fallbacks in order: (a) `--icl_proj_dim 128` with
`--probe_epochs 5` (learned projection to 128-d — changes the
pipeline, apply to ALL cells equally); (b) report v3 alongside v2.5.

Optional extra (cheap, recommended): the training logs already contain
**native-decoder** validation metrics per epoch; additionally running
the native head on the test split (`--head default`, drop the
`--icl_*`/`--tabpfn_*` flags) gives a decoder-independent check of any
DFS effect for ~20 extra minutes total.

## 6. Phase 3 — aggregation and reporting

```bash
python collect_smpnn_results.py \
    --logs logs/smpnn-dfs-eval/*.log \
    --out smpnn_dfs_ablation_results.csv
```

Report template (one table per eval family; per task and macro-avg):

| Config | Task AUROC mean | ±std | Δ mean vs E0 (same backbone) | std ratio vs E0 |
|---|---|---|---|---|
| E0-vanilla | | | — | — |
| E1-vanilla | | | | |
| E3-vanilla | | | | |
| E0-smpnn | | | — | — |
| E1-smpnn | | | | |
| E3-smpnn | | | | |

Decision rules (agreed up front):
- **|Δ mean| > 0.02 with consistent sign across all 3 seeds** → decisive.
- **std ratio ≤ 1/3 at |Δ mean| ≤ 0.01** → variance win (report as a
  first-class result, per the SMPNN precedent).
- Anything within 1 std → tie; do not narrativize it.
- If any comparison is borderline (between 1 and 2 std), extend that
  pair to seeds 45–46 before concluding.

Send back: `smpnn_dfs_ablation_results.csv` + `logs/smpnn-dfs-eval/`
(push to the branch, or email).

## 7. Pitfalls checklist

| Pitfall | Guard |
|---|---|
| Eval flags ≠ training flags | Scripts pair them; never hand-edit a single eval command without checking the cell's training flags |
| DFS artifacts missing/stale | `FileNotFoundError` names the converter; artifacts generated before commit `ba7a993` must be regenerated (name-embedding determinism fix) |
| TabPFN v2.5 512-feature envelope | Single-cell smoke test first (§5); fallbacks listed there |
| Single-seed conclusions | 3-seed minimum here; extend borderline pairs to 5 |
| OOM at training | `--batchsize 256 → 128`; DFS adds 33–41 columns on enriched rows |
| Comparing across eval families | o1-test and o2-test tables are different populations — never mix rows |

## 8. Compute summary

| Phase | Runs | GPU-h |
|---|---|---|
| Training (full) | 18 | ~65–80 |
| Training (reusing E0-vanilla ckpts) | 12–15 | ~45–60 |
| TabPFN v2.5 eval | 36 | ~3 |
| Optional native-head eval | 36 | ~1 |
| **Total** | | **~50–85** |
