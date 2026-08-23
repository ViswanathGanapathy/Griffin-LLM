# Commerce-1 → Commerce-2 Runbook: Vanilla Griffin + DFS, TabICL v2 Head

Step-by-step instructions for the first commerce experiment cycle:
convert DFS artifacts (Step 0), train **vanilla Griffin** on
**commerce-1**, evaluate zero-shot on **commerce-2** with **TabICL v2**
as the frozen in-context head.

Branch: `smpnn-ablations` of `ViswanathGanapathy/Griffin-LLM`.
Companion docs: [DFS_INTEGRATION_WALKTHROUGH.md](DFS_INTEGRATION_WALKTHROUGH.md)
(code-level flow, §0a = cutoff semantics),
[DFS_GRIFFIN_DESIGN.md](DFS_GRIFFIN_DESIGN.md) (design rationale),
[DFS_ABLATION_PLAN.md](DFS_ABLATION_PLAN.md) (full 6-cell matrix).

The task families (resolved via `task_names.yaml`):

| Family | Tasks |
|---|---|
| `commerce-1` | diginetica-downsample-ctr, rel-hm-item-sales, rel-hm-user-churn, retailrocket-cvr, seznam-charge, seznam-prepay |
| `commerce-2` | amazon-churn, amazon-rating, outbrain-small-ctr, rel-avito-ad-ctr, rel-avito-user-clicks, rel-avito-user-visits |

---

## 0. Why Step 0 exists (read this once)

DFS values for a supervised task are **computed online** in the
DataLoader workers, at each seed's task cutoff τ. But "online" covers
only the aggregation arithmetic. The same code
(`dfscore.compute_d1_at` / `compute_d2_at`) has **two call sites** —
the converter calls it offline over *all rows of a type*, the loader
calls it online over *the batch's rows* — and the online site depends
on four artifacts only the offline pass can produce:

| Artifact | File | Why it must be offline |
|---|---|---|
| Column layout | `dfs/metadfs.yaml` | Fixed column list/order per type. `getdfsfeat` looks the type up here; **a type that is missing silently gets NO DFS columns** — an "E1" run without Step 0 trains without error but is actually E0. Also binds column order to checkpoints. |
| Cutoff-mode normalization stats (`d1_at`/`d2_at`) | `dfs/metadfs.yaml` | Raw aggregates are z-normed into floatemb's ~N(0,1) range with **population** mean/std, fitted over ~20K (row, τ) pairs sampled from the task tables. Batch statistics would make the same row encode differently per batch and diverge train vs test. Fitted by `--cutoff_stats`; the loader **hard-refuses** to serve task cutoffs without them. |
| Own-timestamp store | `dfs/<type>/d1.pt`, `d2.pt` | Speed, not correctness: temporal rows' d1 at their own time. Served when provably identical to the τ-evaluation (τ == own ts: amazon-rating, retailrocket-cvr, seznam-\*), and reused for temporal d2 hubs (own window ⊂ τ-window). |
| Column-name embeddings | `dfs/dfsfeatnameemb.pt` | Deterministic (md5-seeded) name vectors the column attention uses; must be identical across runs/reloads. |

**Worked example** (rel-hm-user-churn; root `rel-hm-customer` is a
non-temporal dimension table; neighbor `rel-hm-transactions` is
temporal with a `price` column). Batch seed = customer #123 at
τ = 2020-09-07:

* **Online, every batch:** `getedge(#123, INF, τ)` finds the 57
  transactions with ts < τ; scatter count/mean/max →
  `count = 57 → log1p ≈ 4.06`, `mean(price) = 0.031`. At depth 2, the
  57 transactions' own d1 vectors are **looked up** from the offline
  store (each transaction's time < τ, so its window is a safe subset);
  scatter-mean → d2. Values are z-normed with the offline `d1_at`
  stats — e.g. (4.06 − 3.2)/1.1 = 0.78 — then embedded and appended
  as extra columns.
* **Offline, once:** the layout that says those columns exist and in
  what order; the 3.2/1.1 stats; the transactions' own-ts d1 store;
  the name embeddings.

Decision rule per type per batch (`hdataset._dfs_values`): *is the
requested cutoff exactly what the store was computed at?* Yes (cutoff
None, or temporal type with τ == every row's own ts) → read the store,
zero online work. No (every non-temporal root like `customer` or
`drivers`) → compute now for these rows. A per-τ store is impossible —
tasks carry up to ~384K distinct cutoffs, which is exactly why the old
fully-precomputed design leaked (see DFS_INTEGRATION_WALKTHROUGH.md §0a).

---

## Step 0 — Convert DFS artifacts (v2, with cutoff stats)

Needed **before any E1/E2/E3 cell**; E0 needs none of this. Both
families are needed — commerce-1 for training, commerce-2 because the
eval builds DFS columns for its batches at eval time. Both commands
merge into the same `datasets/joint-v65/dfs/`. Label-holder
(`is_target`) types are deliberately excluded — they are never DFS
sources by default.

```bash
conda activate griffin
cd ~/Griffin

# commerce-1 (27 non-target node types), ~run first
python dataconverterdfs.py datasets/joint-v65 --batch_rows 32768 \
  --nodetypes diginetica-downsample-Click diginetica-downsample-ClickQueryTest diginetica-downsample-Orders diginetica-downsample-Product diginetica-downsample-ProductNameToken diginetica-downsample-Purchase diginetica-downsample-Query diginetica-downsample-QueryResult diginetica-downsample-QuerySearchstringToken diginetica-downsample-Session diginetica-downsample-Token diginetica-downsample-User diginetica-downsample-View rel-hm-article rel-hm-customer rel-hm-transactions retailrocket-Category retailrocket-Item retailrocket-ItemAvailability retailrocket-ItemCategory retailrocket-ItemProperty retailrocket-View retailrocket-Visitor seznam-Client seznam-Dobito seznam-Probehnuto seznam-ProbehnutoMimoPenezenku \
  --cutoff_stats --tasks diginetica-downsample-ctr rel-hm-item-sales rel-hm-user-churn retailrocket-cvr seznam-charge seznam-prepay \
  2>&1 | tee logs/dfsconv-commerce-1.log

# commerce-2 (25 non-target node types)
python dataconverterdfs.py datasets/joint-v65 --batch_rows 32768 \
  --nodetypes amazon-Customer amazon-Product amazon-Review outbrain-small-Click outbrain-small-DocumentsCategory outbrain-small-DocumentsEntity outbrain-small-DocumentsMeta outbrain-small-DocumentsTopic outbrain-small-Event outbrain-small-Pageview outbrain-small-PromotedContent outbrain-small-User rel-avito-AdsInfo rel-avito-Category rel-avito-IPInfo rel-avito-Location rel-avito-PhoneRequestsStream rel-avito-SearchInfo rel-avito-SearchStream rel-avito-UserAgent rel-avito-UserAgentFamily rel-avito-UserAgentOS rel-avito-UserDevice rel-avito-UserInfo rel-avito-VisitStream \
  --cutoff_stats --tasks amazon-churn amazon-rating outbrain-small-ctr rel-avito-ad-ctr rel-avito-user-clicks rel-avito-user-visits \
  2>&1 | tee logs/dfsconv-commerce-2.log
```

Practical notes (15 GB RAM machine):

* Run the two sequentially, not in parallel. The rel-avito / outbrain
  stream tables are the slow part; degree-skewed hubs are handled by
  adaptive chunk-halving, but expect hours, not minutes.
* **Do not trust the exit code alone** — a killed background run has
  produced "exit 0" before. Verify the inventory:

  ```bash
  ls datasets/joint-v65/dfs/rel-hm-customer/d1.pt \
     datasets/joint-v65/dfs/amazon-Customer/d1.pt
  grep -c "d1_at" datasets/joint-v65/dfs/metadfs.yaml   # > 0
  tail -3 logs/dfsconv-commerce-1.log                   # "Done. ... Cutoff-mode stats fitted for: [...]"
  ```
* If d2 conversion crawls on avito's dimension hubs, add
  `--d2_nontemporal_hubs skip` (zero-fills those d2 columns; never
  falls back to leaky all-time values).
* Old (v1) artifacts are refused by the loader with a regeneration
  message — that is intended (they were computed with leaky
  own-timestamp/all-time semantics).

---

## Step 1 — Train vanilla Griffin on commerce-1

`BACKBONES="vanilla"` = original Griffin (`--num_mp 4`, no SMPNN
blocks). The E0 baseline needs **no** DFS artifacts — start it while
Step 0 runs:

```bash
# Baseline (no DFS) — start immediately
TRAIN_FAMILY=commerce-1 BACKBONES="vanilla" CELLS="E0" SEEDS="42" ./run_smpnn_dfs.sh

# DFS cells — after Step 0's commerce-1 conversion is verified
TRAIN_FAMILY=commerce-1 BACKBONES="vanilla" CELLS="E1 E3" SEEDS="42" ./run_smpnn_dfs.sh
```

* Cells: E0 = no DFS; E1 = `dfs_fewshot_depth 1` (leaves only);
  E2 = `dfs_root_depth 2` (roots only); E3 = both.
* Checkpoints: `checkpoints/smpnn-dfs-commerce-1-s42-{E0,E1,E3}-vanilla/best_checkpoint`,
  each with a `dfs_config.json` binding it to its DFS flags (loading
  with mismatched flags fails fast; `--dfs_override` to force).
* Logs: `logs/smpnn-dfs-commerce-1/`. Add seeds via
  `SEEDS="42 43 44"`; add the SMPNN row later with
  `BACKBONES="smpnn"` (or `"smpnn vanilla"` for both).
* Training uses Griffin's native decoder; the ICL head enters only at
  evaluation.

---

## Step 2 — Evaluate on commerce-2 with TabICL v2

```bash
TRAIN_FAMILY=commerce-1 EVAL_FAMILY=commerce-2 ICL_HEAD=tabicl \
  BACKBONES="vanilla" CELLS="E0" SEEDS="42" ./run_smpnn_dfs_eval.sh
# after E1/E3 training + commerce-2 Step 0:  CELLS="E0 E1 E3"
```

This runs the frozen trained encoder over each commerce-2 task's test
split and hands the 512-d embeddings straight to TabICL
(`--no_icl_projection --probe_epochs 0` — no learned projection, no
probe fine-tuning). Logs/results:
`logs/smpnn-dfs-eval/commerce-1-s42-E0-vanilla-commerce2-tabiclv2.log`.

What the TabICL head does (see `tabular_heads.py::TabICLHead`):

* **Separate checkpoints for classification vs regression — automatic.**
  A fresh `TabICLHead(task_type=...)` is created per task from
  `metatask`'s `task_type`. Classification tasks resolve the `v2`
  alias to `tabicl-classifier-v2-20260212.ckpt`; regression tasks
  (amazon-rating, rel-avito-ad-ctr) let the tabicl library load its
  own default **regressor** weights — forcing the classifier file into
  the regressor loads but then crashes, so don't. You manage nothing
  by hand; `TABICL_VERSION=v2` is the default.
* **In-context examples:** the train split's embeddings are stored as
  ICL context, subsampled to `--icl_max_context` when larger —
  stratified per class for classification, random for regression. The
  scripts use **30,000** so TabICL and TabPFN numbers share a context
  budget (TabICL's practical ceiling in the wrapper is 48,000; raising
  it is a follow-up ablation, not the first run).
* **Bagging:** built in — `--icl_n_estimators 8` ensemble members over
  feature shuffles, predictions averaged. No extra manual bagging; for
  a context-resampling bag, rerun with different `SEEDS` and average.

To compare heads on the same checkpoints, rerun with
`ICL_HEAD=tabpfn` (tags encode the head, so both evals coexist).

---

## Reading the results

Per-task metrics are printed at the end of each eval log
(metric per `metatask`: AUROC for the binary tasks, MAE for
regression). The comparisons this cycle supports:

* **E1-vanilla − E0-vanilla on commerce-2** — does DFS on fewshot
  leaves improve zero-shot transfer?
* **E3-vanilla − E1-vanilla** — marginal value of root-node DFS.
* Same rows with `ICL_HEAD=tabpfn` — head sensitivity.

All commerce artifacts are generated post-cutoff-fix (format v2), so
these are leak-free numbers from the start — there is no "E-leaky" row
to worry about for commerce, unlike the historical others-1 results
(see the banner in DFS_ABLATION_PLAN.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: ... metadfs.yaml` | Step 0 never ran. |
| `RuntimeError: ... format v1 ... requires v2` | Old artifacts; rerun Step 0 (the message includes the command). |
| `RuntimeError: ... no cutoff-mode normalization stats ('d1_at' ...)` | Step 0 ran without `--cutoff_stats --tasks ...`; rerun with them. |
| E1 run trains but metrics match E0 exactly | The root/leaf types are missing from `metadfs.yaml` (silent no-op) — check the type list you converted. |
| `RuntimeError: DFS config mismatch` at checkpoint load | Checkpoint trained with different `dfs_*` flags; use matching flags or `--dfs_override`. |
| Converter killed mid-run, log stops silently | RAM; lower `--batch_rows`, run families sequentially, verify inventory as above. |
