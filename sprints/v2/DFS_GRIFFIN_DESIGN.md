# DFS + Griffin: Deep Feature Synthesis as a Complement to Sampled Message Passing

Design and implementation notes for integrating RDBLearn-style Deep
Feature Synthesis (DFS) into Griffin's MPNN pipeline.

**References:**
- RDBLearn (arXiv 2602.18495): DFS-flattened relational data + frozen
  single-table ICL head (TabPFN/TabICL) — competitive with RelBench
  GNNs with no graph learning at all.
- fastdfs (github.com/HKUSHXLab/fastdfs): fast DFS engine with
  `max_depth` control, cutoff-time filtering, aggregation primitives
  (mean, max, count, sum, mode).

---

## 1. Why DFS + MPNN, not DFS vs MPNN

RDBLearn shows that *exact, unsampled* aggregate features feeding a
frozen ICL head match learned GNN encoders on many RelBench tasks.
Griffin's RMPNN is the mirror image: *learned, task-conditioned*
aggregation over *fanout-sampled* subgraphs. The two are duals:

| Axis | DFS aggregates | Griffin RMPNN |
|---|---|---|
| Neighborhood coverage | Exact (all neighbors) | Sampled (fanout=20/hop) |
| Aggregation weights | Fixed primitives | Learned, gated, task-conditioned |
| Train-time cost | Zero (precomputed) | Per-batch sampling + MP |
| Output variance | **Deterministic** | Stochastic across seeds |

Our 5-seed study ([SMPNN_MASTER_RESULTS.md](SMPNN_MASTER_RESULTS.md))
identified seed-variance as the dominant reliability problem of the
sampled-MPNN pipeline (Vanilla-4 seed-std up to ±0.066). DFS features
are deterministic — injecting them attacks that variance at its source
(sampling noise in the aggregation path). This is the specific
hypothesis the hybrid tests.

## 2. The two integration steps

### Step 1 — One-hop DFS for fewshot leaves (`--dfs_fewshot_depth 1`)

**Current behaviour** (confirmed in code): fewshot nodes are bare
leaves. `LoaderWrapper.fewshotsubgraph` forces `hop=0`
(hloaderwrapper.py), so a fewshot node carries only its own columns +
label, and receives no information about its own relational
neighborhood. Expanding leaves to hop≥1 was previously rejected
because it multiplies batch node count by the leaf count × per-hop
fanout.

**Change**: append each leaf's precomputed **depth-1 DFS columns**
(count / mean / max per relation over strictly-past neighbors) to its
feature set at load time. The leaf arrives carrying a summary of its
one-hop neighborhood at zero graph-expansion cost. Because Griffin
builds initial embeddings by column-name-conditioned attention over
(name-emb, value-emb) pairs, extra columns require **no model change**
and existing checkpoints still load.

### Step 2 — Two-hop DFS for root nodes (`--dfs_root_depth 2`)

Append **depth-2 DFS columns** (aggregates of neighbors' depth-1
aggregates, no-backtrack) to the root node type in the main sampled
subgraph. Every root row then enters MP layer 0 with:

    [ raw columns | exact 2-hop DFS summary ]  ⊕  sampled subgraph MP on top

The column attention learns per-task how much to weight the exact
fixed aggregates vs the learned sampled ones. Feasible with the same
mechanism as Step 1; the only extra care is masking (below).

## 3. Correctness rules (where naive DFS breaks)

1. **Temporal leakage.** DFS features are evaluated at the **query
   cutoff τ** — the task timestamp of the seed a row was reached from —
   with a **strict `<` filter**: a neighbor row is included iff
   `nb_ts < τ`. τ is propagated to every depth (fastdfs / featuretools
   `cutoff_time` semantics). Implementation reuses Griffin's existing
   `getedge` timestamp mask (`adj.masked_fill_(adjtimestamp >= timestamp, -1)`),
   the same mask the MPNN sampler uses, so the exact DFS summary and
   the sampled neighborhood always describe the same temporal window.
   The precomputed store holds *own-timestamp* values and is used only
   when it is provably identical to the τ evaluation (cutoff is None,
   or the type is temporal and τ equals every row's own timestamp —
   Completion pretraining and the many RelBench tasks whose cutoff IS
   the row time); otherwise features are computed online at τ and
   z-normalized with dedicated cutoff-mode stats (`d1_at`/`d2_at`,
   fitted at sampled task cutoffs via `--cutoff_stats`).
   **Non-temporal node types never skip the cutoff for supervised
   tasks** — that was the original leak (a driver's all-history
   aggregates include the evaluation window). They fall back to
   all-time only when no query time exists at all (Completion-style
   `cutoff=None`), where the label is a present cell, not future
   information. Label-holder (`is_target`) relations are excluded from
   DFS sources by default (`--include_target_rels` to opt in, refused
   without `--cutoff_stats`).

2. **Depth-2 self-leakage (A→B→A backtracking).** When aggregating
   B's depth-1 features into A, features of B that were derived from
   the B→A reverse relation are **excluded** (standard DFS
   no-backtrack rule). Without this, a root's own label-bearing column
   can leak into its own depth-2 feature through any neighbor.

3. **Feature explosion** (RDBLearn's noted failure mode). Controls:
   - Only float-typed columns are aggregated (mean of category IDs or
     text-dedup indices is meaningless; categorical/text columns
     contribute only to the `count` primitive).
   - `--max_cols_per_rel` caps value-columns per relation (default 8).
   - Depth 2 uses mean only (no max-of-max explosion) and
     `--max_d2_feats` caps the total (default 32, kept by variance).
   - All features are **z-normalized** over the node population —
     Griffin's `floatenc` was trained on ~N(0,1) inputs
     (hFloatEmb.py), so this also matches the encoder's operating
     range.

4. **is_target seed types** are never DFS *sources* (they have no
   outgoing adjacency in Griffin), but they do appear as neighbors —
   their numeric label columns are aggregated under the strict-past
   rule, which is legitimate temporal feature engineering ("mean past
   churn among my neighbors"), the same signal RDBLearn exploits.

## 4. Implementation map

```
dataconverterdfs.py        NEW   offline DFS computation (depth 1 + 2)
  reads:  metanode/metaadj + node/*/feat + edge/*/adj (+ timestamps)
  writes: dfs/<nodetype>/d1.pt          (num_nodes, C1) float32, z-normed
          dfs/<nodetype>/d2.pt          (num_nodes, C2)
          dfs/metadfs.yaml              names + norm stats per type/depth
          dfs/dfsfeatnameemb.pt         name -> 512-d embedding

hdataset.py                MOD   Graph.load_dfs() lazy loader;
                                 subgraph(..., dfs_depth=0) appends DFS
                                 columns (via floatemb float path) to the
                                 ROOT node type's features + name embs

hloaderwrapper.py          MOD   subgraphargs carries dfs_depth (root) and
                                 dfs_fewshot_depth (popped by wrapper);
                                 fewshotsubgraph overrides dfs_depth for
                                 leaves; target_feat_mask padded with True
                                 for appended DFS columns at the two
                                 mask[0][mapping] sites + retrieval slicing

hmaintask_combine.py       MOD   --dfs_root_depth {0,1,2}, --dfs_fewshot_depth {0,1,2}
hmaintask_combine_llm.py   MOD   same flags (eval path)

run_smpnn_dfs.sh           NEW   experiment matrix (see §6)
```

**Name embeddings for DFS columns**: composed from existing
embeddings — `emb("dfs1 mean rel col") := normalize(edgenameemb[rel] +
featnameemb[col] + primitive_tag)` by default (deterministic, no GPU);
`--nomic` switches to true Nomic encoding of the human-readable
feature name, matching `dataconverterpost.py`.

**Why not call fastdfs directly?** Griffin's processed datasets store
CSR-style adjacency + Arrow feature tables, not raw DataFrames.
`Node.getedge(idx, INF, timestamp)` already returns the exact
timestamp-filtered full neighborhood — one-hop DFS with
mean/max/count is precisely a fixed-weight unsampled RMPNN pass over
it. Reimplementing natively (~200 lines, torch scatter ops) avoids a
DataFrame round-trip and a new dependency while keeping semantics
identical to fastdfs `max_depth∈{1,2}` with cutoff times.

## 4b. Semantics vs featuretools / fastdfs (verified parity + deliberate deviations)

`test_dfs_parity.py` runs a two-phase numerical parity test against
**real featuretools** (isolated venv, so the griffin env is untouched):
a synthetic users←orders←items RDB with 25% NaNs and per-row temporal
cutoffs, comparing Count / Mean / Max at depth 1 and Mean-of-Mean at
depth 2. Status: **all aggregates match featuretools 1.31.0** to
float32 precision (max|Δ| ≈ 2e-6).

Verified-identical semantics:

| Aspect | featuretools / fastdfs (SQL) | Ours |
|---|---|---|
| Mean with NULLs | skip NULLs (`sum/count` over non-null) | same (fixed — early version zero-filled, biasing means toward 0) |
| Max with NULLs | skip NULLs | same (fixed — early version mapped NaN→0, corrupting all-negative columns) |
| Count | counts rows | same |
| Depth-2 shape | agg of child's agg (e.g. `MEAN(orders.MEAN(items.price))`) | same |
| Empty aggregate | NaN → caller-imputed | 0 before z-normalization |

Deliberate deviations (each documented and defensible):

1. **Cutoff inclusivity.** featuretools includes rows at
   `time <= cutoff`; we use strict `nb_ts < node_ts` (Griffin's
   `getedge` mask). Strict is safer against same-timestamp label
   leakage; the parity test aligns them via `cutoff = ts − 1s`.
2. **Depth-2 temporal window — temporal hubs only.** featuretools
   recomputes the whole feature tree at the *target's* cutoff; for a
   **temporal** hub we aggregate the hub's *precomputed* own-timestamp
   d1. Since `ts(hub) < τ`, the hub's window is a subset of the
   target's — strictly conservative, never leaky, and free to look up.
   A **non-temporal** hub (a dimension row) has no own timestamp; its
   precomputed d1 would be all-time and leak the future straight
   through the dimension table, so it is **recomputed at the querying
   row's cutoff** (`compute_d2_at`, deduplicated over unique
   (hub, cutoff) pairs with a budget guard; `--d2_nontemporal_hubs
   skip` zero-fills instead — it never falls back to the all-time
   value).
3. **Primitive set.** count/mean/max only at d1, mean-only at d2 (vs
   featuretools' default sum/std/skew/min/mode/… and transform
   primitives) — the explosion-control choice from §3.
4. **Direct features.** featuretools copies parent attributes across
   many-to-one relations as "direct features"; our mean over a
   singleton neighbor set equals the same copy (plus a constant
   count=1), so these are subsumed, with mild redundancy.
5. **No cross-library code reuse.** The implementation is native
   (torch scatter over Griffin's Arrow/CSR structures) — see §4's
   "Why not call fastdfs directly?".

## 5. Backward compatibility

- `dfs_depth=0` / `dfs_fewshot_depth=0` (defaults) → bit-identical to
  current behaviour; no DFS artifacts required on disk.
- Enabling flags on a dataset without `dfs/` artifacts → clear error
  telling you to run `dataconverterdfs.py` first.
- No model-parameter changes: column attention is shape-agnostic, so
  **existing checkpoints load unchanged** and can even be *evaluated*
  with DFS columns switched on (the attention simply attends over more
  columns — though embeddings shift, so treat that as a new variant,
  not a free lunch).

## 6. Experiment plan (run_smpnn_dfs.sh)

Backbone: SMPNN-6 α=1e-2 (current best on o1→o2) and Vanilla-4
(baseline), on others-1, 3 seeds initially:

| Tag | dfs_fewshot | dfs_root | Tests |
|---|---|---|---|
| E0 baseline | 0 | 0 | reproduces existing numbers |
| E1 fewshot-dfs1 | 1 | 0 | Step 1: leaf enrichment alone |
| E2 root-dfs2 | 0 | 2 | Step 2: root enrichment alone |
| E3 both | 1 | 2 | additivity |

Predictions worth registering up front: (a) E1 should mainly help
tasks where fewshot label context matters (retrieval-style, e.g.
churn); (b) E2 attacks sampling variance, so its clearest signature
would be **reduced seed-std** relative to E0 even at equal mean —
the same criterion our SMPNN study used; (c) if E3 ≈ E2, the root
aggregates subsume the leaf ones (leaves' info flows through the
fewshot edge anyway).

## 7. Open questions / future work

- **Learned DFS gating**: a per-DFS-column learned scale (à la SMPNN's
  α) initialised near zero would let the model ignore unhelpful
  aggregates gracefully. Column attention already provides implicit
  gating; explicit gating is a follow-up ablation.
- **Depth-2 primitive set**: mean-only is conservative; fastdfs
  supports max/count at depth 2 as well.
- **DFS for the ICL head directly**: RDBLearn's own setting —
  concatenating DFS features to Griffin *embeddings* at the
  TabPFN/TabICL input (bypassing the encoder) is a third integration
  point, and a strong baseline our matrix should eventually include:
  `[Griffin emb | DFS] → TabPFN` vs `Griffin-with-DFS emb → TabPFN`.
