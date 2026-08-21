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

1. **Temporal leakage.** DFS features are computed at each node's own
   timestamp with a **strict `<` cutoff** — a neighbor row is included
   iff `nb_ts < node_ts`. Implementation reuses Griffin's existing
   `getedge` timestamp mask (`adj.masked_fill_(adjtimestamp >= timestamp, -1)`),
   which is exactly this rule. Strict inequality also excludes
   same-timestamp rows (including, transitively, the row itself).
   Non-temporal node types (all timestamps = int64-min) skip the
   cutoff.

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
