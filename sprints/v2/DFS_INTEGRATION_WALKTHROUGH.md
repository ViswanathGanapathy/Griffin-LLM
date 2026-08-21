# DFS × MPNN Integration Walkthrough — Code Review Guide

A region-by-region trace of how Deep Feature Synthesis is integrated
into Griffin's MPNN pipeline, for **(1) fewshot-fanout leaves** and
**(2) root nodes**, with exact code anchors, the invariant each region
must uphold, and which test proves it. Written for careful review.

Line numbers are exact as of commit `44e821e` on `smpnn-ablations`.

---

## 0. The design in one paragraph

DFS features are **precomputed offline per node** (converter) and
**appended as extra feature columns at load time** (runtime). Nothing
in the model changes: Griffin's layer-0 embedding is column-name-
conditioned attention over `(name-emb, value-emb)` pairs, which is
shape-agnostic in the column count. The entire integration therefore
lives in three places — the converter that makes the artifacts, the
`Graph.subgraph` tail that appends columns, and the loader that (a)
routes a separate DFS depth to the fewshot-leaf path and (b) pads the
target-column mask for the extra columns. The MPNN then runs
unchanged on richer node representations.

```
OFFLINE                          RUNTIME (per batch)
dataconverterdfs.py              hmaintask_combine*.py  (flags)
  d1.pt / d2.pt per node type      └─ subgraphargs {dfs_depth, dfs_fewshot_depth}
  metadfs.yaml (names+stats)          └─ hloaderwrapper.py  (routing + masks)
  dfsfeatnameemb.pt                       └─ hdataset.py Graph.subgraph (append)
                                              └─ hmodel*.py  (UNCHANGED)
```

### 0b. Where DFS and the MPNN updates combine

DFS and native message passing ARE combined — at the
**representation level**, not in the update rule:

```
vanilla:      x0 = Attn( taskfeat ; native columns )
DFS-Griffin:  x0 = Attn( taskfeat ; native ∪ DFS columns )
then, unchanged:  x ← MPNN/SMPNN layer updates on x
```

Two properties make this deeper than "input features":

1. **Per-layer re-attention.** Griffin's forward loop
   ([hmodel_smpnn.py:338-353](../../hmodel_smpnn.py#L338-L353); same
   structure in hmodel.py) re-merges the raw column set at EVERY MP
   layer via `nodefeataggr[i]`, conditioned on a task prompt refreshed
   from the evolving node states (`lintask[i]`). DFS columns are
   therefore consulted L times with layer-specific, task-conditioned
   weights — the column attention is the (learned) gate that decides
   how much exact-aggregate vs learned-sampled signal each layer uses.
2. **Graph propagation.** Once merged into `x`, DFS-derived signal
   travels through messages like any feature: leaf summaries reach the
   target over the `"fewshot"` edge; enriched root-type neighbors'
   summaries travel over relational edges.

Alternatives deliberately not taken: a fourth branch in the layer
equation (`+ g_dfs · MLP(dfs)`) — extra parameters, breaks checkpoint
compatibility, redundant with per-layer re-attention; and head-level
concatenation (`[Griffin emb | DFS] → TabPFN`) — that is RDBLearn's
own configuration, tracked as a future baseline in
[DFS_GRIFFIN_DESIGN.md §7](DFS_GRIFFIN_DESIGN.md).

---

## 1. Phase 0 — offline artifact computation (shared by both paths)

File: [dataconverterdfs.py](../../dataconverterdfs.py)

### 1.1 The aggregation core — `aggregate_batch` (lines 108–142)

Computes, for one (relation, row-batch): `count` over neighbor rows,
and NaN-skipping `mean` / `max` per float column, via torch scatter.

**Invariants to verify:**
- Mean = `sum(non-NaN) / count(non-NaN)` — NOT divided by total rows.
- Max is taken over non-NaN values only (NaN→−inf before `amax`,
  −inf→0 afterwards for empty sets).
- `count` counts neighbor rows (featuretools `Count` semantics), and
  is later `log1p`'d (line ~216) before z-normalization.

**Proof:** [test_dfs_parity.py](../../test_dfs_parity.py) — numerical
match vs real featuretools 1.31.0 (isolated venv), incl. a 25%-NaN
negative-mean column. Run: phase `a` in the griffin env, phase `b` in
a featuretools venv.

### 1.2 Depth 1 — `compute_depth1` (lines 144–219)

Per node type A: enumerate its relations from `metanode[A]["in"]+["out"]`,
pick ≤`max_cols_per_rel` float columns of each neighbor type, then for
each row-batch call **`node.getedge(idx, INF, ts)`** (line 197).

**This one line is the temporal-correctness keystone.** `getedge` is
Griffin's own neighbor fetch; with a timestamp it applies
[hdataset.py:99](../../hdataset.py#L99)
`adj.masked_fill_(adjtimestamp >= timestamp, -1)` — i.e. **strict
past** (`nb_ts < node_ts` survives). DFS inherits the exact leakage
rule that subgraph sampling itself uses; there is no second,
subtly-different filter to review. `INF` fanout = no sampling = exact
aggregates. Non-temporal types (all timestamps == int64-min) pass
`ts=None` → no filter (line ~196, via `is_temporal`, lines 96–100).

**Reviewer checks:** (a) `ts` is the *source row's own* timestamp
(line ~195: `ts_all[idx]`), so every node's features are "as of
itself"; (b) column selection excludes `Griffin_text_` and non-float
columns (`float_cols`, lines 80–94) — means over category codes or
text-dedup indices would be meaningless.

**Proof:** T4 in [test_dfs_vanilla.py](../../test_dfs_vanilla.py)
recomputes a stored d1 mean by hand from raw adjacency and matches the
denormalized artifact.

### 1.3 Depth 2 — `compute_depth2` (lines 221–280)

d2 of A via relation r to B = mean over A's strict-past B-neighbors of
**B's precomputed d1 features**, excluding B-features derived from the
reverse of r:

- **No-backtrack rule**, lines 240–241: `core_rel(br) == core_rel(r)
  → skip`. `core_rel` (line 103) strips the `head of `/`tail of ` role
  prefix, so a relation and its reverse share a core string. This
  blocks the A→B→A path through which a row's own (label-bearing)
  column could leak back into itself.
- Neighbor sets reuse the same `getedge` strict-past call (line 259).
- Cap: `--max_d2_feats` kept by variance (lines ~274–279).

**Temporal semantics note (deliberate deviation from featuretools):**
B's d1 was computed at *B's own* timestamp; since inclusion requires
`ts_B < ts_A`, the nested window is a **subset** of what featuretools
(single target-cutoff recomputation) would use — strictly more
conservative. Documented in
[DFS_GRIFFIN_DESIGN.md §4b](DFS_GRIFFIN_DESIGN.md).

### 1.4 Normalization + persistence — `znorm` + `main` (lines 281–, 318–)

Z-normalize per feature over all nodes (floatenc operates on ~N(0,1)),
save `dfs/<nt>/d1.pt`, `dfs/<nt>/d2.pt`, `metadfs.yaml`
(names + mean/std per feature — enough to denormalize for audits),
`dfsfeatnameemb.pt`. **Merge-not-clobber:** partial `--nodetypes`
runs load and update existing meta/nameemb (lines ~330–346) instead of
overwriting earlier runs.

### 1.5 Column-NAME embeddings — `build_nameemb` / `primitive_tag` / `compose_nameemb` ([dataconverterdfs.py:63–76, 78–85, 295–](../../dataconverterdfs.py#L63-L85))

Every DFS column gets a name embedding, because Griffin's column
attention scores columns against the task prompt *via their name
embeddings* — a DFS column without one would be semantically
unaddressable. Two generation modes:

- **Compositional (default, CPU-only).** The embedding is a normalized
  sum of embeddings the dataset already has:
  - `dfs1__count__<rel>`         → `norm( edgenameemb[rel] + tag("count") )`
  - `dfs1__<prim>__<rel>__<col>` → `norm( edgenameemb[rel] + featnameemb[col] + tag(prim) )`
  - `dfs2__mean__<rel>__<d1>`    → `norm( edgenameemb[rel] + dfs1_emb[d1] + tag("dfs2_mean") )`
  Because the parts are Nomic vectors, compositions land in the same
  512-d semantic space as native column names; the feature's meaning
  literally *is* (relation ⊕ source column ⊕ primitive). The two-stage
  build in `main` (d1 embeddings first, then d2 which reference them)
  is at lines ~412–422.
- **`--nomic`.** True Nomic encoding of the human-readable name
  (`"dfs1 mean <relation> <column>"`), same model and `"clustering: "`
  prompt as `dataconverterpost.py`. More faithful; needs
  sentence-transformers.

**`primitive_tag` determinism (bug found in review, fixed in
`ba7a993`).** The tag — a fixed pseudo-random unit vector that keeps
mean/max/count of the same (relation, column) from colliding — was
originally seeded with Python's built-in `hash(name)`, which is
**salted per process** (PYTHONHASHSEED) since Python 3.3. Each
converter run was internally consistent but run-to-run different:
regenerating artifacts on another machine would silently unbind
existing checkpoints from their DFS name embeddings. Now seeded by an
md5 content hash (lines 71–73), verified identical across separate
processes. **Reviewer check:** any artifacts generated before
`ba7a993` should be regenerated before being shared or compared
cross-machine.

**Runtime consumption** is §2.2/§2.3: `getdfsfeat` returns the stacked
name embeddings alongside the value embeddings, the append site
extends `nodenameemb` together with the features, and the column
attention treats DFS names identically to native ones (including the
per-layer re-attention of §0b).

---

## 2. Runtime shared machinery

### 2.1 Lazy artifact loading — `Graph.load_dfs` ([hdataset.py:181–207](../../hdataset.py#L181-L207))

Loads meta + tensors + name embeddings once per process. Raises
`FileNotFoundError` with the converter command if artifacts are
missing — enabling a DFS flag on an unconverted dataset fails loudly,
never silently.

### 2.2 Column materialization — `Graph.getdfsfeat` ([hdataset.py:209–233](../../hdataset.py#L209-L233))

For rows `idx` of a node type: slice the z-normed matrix, embed each
column through **the same `floatemb` float path as native numeric
columns** (so DFS values get the pretrained `floatenc` MLP, not a raw
repeat), stack to `(B, C_dfs, D)`, and stack the matching name
embeddings `(C_dfs, D)`.

**Reviewer checks:** depth is *cumulative* (`depth==2` → d1+d2
columns, line 217); a type without artifacts returns `(None, None)`
(line 214-215) so enrichment degrades gracefully per type; column
order is `meta[...]["names"]` order — the same order the matrices were
saved in (converter lines ~160–176 fix the layout before filling).

### 2.3 The append site — `Graph.subgraph` tail ([hdataset.py:346–380](../../hdataset.py#L346-L380))

The **only** mutation of the sampled subgraph:

```python
dfs_root_idx = None
for nodetype in node:                                  # line 347
    if dfs_depth > 0 and nodetype == root_nodetype:
        dfs_root_idx = node[nodetype]                  # RAW indices, captured
    node[nodetype] = ...getfeat(node[nodetype], ...)   # then replaced by features
...
if dfs_depth > 0 and dfs_root_idx is not None:         # line 367
    dfsfeat, dfsnameemb = self.getdfsfeat(root_nodetype, dfs_root_idx, dfs_depth, floatemb)
    if dfsfeat is not None:
        node[root_nodetype]      = concat((feats, dfsfeat), dim=1)      # columns
        nodenameemb[root_nodetype] = concat((names, dfsnameemb), dim=0)
```

**Invariants:**
- `dfs_root_idx` is captured **before** `getfeat` replaces indices
  with feature tensors — the append is row-aligned by construction
  (same index vector feeds both `getfeat` and `getdfsfeat`).
- Only the **root node type**'s entry is enriched; other types in the
  subgraph are untouched. (In the fewshot hop-0 call the root type is
  the leaf type, so "root" enrichment *is* leaf enrichment there.)
- Feature tensor and name-embedding are extended **together** — a
  desync would break column attention silently, hence T2's
  `nne.shape[0] == feat.shape[1]` assertion.
- `dfs_depth=0` (default): `dfs_root_idx` stays `None`, the block is
  dead code → **bit-identical to the pre-DFS pipeline** (T1 proves
  equality under a fixed RNG seed).

---

## 3. Path A — fewshot-fanout leaves (`--dfs_fewshot_depth`)

The complete call chain, outermost first:

| # | Region | What happens |
|---|---|---|
| A1 | [hmaintask_combine.py:79–80](../../hmaintask_combine.py#L79-L80), argparse 401–412 (eval: [hmaintask_combine_llm.py:1418–1419](../../hmaintask_combine_llm.py#L1418-L1419), 2627–2636) | `--dfs_fewshot_depth` enters `subgraphargs` as key `"dfs_fewshot_depth"`; `getattr(..., 0)` default keeps foreign callers safe |
| A2 | [hloaderwrapper.py:217–222](../../hloaderwrapper.py#L217-L222) | `LoaderWrapper.__init__` **copies** subgraphargs and **pops** `dfs_fewshot_depth` into `self.dfs_fewshot_depth` — the pop is what keeps `**subgraphargs` valid for `Graph.subgraph` (which has no such parameter). The copy (`dict(subgraphargs)`) prevents mutating the caller's dict |
| A3 | [hloaderwrapper.py:253–260](../../hloaderwrapper.py#L253-L260) | `fewshotsubgraph` overrides `tmpargs["hop"] = 0` (pre-existing) **and now `tmpargs["dfs_depth"] = self.dfs_fewshot_depth`** — the fewshot-leaf subgraph call gets its own DFS depth, independent of the main subgraph's |
| A4 | [hdataset.py:346–380](../../hdataset.py#L346-L380) | With `hop=0`, the subgraph contains only the leaf rows of the root (=leaf) type; §2.3's append enriches exactly them. `adj` is empty → no edges added |
| A5 | [hloaderwrapper.py:447–452](../../hloaderwrapper.py#L447-L452) (`LoaderWrapperTask`), 539+ (LLM variant) | Unchanged merge: `unifyheterograph` builds the leaf entries (each `(colnameemb, feat)` tuple carries its **own** column count — the leaf entry has `C_native + C_d1` columns while main-graph entries keep theirs), `mergefewshotgraph` attaches leaves via the synthetic `"fewshot"` edge. Heterogeneous per-entry column counts were already supported (different node types have different C) — that pre-existing property is what makes this work with zero changes here |

**What a leaf now carries into the MPNN:** its native columns + its
label (unmasked, as before) + a one-hop aggregate summary of its own
neighborhood — with **no graph expansion** (batch node count is
unchanged vs `dfs_fewshot_depth=0`).

**Verified shapes (joint-v65, rel-f1-drivers):** leaf entry
`2 → 35` columns (2 native + 33 d1); `adj == {}` at hop 0 (T2).

**Leakage question a reviewer must ask:** does a leaf's DFS summary
contain anything the target row shouldn't see? A leaf's d1 was
computed at *the leaf's own timestamp*, and the leaf itself was
sampled from the target's strict past (`Graph.fewshot`,
[hdataset.py:382+](../../hdataset.py#L382)). So the leaf's aggregates
cover a window that ends before the leaf's own time, which precedes
the target's time. Windows are nested; no future information reaches
the target.

## 4. Path B — root nodes (`--dfs_root_depth`)

| # | Region | What happens |
|---|---|---|
| B1 | [hmaintask_combine.py:79](../../hmaintask_combine.py#L79) / [hmaintask_combine_llm.py:1418](../../hmaintask_combine_llm.py#L1418) | `--dfs_root_depth` enters subgraphargs as key `"dfs_depth"` — the name `Graph.subgraph` actually accepts. (Two different dict keys → two different destinations; this asymmetry is the routing mechanism) |
| B2 | [hloaderwrapper.py:262](../../hloaderwrapper.py#L262) (`subgraph`) | Main subgraph call forwards `**self.subgraphargs` unchanged → `dfs_depth = dfs_root_depth` |
| B3 | [hdataset.py:346–380](../../hdataset.py#L346-L380) | §2.3 appends d1+d2 columns to **every node of the root type in the subgraph** — the seed rows (first `len(root_nodeidx)`, per `mapping == arange`, line 343) *and* any same-type rows found at deeper hops. Uniform per-type column count is required by the `(colnameemb, feat)` tuple representation, so this is the only consistent choice |
| B4 | [hloaderwrapper.py:436–439](../../hloaderwrapper.py#L436-L439) (`LoaderWrapperTask`), [505–506](../../hloaderwrapper.py#L505-L506) (LLM variant) | **Mask padding.** `mask[0]` is sized from the (already-widened) feature tensor; `target_feat_mask` (native width, from `Task.get_retrieval/get_regression`, [hdataset.py:375–378](../../hdataset.py#L375-L378)) is padded with `True` (=visible) via `pad_feat_mask` before `mask[0][mapping] = ¬padded`. DFS columns are **never masked**: they aggregate *neighbor* values under strict-past, never the row's own target cell |
| B5 | [hloaderwrapper.py:443–445](../../hloaderwrapper.py#L443-L445), [537–538](../../hloaderwrapper.py#L537-L538) | **Width guard for fewshot similarity.** `fewshotroot → Graph.fewshot` compares masks against *native* `getfeat` output (`rootfeat`, [hdataset.py:397–398](../../hdataset.py#L397-L398)); we slice `mask[0][mapping][:, :native_c]`. Today `prefetch_factor==1` means that code path is dormant, but the slice makes it correct if re-enabled |
| B6 | [hloaderwrapper.py:350](../../hloaderwrapper.py#L350), [385](../../hloaderwrapper.py#L385) (Retrieval/Regression wrappers) | These two loaders **slice** columns by `target_feat_mask` (`node[nt][:, tfm]`); padding with `True` keeps DFS columns through the slice. Not used by the main task path, but now DFS-safe |
| B7 | [hloaderwrapper.py:512](../../hloaderwrapper.py#L512) (LLM variant) | `feature_names` zips `metanode` feat list against the **unpadded** mask (names cover native columns only; `zip` truncates) — LLM prompts are unaffected by DFS columns, by design |

**pad_feat_mask itself** ([hloaderwrapper.py:73–88](../../hloaderwrapper.py#L73-L88)):
pads a `(C,)` bool visibility mask with `True` up to `num_cols`;
returns unchanged when `pad <= 0` — so with DFS off it is the
identity, and every call site degrades to the original code path.

**Verified shapes (joint-v65, rel-f1-driver-position batch):** root
entry `(117, 43, 512)` = 2 native + 33 d1 + 8 d2; `mask[0]` `(117,
43)`; leaf entries 35 columns; vanilla `hmodel.py` forward+backward
runs and produces gradients (T5), SMPNN forward likewise.

**The mask question a reviewer must ask (the sharpest one):** is it
safe that DFS columns are *always visible*, even on the target row of
a task whose native target column is masked? Yes, for two stacked
reasons: (1) d1 aggregates only *neighbor rows'* columns — the row's
own cells never enter its own d1; (2) the route by which a neighbor's
d1 could embed the row's own target value (row → neighbor → row at
depth 2) is exactly what the no-backtrack rule (§1.3) severs, and
temporal types additionally require `ts < ts` (impossible) for
self-inclusion. Residual risk — diamond paths through two *different*
relations on non-temporal types — is shared with every DFS system and
called out in [DFS_GRIFFIN_DESIGN.md §3](DFS_GRIFFIN_DESIGN.md).

---

## 5. Reviewer checklist (region → verify → covering test)

| Region | Verify | Covered by |
|---|---|---|
| `aggregate_batch` NaN handling | mean/max skip NaNs; count counts rows | `test_dfs_parity.py` vs featuretools 1.31.0 |
| `getedge` reuse (conv. 197, 259) | strict `<` cutoff, own-timestamp, INF=no sampling | T4 hand-recompute; same mask as training path by construction |
| No-backtrack (conv. 240–241) | reverse-relation d1 features excluded from d2 | T-check in the earlier integration run (name scan); re-runnable via the assert in that script |
| z-norm + log1p | d1.pt mean≈0/std≈1; counts log1p'd pre-norm | integration check (`d1 mean~0/std~1`) |
| `load_dfs` failure mode | missing artifacts → explicit error naming converter | code read (lines 185–190) |
| `getdfsfeat` alignment | names order == matrix column order; cumulative depth | T2 |
| Name-emb generation (§1.5) | compositional parts resolve (rel/feat/d1 keys exist); two-stage d1→d2 build order | code read (build_nameemb, main ~412–422) |
| `primitive_tag` determinism | identical vectors across separate processes (md5, not salted `hash()`) | 2-process check in `ba7a993` commit; artifacts pre-`ba7a993` must be regenerated |
| Append site row alignment | indices captured pre-`getfeat`; feat+nameemb extended together | T1 (native cols bit-equal), T2 |
| `dfs_depth=0` no-op | bit-identical batches under fixed seed | T1 |
| Fewshot routing (A2/A3) | pop keeps `**subgraphargs` valid; leaf call gets own depth | T6 (all 4 flag combos), T2 (hop-0 shape) |
| Mask padding (B4) | native masked cols still masked; DFS cols never masked | T3; T3b re-runs on virus-wnv-pred (non-empty `masked_feat`) once its artifacts finish converting |
| Similarity-mask slice (B5) | native-width slice passed to `fewshot()` | code read (dormant path; `prefetch_factor==1`) |
| Retrieval/Regression slicing (B6) | padded mask keeps DFS cols through boolean slice | code read; loaders unused in main path |
| Vanilla + SMPNN forward | both backbones consume enriched batches | T5 (vanilla, with backward), earlier SMPNN check |
| Eval contract | same dfs flags at eval as training | enforced by convention; flag help-text + `run_smpnn_dfs_eval.sh` encode it |

Run everything:

```bash
python test_dfs_vanilla.py                 # T1–T6 (+T3b when virus artifacts exist)
python test_dfs_parity.py --phase a --out /tmp/dfs_parity
<ft-venv>/bin/python test_dfs_parity.py --phase b --out /tmp/dfs_parity
```

## 6. Known edge cases & residual risks (keep on the radar)

1. **Checkpoint/flag mismatch at eval** is *not* detected
   automatically — the model happily attends over a different column
   set and produces plausible-but-wrong numbers. Mitigation is
   procedural (scripts hard-code matching flags). A future guard could
   stash the dfs config inside the checkpoint dir.
2. **Non-temporal diamond paths** (§4's residual risk): two distinct
   relations forming a cycle could route a non-temporal row's masked
   column back at depth 2. None of the o1/o2 target types exhibit
   this today (their masked tasks are temporal or the masked column is
   text-typed and thus never aggregated), but a new dataset should be
   checked.
3. **`LoaderWrapperCompletion`** (pretraining) was deliberately left
   without DFS flags — its random column-masking game would otherwise
   sometimes pick DFS columns as prediction targets. If DFS is ever
   wanted there, that interaction needs its own design.
4. **Memory**: +33–41 columns on enriched rows is a modest activation
   increase; at `batchsize 256` no OOM observed, but tight-memory
   configs should watch it (drop to 128 first).
5. **`optuna_griffin_llm.py` / downsample eval** received
   `fanout_decay` but not the DFS keys — enabling DFS there requires
   adding the two subgraphargs lines (mirroring
   [hmaintask_combine.py:79–80](../../hmaintask_combine.py#L79-L80)).
