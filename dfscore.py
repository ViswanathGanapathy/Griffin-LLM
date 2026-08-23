"""Cutoff-aware Deep Feature Synthesis core, shared by the offline converter
(``dataconverterdfs.py``) and the online loader path (``hdataset.py``).

Every aggregate is parameterised by an explicit per-row **cutoff** tau:

    d1_{r,c,prim}(v, tau) = prim{ c(u) : u in N_r(v), ts(u) < tau }
    d2_{r,f}(v, tau)      = mean{ d1_f(u, tau) : u in N_r(v), ts(u) < tau }

``tau = None`` reproduces the historical behaviour (each row evaluated at
its own stored timestamp, no cutoff at all for non-temporal types), which
remains correct only for Completion-style pretraining where the query time
*is* the row's timestamp. Task-driven consumers must pass the seed's task
timestamp; see ``Graph.getdfsfeat``.

Non-temporal neighbours carry ``ts == INT64_MIN`` in the ``___TIMESTAMP``
adjacency companion column, so they pass any cutoff — dimension rows are
always visible. This matches fastdfs' cutoff propagation rule.

Values produced here are **raw** (unnormalised) except that ``count`` is
log1p'd; z-normalisation happens at read time from stats in metadfs.yaml,
so the online and offline paths agree exactly.
"""
import hashlib

import torch

from hdataset import INF, edgename2tail

INT64_MIN = -9223372036854775808

#: Aggregation primitives, in the canonical column order used by layouts.
VALUE_PRIMS = ("mean", "max", "min", "std")
DEFAULT_PRIMS = ("count", "mean", "max")


def sanitize(relname: str) -> str:
    """Short deterministic token for a relation name, for feature naming."""
    s = relname.replace("head of ", "H_").replace("tail of ", "T_")
    return s.replace(":", "__").replace(" ", "_")


def core_rel(relname: str) -> str:
    """The role-free relation core, shared by a relation and its reverse."""
    return relname.replace("head of ", "").replace("tail of ", "")


def primitive_tag(name: str, dim: int) -> torch.Tensor:
    """Deterministic pseudo-random unit vector for an aggregation primitive.

    Seeded by a content hash (md5), NOT Python's built-in hash() — the
    latter is salted per process (PYTHONHASHSEED), which would make name
    embeddings differ across converter runs and silently unbind
    checkpoints from regenerated artifacts.
    """
    seed = int.from_bytes(hashlib.md5(name.encode()).digest()[:4], "little")
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def compose_nameemb(parts: list, dim: int) -> torch.Tensor:
    """Compositional name embedding: normalized sum of part embeddings."""
    v = torch.zeros(dim)
    for p in parts:
        v = v + p
    n = v.norm()
    return v / n if n > 0 else v


def is_timestamp_col(colname: str) -> bool:
    """Is this a datetime-derived column?

    Such columns hold epoch magnitudes (~1e18 ns), so their `max` is a
    near-deterministic function of the cutoff itself and they dominate any
    raw-variance ranking. `mean` is kept (recency is a real signal); `max`
    and depth-2 propagation are dropped.
    """
    return "TIMESTAMP(" in colname or colname.endswith("__timestamp")


def is_temporal(graph, nodetype: str) -> bool:
    """Does this node type carry real timestamps?

    Memoised on the graph: the answer needs a full timestamp column scan,
    and compute_d2_at asks per relation per chunk.
    """
    cache = getattr(graph, "_dfs_temporal", None)
    if cache is None:
        cache = graph._dfs_temporal = {}
    if nodetype not in cache:
        ts = graph.nodes[nodetype].feat["timestamp"]
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts)
        cache[nodetype] = bool((ts != INT64_MIN).any())
    return cache[nodetype]


def float_cols(graph, nodetype: str, max_cols: int) -> list:
    """Float-typed (non-text) columns of a node type, capped at max_cols."""
    node = graph.nodes[nodetype]
    cols = []
    for c in node.featlist:
        if "Griffin_text_" in c:
            continue
        sample = node.feat[0][c]
        if torch.is_tensor(sample) and torch.is_floating_point(sample):
            cols.append(c)
        elif isinstance(sample, float):
            cols.append(c)
        if len(cols) >= max_cols:
            break
    return cols


def process_in_chunks(num_rows: int, fn, min_rows: int = 1024):
    """Run ``fn(a, b)`` over positional row ranges, halving on CPU OOM.

    getedge pads every row's neighbor list to the batch max
    (pad_sequence), so a single high-degree hub row (e.g. a prolific
    stackexchange user with 50K+ votes) can make a large batch demand tens
    of GB. Splitting the chunk shrinks rows-x-maxlen until it fits;
    per-row results are independent, so splitting is exact.
    """
    stack = [(0, num_rows)]
    while stack:
        a, b = stack.pop()
        try:
            fn(a, b)
        except RuntimeError as e:
            if "allocate" in str(e).lower() and (b - a) > min_rows:
                m = (a + b) // 2
                print(f"    [oom-split] rows {a}..{b} -> two halves", flush=True)
                stack += [(m, b), (a, m)]
            else:
                raise


def aggregate_batch(src, tar, values_by_feat, num_src, prims=DEFAULT_PRIMS):
    """Scatter aggregates for one (relation, batch).

    src: (E,) local source indices; tar: (E,) global neighbor indices.
    values_by_feat: {featname: (E,) float tensor} aligned to ``tar``.
    Returns dict {suffix: (num_src,) tensor} with 'count' and '<prim>__<c>'.

    NaN semantics match featuretools / SQL aggregates (verified by
    test_dfs_parity.py): NaN values are SKIPPED — mean is
    sum(non-NaN)/count(non-NaN), max is over non-NaN values only. 'count'
    counts neighbor ROWS (featuretools Count), not non-NaN values. Empty
    aggregates are 0, matching the pre-normalization fill used throughout.
    """
    out = {}
    ones = torch.ones_like(src, dtype=torch.float32)
    out["count"] = torch.zeros(num_src).scatter_add_(0, src, ones)
    for c, v in values_by_feat.items():
        v = v.to(torch.float32)
        valid = torch.isfinite(v)
        v_zero = torch.where(valid, v, torch.zeros_like(v))
        n_valid = torch.zeros(num_src).scatter_add_(
            0, src, valid.to(torch.float32)
        )
        s = torch.zeros(num_src).scatter_add_(0, src, v_zero)
        if "mean" in prims or "std" in prims:
            mean = s / n_valid.clamp_min(1.0)
            if "mean" in prims:
                out[f"mean__{c}"] = mean
        if "std" in prims:
            # ddof=1, matching pandas/featuretools; <2 valid values -> 0
            sq = torch.zeros(num_src).scatter_add_(0, src, v_zero * v_zero)
            var = (sq - n_valid * mean * mean) / (n_valid - 1).clamp_min(1.0)
            out[f"std__{c}"] = torch.where(
                n_valid >= 2, var.clamp_min(0.0).sqrt(), torch.zeros_like(var)
            )
        if "max" in prims:
            m = torch.full((num_src,), float("-inf")).scatter_reduce_(
                0, src, torch.where(valid, v, torch.full_like(v, float("-inf"))),
                reduce="amax", include_self=True,
            )
            m[torch.isinf(m)] = 0.0
            out[f"max__{c}"] = m
        if "min" in prims:
            m = torch.full((num_src,), float("inf")).scatter_reduce_(
                0, src, torch.where(valid, v, torch.full_like(v, float("inf"))),
                reduce="amin", include_self=True,
            )
            m[torch.isinf(m)] = 0.0
            out[f"min__{c}"] = m
    return out


# ─────────────────────────── layout / plan ───────────────────────────

def build_d1_layout(graph, nodetype: str, max_cols_per_rel: int = 8,
                    prims=DEFAULT_PRIMS, include_target_rels: bool = False):
    """Fixed column layout for a node type's depth-1 features.

    Deterministic given (graph, nodetype, args) so that the online path
    reproduces the converter's column order exactly.

    include_target_rels: when False (default), relations whose tail is an
    ``is_target`` node type are dropped entirely. Their columns hold task
    labels, so aggregating them is only defensible under an explicit
    cutoff ("mean past label among my neighbours").
    """
    rels = graph.metanode[nodetype]["in"] + graph.metanode[nodetype]["out"]
    dropped = []
    kept = []
    for r in rels:
        if not include_target_rels and graph.nodes[edgename2tail(r)].is_target:
            dropped.append(r)
        else:
            kept.append(r)

    relcols = {r: float_cols(graph, edgename2tail(r), max_cols_per_rel)
               for r in kept}
    names, parts, rel_of_feat = [], [], []
    for r in kept:
        tag = sanitize(r)
        if "count" in prims:
            names.append(f"dfs1__count__{tag}")
            parts.append([("rel", r), ("prim", "count")])
            rel_of_feat.append(r)
        for c in relcols[r]:
            for prim in VALUE_PRIMS:
                if prim not in prims:
                    continue
                # epoch-magnitude columns: keep mean (recency), drop the rest
                if is_timestamp_col(c) and prim != "mean":
                    continue
                names.append(f"dfs1__{prim}__{tag}__{c}")
                parts.append([("rel", r), ("feat", c), ("prim", prim)])
                rel_of_feat.append(r)
    return {
        "rels": kept, "dropped_target_rels": dropped, "relcols": relcols,
        "names": names, "parts": parts, "rel_of_feat": rel_of_feat,
        "prims": list(prims), "max_cols_per_rel": max_cols_per_rel,
        "include_target_rels": include_target_rels,
        "col_index": {n: i for i, n in enumerate(names)},
    }


def build_d2_plan(graph, nodetype: str, layouts: dict):
    """Legal (relation, tail-d1-column) pairs for depth 2.

    Excludes the traversed relation's reverse (no-backtrack rule) and
    timestamp-derived source columns (C9).
    """
    rels = layouts[nodetype]["rels"]
    plan = []  # (rel, j_in_tail_layout, d2_name, tail_d1_name)
    for r in rels:
        tail = edgename2tail(r)
        if tail not in layouts:
            continue
        tl = layouts[tail]
        for j, (bn, br) in enumerate(zip(tl["names"], tl["rel_of_feat"])):
            if core_rel(br) == core_rel(r):
                continue  # no immediate backtracking
            if "TIMESTAMP(" in bn or bn.endswith("__timestamp"):
                continue
            plan.append((r, j, f"dfs2__mean__{sanitize(r)}__{bn}", bn))
    return plan


def plan_names(plan):
    return [n for _, _, n, _ in plan]


def plan_parts(plan):
    return [[("rel", r), ("dfs1", bn), ("prim", "dfs2_mean")]
            for r, _, _, bn in plan]


# ─────────────────────────── column sources ───────────────────────────

class FullColumnCache:
    """Materialises whole feature columns. Fast for the converter, which
    sweeps every row of a type anyway."""

    def __init__(self, graph):
        self.graph = graph
        self._cache = {}

    def _col(self, tail, c):
        if (tail, c) not in self._cache:
            col = self.graph.nodes[tail].feat[c]
            if not torch.is_tensor(col):
                col = torch.tensor(col)
            self._cache[(tail, c)] = col
        return self._cache[(tail, c)]

    def __call__(self, tail, cols, tar):
        return {c: self._col(tail, c)[tar] for c in cols}


class RowGather:
    """Reads only the rows an edge batch touches. Bounded memory, so this
    is what the training loader uses — a full column of amazon-Review is
    13.7M values per feature per worker."""

    def __init__(self, graph):
        self.graph = graph

    def __call__(self, tail, cols, tar):
        if len(tar) == 0:
            return {c: torch.zeros(0) for c in cols}
        uniq, inv = torch.unique(tar, return_inverse=True)
        data = self.graph.nodes[tail].feat[uniq]
        out = {}
        for c in cols:
            v = data[c]
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            out[c] = v[inv]
        return out


# ─────────────────────────── computation ───────────────────────────

def compute_d1_at(graph, nodetype: str, idx: torch.Tensor, cutoff, layout,
                  colsource, batch_rows: int = 65536, progress: str = None):
    """Depth-1 aggregates for rows ``idx`` of ``nodetype``, each at its own
    cutoff.

    idx:    (B,) int64 global node indices (duplicates allowed).
    cutoff: (B,) int64 per-row cutoff, or None for "own timestamp"
            semantics (temporal types) / no cutoff at all (non-temporal).
    Returns (B, C1) float32 **raw** values, counts already log1p'd.
    """
    node = graph.nodes[nodetype]
    names, relcols, col_index = layout["names"], layout["relcols"], layout["col_index"]
    prims = tuple(layout["prims"])
    out = torch.zeros((len(idx), len(names)), dtype=torch.float32)
    if not names or len(idx) == 0:
        return out

    def _chunk(a, b):
        i = idx[a:b]
        cut_chunk = None if cutoff is None else cutoff[a:b]
        ttadj = node.getedge(i, INF, cut_chunk)
        for r, (src, tar) in ttadj.items():
            if r not in relcols:
                continue
            tail = edgename2tail(r)
            aggs = aggregate_batch(
                src, tar, colsource(tail, relcols[r], tar), b - a, prims
            )
            tag = sanitize(r)
            k = col_index.get(f"dfs1__count__{tag}")
            if k is not None:
                out[a:b, k] = aggs["count"]
            for c in relcols[r]:
                for prim in VALUE_PRIMS:
                    k = col_index.get(f"dfs1__{prim}__{tag}__{c}")
                    if k is not None:
                        out[a:b, k] = aggs[f"{prim}__{c}"]

    for start in range(0, len(idx), batch_rows):
        stop = min(start + batch_rows, len(idx))
        process_in_chunks(stop - start, lambda a, b, o=start: _chunk(o + a, o + b))
        if progress and (start // batch_rows) % 10 == 0:
            print(f"    [{progress}] d1 rows {start}..{stop}", flush=True)

    for i, n in enumerate(names):
        if "__count__" in n:
            out[:, i] = torch.log1p(out[:, i])
    return out


def compute_d2_at(graph, nodetype: str, idx: torch.Tensor, cutoff, plan,
                  layouts, d1_raw_of, colsource, batch_rows: int = 65536,
                  nontemporal_hubs: str = "recompute", hub_budget: int = 2_000_000):
    """Depth-2: mean over each row's strictly-past neighbours of the
    neighbour's depth-1 features.

    Hub semantics (the correctness-critical part):

    * **temporal hub** — reuse the hub's precomputed *own-timestamp* d1.
      Since ``ts(hub) < tau``, the hub's window is a subset of the correct
      cutoff window: conservative, never leaky, and free to look up. This
      is documented deviation #2 in DFS_GRIFFIN_DESIGN.md §4b.
    * **non-temporal hub** — the hub has no timestamp of its own, so its
      own-ts d1 spans *all* history and would leak the future straight
      through a dimension table. It is recomputed at the querying row's
      cutoff.

    ``nontemporal_hubs="skip"`` zero-fills instead of recomputing (cheap
    degradation for oversized hubs). It never falls back to the all-time
    value, which would be silently leaky.
    """
    node = graph.nodes[nodetype]
    out = torch.zeros((len(idx), len(plan)), dtype=torch.float32)
    if not plan or len(idx) == 0:
        return out

    by_rel = {}
    for k, (r, j, _, _) in enumerate(plan):
        by_rel.setdefault(r, []).append((k, j))

    def _chunk(a, b):
        i = idx[a:b]
        c = None if cutoff is None else cutoff[a:b]
        ttadj = node.getedge(i, INF, c)
        for r, (src, tar) in ttadj.items():
            if r not in by_rel:
                continue
            tail = edgename2tail(r)
            ones = torch.ones_like(src, dtype=torch.float32)
            count = torch.zeros(b - a).scatter_add_(0, src, ones)
            safe = count.clamp_min(1.0)

            if cutoff is None or is_temporal(graph, tail):
                hub = d1_raw_of(tail)               # (num_tail, C1) raw
                if hub is None:
                    continue
                for k, j in by_rel[r]:
                    s = torch.zeros(b - a).scatter_add_(0, src, hub[tar, j])
                    out[a:b, k] = s / safe
                continue

            # non-temporal hub: recompute its d1 at the querying row's cutoff
            if nontemporal_hubs == "skip":
                continue
            hub_cut = c[src]
            pairs, inv = torch.unique(
                torch.stack((tar, hub_cut), dim=1), dim=0, return_inverse=True
            )
            if len(pairs) > hub_budget:
                print(f"    [d2] {nodetype} via {r}: {len(pairs)} unique "
                      f"(hub, cutoff) pairs exceeds budget {hub_budget}; "
                      f"zero-filling (use --d2_nontemporal_hubs skip to "
                      f"silence)", flush=True)
                continue
            hub = compute_d1_at(graph, tail, pairs[:, 0], pairs[:, 1],
                                layouts[tail], colsource, batch_rows)
            for k, j in by_rel[r]:
                s = torch.zeros(b - a).scatter_add_(0, src, hub[inv, j])
                out[a:b, k] = s / safe

    for start in range(0, len(idx), batch_rows):
        stop = min(start + batch_rows, len(idx))
        process_in_chunks(stop - start, lambda a, b, o=start: _chunk(o + a, o + b))
    return out


# ─────────────────────── checkpoint/flag binding ───────────────────────

def write_dfs_config(savepath, args):
    """Record the DFS input contract next to the checkpoints (C10).

    The DFS column set is part of the encoder's input contract; evaluating
    a checkpoint under different flags produces plausible-but-wrong
    numbers with no error. This file makes the mismatch loud.
    """
    import json
    import os
    import os.path as osp
    if savepath is None:
        return
    cfg = {
        "dfs_root_depth": getattr(args, "dfs_root_depth", 0),
        "dfs_fewshot_depth": getattr(args, "dfs_fewshot_depth", 0),
        "dfs_format_version": 2,
    }
    os.makedirs(savepath, exist_ok=True)
    with open(osp.join(savepath, "dfs_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def check_dfs_config(loadpath, args, override=False):
    """Compare the checkpoint's recorded DFS contract with current flags.

    loadpath usually points at .../best_checkpoint; the config lives in
    its parent. A checkpoint without a config (pre-guard training runs)
    only warns.
    """
    import json
    import os.path as osp
    if loadpath is None:
        return
    for d in (loadpath, osp.dirname(loadpath.rstrip("/"))):
        p = osp.join(d, "dfs_config.json")
        if osp.exists(p):
            with open(p) as f:
                cfg = json.load(f)
            break
    else:
        print("[dfs-guard] no dfs_config.json next to the checkpoint "
              "(trained before the guard existed) — verify the dfs flags "
              "match training manually", flush=True)
        return
    mismatches = {
        k: (cfg.get(k, 0), getattr(args, k, 0))
        for k in ("dfs_root_depth", "dfs_fewshot_depth")
        if cfg.get(k, 0) != getattr(args, k, 0)
    }
    if mismatches:
        msg = (f"DFS flag mismatch vs checkpoint at {loadpath}: "
               + ", ".join(f"{k}: trained={a} now={b}"
                           for k, (a, b) in mismatches.items()))
        if override:
            print(f"[dfs-guard] {msg} — proceeding due to --dfs_override",
                  flush=True)
        else:
            raise RuntimeError(
                msg + ". The DFS column set is part of the encoder's input "
                "contract; a mismatch silently produces wrong numbers. "
                "Match the flags, or pass --dfs_override to force."
            )


# ─────────────────────────── normalisation ───────────────────────────

def znorm_stats(feat: torch.Tensor):
    mean = feat.mean(dim=0)
    std = feat.std(dim=0).clamp_min(1e-6)
    return mean, std


def apply_norm(feat: torch.Tensor, mean, std):
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=torch.float32)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=torch.float32)
    return (feat - mean) / std


def nondegeneracy_score(feat: torch.Tensor, sample: int = 100_000):
    """Per-column fraction of distinct values — a scale-free stand-in for
    "how much does this column actually say".

    Ranking the depth-2 cap by *raw* variance lets a single epoch-magnitude
    column (~1e18) outrank every genuine feature, and ranking by variance
    after z-normalisation is vacuous (every non-degenerate column is
    exactly 1). Distinct-value fraction collapses constant and
    near-constant columns to ~0 while staying invariant to scale.
    """
    if feat.shape[0] > sample:
        feat = feat[torch.randperm(feat.shape[0])[:sample]]
    n = max(1, feat.shape[0])
    return torch.tensor(
        [len(torch.unique(feat[:, j])) / n for j in range(feat.shape[1])],
        dtype=torch.float32,
    )
