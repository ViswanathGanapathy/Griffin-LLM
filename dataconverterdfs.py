"""Offline Deep Feature Synthesis (DFS) over a processed Griffin dataset.

Computes per-node DFS aggregate features in the style of
RDBLearn (arXiv 2602.18495) / fastdfs (github.com/HKUSHXLab/fastdfs),
but natively over Griffin's processed structures (CSR-ish Arrow
adjacency + node feature tables) instead of raw DataFrames — Griffin's
``Node.getedge`` already provides the timestamp-filtered full
neighborhood, so one-hop DFS is a fixed-weight unsampled aggregation
pass over it.

Depth 1 (per node n of type A, per relation r to neighbor type B):
    dfs1__count__{r}            number of strictly-past neighbors
    dfs1__mean__{r}__{c}        mean of float column c over those neighbors
    dfs1__max__{r}__{c}         max  of float column c over those neighbors

Depth 2 (aggregates of neighbors' depth-1 aggregates):
    dfs2__mean__{r}__{d1name}   mean of B's d1 feature over the same
                                strictly-past neighbor set, EXCLUDING
                                B-features derived from the reverse of r
                                (no-backtrack rule)

Leakage rules:
  * Temporal node types (any timestamp != int64-min): a neighbor row is
    included iff nb_ts < node_ts (strict) — reuses getedge's mask.
  * Non-temporal types: no cutoff (all neighbors).
  * Depth 2 excludes the traversed relation's reverse (no A->B->A).

All features are z-normalized over the node population (Griffin's
floatenc expects ~N(0,1) inputs). Empty aggregates are 0 (mean/max)
before normalization; counts are log1p'd before normalization.

Outputs under <dstpath>/dfs/:
    <nodetype>/d1.pt        float32 (num_nodes, C1)
    <nodetype>/d2.pt        float32 (num_nodes, C2)
    metadfs.yaml            feature names + normalization stats
    dfsfeatnameemb.pt       {feature_name: (D,) float32 name embedding}

Usage:
    python dataconverterdfs.py datasets/joint-v65 \
        [--nodetypes rel-f1-drivers rel-f1-results] \
        [--batch_rows 65536] [--max_cols_per_rel 8] [--max_d2_feats 32] \
        [--skip_d2] [--nomic] [--emb_dim 512]
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch
import yaml

from hdataset import Graph, INF, edgename2tail

INT64_MIN = -9223372036854775808


def sanitize(relname: str) -> str:
    """Short deterministic token for a relation name, for feature naming."""
    s = relname.replace("head of ", "H_").replace("tail of ", "T_")
    return s.replace(":", "__").replace(" ", "_")


def primitive_tag(name: str, dim: int) -> torch.Tensor:
    """Deterministic pseudo-random unit vector for an aggregation primitive."""
    g = torch.Generator().manual_seed(abs(hash(name)) % (2**31))
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def compose_nameemb(parts: list, dim: int) -> torch.Tensor:
    """Compositional name embedding: normalized sum of part embeddings."""
    v = torch.zeros(dim)
    for p in parts:
        v = v + p
    n = v.norm()
    return v / n if n > 0 else v


def float_cols(graph: Graph, nodetype: str, max_cols: int) -> list:
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


def is_temporal(graph: Graph, nodetype: str) -> bool:
    ts = graph.nodes[nodetype].feat["timestamp"]
    if not torch.is_tensor(ts):
        ts = torch.tensor(ts)
    return bool((ts != INT64_MIN).any())


def core_rel(relname: str) -> str:
    """The role-free relation core, shared by a relation and its reverse."""
    return relname.replace("head of ", "").replace("tail of ", "")


def aggregate_batch(src, tar, values_by_feat, num_src):
    """Scatter count/mean/max for one (relation, batch).

    src: (E,) local source indices; tar: (E,) global neighbor indices.
    values_by_feat: {featname: (num_B,) float tensor} — full columns.
    Returns dict {suffix: (num_src,) tensor} with 'count', 'mean__c', 'max__c'.

    NaN semantics match featuretools / SQL aggregates (verified by
    test_dfs_parity.py): NaN values are SKIPPED — mean is
    sum(non-NaN)/count(non-NaN), max is over non-NaN values only.
    'count' counts neighbor ROWS (featuretools Count), not non-NaN
    values. Empty aggregates (no valid neighbor values) are 0, matching
    the pre-normalization fill used throughout.
    """
    out = {}
    ones = torch.ones_like(src, dtype=torch.float32)
    count = torch.zeros(num_src).scatter_add_(0, src, ones)
    out["count"] = count
    for c, colvals in values_by_feat.items():
        v = colvals[tar].to(torch.float32)
        valid = torch.isfinite(v)
        v_zero = torch.where(valid, v, torch.zeros_like(v))
        n_valid = torch.zeros(num_src).scatter_add_(
            0, src, valid.to(torch.float32)
        )
        s = torch.zeros(num_src).scatter_add_(0, src, v_zero)
        out[f"mean__{c}"] = s / n_valid.clamp_min(1.0)
        v_neginf = torch.where(valid, v, torch.full_like(v, float("-inf")))
        m = torch.full((num_src,), float("-inf")).scatter_reduce_(
            0, src, v_neginf, reduce="amax", include_self=True
        )
        m[torch.isinf(m)] = 0.0
        out[f"max__{c}"] = m
    return out


def compute_depth1(graph: Graph, nodetype: str, args):
    """Depth-1 DFS features for all rows of `nodetype`.

    Returns (feat_matrix (N, C1) float32 unnormalized, names, name_parts)
    where name_parts[i] lists the (kind, key) pairs used for embedding
    composition.
    """
    node = graph.nodes[nodetype]
    num = len(node)
    temporal = is_temporal(graph, nodetype)
    ts_all = node.feat["timestamp"]
    if not torch.is_tensor(ts_all):
        ts_all = torch.tensor(ts_all)

    # Discover this type's relations + neighbor value columns from meta
    rels = graph.metanode[nodetype]["in"] + graph.metanode[nodetype]["out"]
    relcols = {}
    for r in rels:
        tail = edgename2tail(r)
        relcols[r] = float_cols(graph, tail, args.max_cols_per_rel)

    # Feature layout (fixed order)
    names, parts, rel_of_feat = [], [], []
    for r in rels:
        tag = sanitize(r)
        names.append(f"dfs1__count__{tag}")
        parts.append([("rel", r), ("prim", "count")])
        rel_of_feat.append(r)
        for c in relcols[r]:
            for prim in ("mean", "max"):
                names.append(f"dfs1__{prim}__{tag}__{c}")
                parts.append([("rel", r), ("feat", c), ("prim", prim)])
                rel_of_feat.append(r)
    if not names:
        return None, [], [], []

    # Preload neighbor value columns once per tail type
    colcache = {}
    for r in rels:
        tail = edgename2tail(r)
        for c in relcols[r]:
            if (tail, c) not in colcache:
                col = graph.nodes[tail].feat[c]
                if not torch.is_tensor(col):
                    col = torch.tensor(col)
                colcache[(tail, c)] = col

    feat = torch.zeros((num, len(names)), dtype=torch.float32)
    col_index = {n: i for i, n in enumerate(names)}

    for start in range(0, num, args.batch_rows):
        idx = torch.arange(start, min(start + args.batch_rows, num))
        ts = ts_all[idx] if temporal else None
        ttadj = node.getedge(idx, INF, ts)
        for r, (src, tar) in ttadj.items():
            if r not in relcols:
                continue
            tail = edgename2tail(r)
            values = {c: colcache[(tail, c)] for c in relcols[r]}
            aggs = aggregate_batch(src, tar, values, len(idx))
            tag = sanitize(r)
            feat[idx, col_index[f"dfs1__count__{tag}"]] = aggs["count"]
            for c in relcols[r]:
                for prim in ("mean", "max"):
                    feat[idx, col_index[f"dfs1__{prim}__{tag}__{c}"]] = (
                        aggs[f"{prim}__{c}"]
                    )
        if start == 0 or (start // args.batch_rows) % 10 == 0:
            print(f"    [{nodetype}] d1 rows {start}..{idx[-1].item()}")

    # log1p counts (heavy-tailed) before normalization
    for i, n in enumerate(names):
        if "__count__" in n:
            feat[:, i] = torch.log1p(feat[:, i])
    return feat, names, parts, rel_of_feat


def compute_depth2(graph: Graph, nodetype: str, d1_store, args):
    """Depth-2: mean over strictly-past neighbors of their d1 features,
    excluding neighbor features derived from the reverse relation."""
    node = graph.nodes[nodetype]
    num = len(node)
    temporal = is_temporal(graph, nodetype)
    ts_all = node.feat["timestamp"]
    if not torch.is_tensor(ts_all):
        ts_all = torch.tensor(ts_all)

    rels = graph.metanode[nodetype]["in"] + graph.metanode[nodetype]["out"]
    # Which (rel, neighbor-d1-feature) pairs are legal (no-backtrack)?
    plan = []  # (rel, d1_col_idx_in_B, d1_name)
    for r in rels:
        tail = edgename2tail(r)
        if tail not in d1_store:
            continue
        b_names, b_rels = d1_store[tail]["names"], d1_store[tail]["rel_of_feat"]
        for j, (bn, br) in enumerate(zip(b_names, b_rels)):
            if core_rel(br) == core_rel(r):
                continue  # no immediate backtracking
            plan.append((r, j, bn))
    if not plan:
        return None, [], []

    names = [f"dfs2__mean__{sanitize(r)}__{bn}" for r, _, bn in plan]
    parts = [[("rel", r), ("dfs1", bn), ("prim", "dfs2_mean")]
             for r, _, bn in plan]
    feat = torch.zeros((num, len(plan)), dtype=torch.float32)

    # Group plan entries by relation for one scatter pass per relation
    by_rel = {}
    for k, (r, j, _) in enumerate(plan):
        by_rel.setdefault(r, []).append((k, j))

    for start in range(0, num, args.batch_rows):
        idx = torch.arange(start, min(start + args.batch_rows, num))
        ts = ts_all[idx] if temporal else None
        ttadj = node.getedge(idx, INF, ts)
        for r, (src, tar) in ttadj.items():
            if r not in by_rel:
                continue
            tail = edgename2tail(r)
            b_feat = d1_store[tail]["feat"]  # (num_B, C1_B) unnormalized
            ones = torch.ones_like(src, dtype=torch.float32)
            count = torch.zeros(len(idx)).scatter_add_(0, src, ones)
            safe = count.clamp_min(1.0)
            for k, j in by_rel[r]:
                v = b_feat[tar, j]
                s = torch.zeros(len(idx)).scatter_add_(0, src, v)
                feat[idx, k] = s / safe
    # Cap total d2 features by variance
    if feat.shape[1] > args.max_d2_feats:
        keep = torch.topk(feat.var(dim=0), args.max_d2_feats).indices.sort().values
        feat = feat[:, keep]
        names = [names[i] for i in keep.tolist()]
        parts = [parts[i] for i in keep.tolist()]
    return feat, names, parts


def znorm(feat: torch.Tensor):
    mean = feat.mean(dim=0)
    std = feat.std(dim=0).clamp_min(1e-6)
    return (feat - mean) / std, mean, std


def build_nameemb(all_parts, graph: Graph, dfs1_embs, dim, use_nomic, readable):
    """Name embeddings — compositional by default, Nomic if requested."""
    if use_nomic:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", device="cuda:0",
            cache_folder="cache_data/model", trust_remote_code=True,
            truncate_dim=dim,
        )
        out = {}
        for name, text in readable.items():
            e = model.encode(text, prompt="clustering: ", convert_to_numpy=True)
            e = e / np.linalg.norm(e)
            out[name] = torch.from_numpy(e).to(torch.float32)
        return out
    out = {}
    for name, parts in all_parts.items():
        vecs = []
        for kind, key in parts:
            if kind == "rel":
                vecs.append(graph.edgenameemb[key])
            elif kind == "feat":
                vecs.append(graph.featnameemb[key])
            elif kind == "dfs1":
                vecs.append(dfs1_embs[key])
            else:  # primitive
                vecs.append(primitive_tag(key, dim))
        out[name] = compose_nameemb(vecs, dim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dstpath", type=str, help="processed Griffin dataset dir")
    ap.add_argument("--nodetypes", nargs="*", default=None,
                    help="restrict to these node types (default: all non-target "
                         "types with adjacency)")
    ap.add_argument("--batch_rows", type=int, default=65536)
    ap.add_argument("--max_cols_per_rel", type=int, default=8)
    ap.add_argument("--max_d2_feats", type=int, default=32)
    ap.add_argument("--skip_d2", action="store_true")
    ap.add_argument("--nomic", action="store_true",
                    help="use Nomic to embed feature names (needs GPU + "
                         "sentence_transformers); default is compositional")
    ap.add_argument("--emb_dim", type=int, default=512)
    args = ap.parse_args()

    graph = Graph(args.dstpath)
    dfsdir = osp.join(args.dstpath, "dfs")
    os.makedirs(dfsdir, exist_ok=True)

    types = args.nodetypes or [
        nt for nt in graph.metanode
        if not graph.nodes[nt].is_target and graph.nodes[nt].adj is not None
    ]
    print(f"DFS over {len(types)} node types: {types}")

    # Merge-with-existing so partial --nodetypes runs accumulate instead
    # of clobbering earlier runs' metadata/name-embeddings.
    meta, all_parts, readable = {}, {}, {}
    if osp.exists(osp.join(dfsdir, "metadfs.yaml")):
        with open(osp.join(dfsdir, "metadfs.yaml")) as f:
            meta = yaml.safe_load(f) or {}
        for nt in types:
            meta.pop(nt, None)   # recomputed below
    existing_nameemb = {}
    if osp.exists(osp.join(dfsdir, "dfsfeatnameemb.pt")):
        existing_nameemb = torch.load(
            osp.join(dfsdir, "dfsfeatnameemb.pt"),
            map_location="cpu", weights_only=True,
        )
    d1_store = {}

    # ── Pass 1: depth 1 for every type ──
    for nt in types:
        print(f"[d1] {nt} ({graph.metanode[nt]['num']} rows)")
        feat, names, parts, rel_of_feat = compute_depth1(graph, nt, args)
        if feat is None:
            print(f"    no aggregatable relations; skipping")
            continue
        d1_store[nt] = {"feat": feat, "names": names, "rel_of_feat": rel_of_feat}
        for n, p in zip(names, parts):
            all_parts[n] = p
            readable[n] = n.replace("__", " ")

    # ── Pass 2: depth 2 (uses unnormalized d1 of neighbors) ──
    d2_store = {}
    if not args.skip_d2:
        for nt in types:
            print(f"[d2] {nt}")
            feat, names, parts = compute_depth2(graph, nt, d1_store, args)
            if feat is None:
                continue
            d2_store[nt] = {"feat": feat, "names": names}
            for n, p in zip(names, parts):
                all_parts[n] = p
                readable[n] = n.replace("__", " ")

    # ── Normalize + save ──
    for nt, store in d1_store.items():
        normed, mean, std = znorm(store["feat"])
        os.makedirs(osp.join(dfsdir, nt), exist_ok=True)
        torch.save(normed, osp.join(dfsdir, nt, "d1.pt"))
        meta.setdefault(nt, {})["d1"] = {
            "names": store["names"],
            "mean": mean.tolist(), "std": std.tolist(),
        }
    for nt, store in d2_store.items():
        normed, mean, std = znorm(store["feat"])
        os.makedirs(osp.join(dfsdir, nt), exist_ok=True)
        torch.save(normed, osp.join(dfsdir, nt, "d2.pt"))
        meta.setdefault(nt, {})["d2"] = {
            "names": store["names"],
            "mean": mean.tolist(), "std": std.tolist(),
        }

    # ── Name embeddings ──
    dfs1_embs = {}
    # two-stage: d1 embs first (d2 composition references them)
    d1_names = {n for st in d1_store.values() for n in st["names"]}
    stage1 = {n: p for n, p in all_parts.items() if n in d1_names}
    dfs1_embs = build_nameemb(stage1, graph, {}, args.emb_dim, args.nomic,
                              {n: readable[n] for n in stage1})
    stage2 = {n: p for n, p in all_parts.items() if n not in d1_names}
    dfs2_embs = build_nameemb(stage2, graph, dfs1_embs, args.emb_dim,
                              args.nomic, {n: readable[n] for n in stage2})
    torch.save({**existing_nameemb, **dfs1_embs, **dfs2_embs},
               osp.join(dfsdir, "dfsfeatnameemb.pt"))

    with open(osp.join(dfsdir, "metadfs.yaml"), "w") as f:
        yaml.dump(meta, f)
    total_d1 = sum(len(s["names"]) for s in d1_store.values())
    total_d2 = sum(len(s["names"]) for s in d2_store.values())
    print(f"Done. {len(d1_store)} types, {total_d1} d1 feats, "
          f"{total_d2} d2 feats -> {dfsdir}")


if __name__ == "__main__":
    main()
