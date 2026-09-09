"""Offline Deep Feature Synthesis (DFS) over a processed Griffin dataset.

Computes per-node DFS aggregate features in the style of
RDBLearn (arXiv 2602.18495) / fastdfs (github.com/HKUSHXLab/fastdfs),
but natively over Griffin's processed structures (CSR-ish Arrow
adjacency + node feature tables) instead of raw DataFrames — Griffin's
``Node.getedge`` already provides the timestamp-filtered full
neighborhood, so one-hop DFS is a fixed-weight unsampled aggregation
pass over it. The aggregation core lives in ``dfscore.py`` and is shared
with the online (cutoff-aware) loader path.

Depth 1 (per node n of type A, per relation r to neighbor type B):
    dfs1__count__{r}            number of strictly-past neighbors
    dfs1__mean__{r}__{c}        mean of float column c over those neighbors
    dfs1__max__{r}__{c}         max  of float column c over those neighbors
    (optionally min/std via --d1_prims)

Depth 2 (aggregates of neighbors' depth-1 aggregates):
    dfs2__mean__{r}__{d1name}   mean of B's d1 feature over the same
                                strictly-past neighbor set, EXCLUDING
                                B-features derived from the reverse of r
                                (no-backtrack rule)

WHAT THIS FILE DOES AND DOES NOT PRODUCE
----------------------------------------
The stored ``d1.pt`` / ``d2.pt`` are evaluated at each row's **own
timestamp**, which is the correct query time only for Completion-style
pretraining. Task-driven training must evaluate at the **seed's task
timestamp**; that is computed online by ``Graph.getdfsfeat(..., cutoff=)``
using ``dfscore``. This converter's job for the task path is to fix the
things that must be shared and stable:

  * the column layout (names / order), so checkpoints stay bound to it;
  * the normalization statistics, including a separate set sampled at
    real task cutoffs (``--cutoff_stats``), since the at-cutoff value
    distribution differs from the at-own-timestamp one;
  * the name embeddings;
  * the own-timestamp d1, reused as the cheap temporal-hub source at
    depth 2.

Values on disk are **raw** (unnormalized; counts log1p'd). Normalization
is applied at read time from metadfs.yaml, so the online and offline
paths agree exactly rather than to within a de-normalization round trip.

Leakage rules:
  * A neighbor row is included iff nb_ts < cutoff (strict), where cutoff
    is the row's own timestamp here and the task timestamp online.
  * Non-temporal neighbors (ts == INT64_MIN) always pass — dimension rows.
  * Depth 2 excludes the traversed relation's reverse (no A->B->A).
  * Relations into ``is_target`` types are dropped unless
    --include_target_rels (their columns are task labels).

Outputs under <dstpath>/dfs/:
    <nodetype>/d1.pt        float32 (num_nodes, C1) RAW
    <nodetype>/d2.pt        float32 (num_nodes, C2) RAW
    metadfs.yaml            layout + normalization stats
    dfsfeatnameemb.pt       {feature_name: (D,) float32 name embedding}

Usage:
    python dataconverterdfs.py datasets/joint-v65 \
        [--nodetypes rel-f1-drivers rel-f1-results] \
        [--cutoff_stats --tasks others-1] \
        [--batch_rows 65536] [--max_cols_per_rel 8] [--max_d2_feats 32] \
        [--skip_d2] [--nomic] [--emb_dim 512]
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch
import yaml

import dfscore
from dfscore import (
    DEFAULT_PRIMS, FullColumnCache, apply_norm, build_d1_layout,
    build_d2_plan, compute_d1_at, compute_d2_at, compose_nameemb,
    nondegeneracy_score, plan_names, plan_parts, primitive_tag, sanitize,
    znorm_stats,
)
from hdataset import Graph, edgename2tail

INT64_MIN = dfscore.INT64_MIN

#: Bumped when the on-disk layout changes in a way old artifacts cannot
#: satisfy. v2: raw (unnormalized) storage + cutoff-aware metadata.
FORMAT_VERSION = 2


def own_timestamps(graph, nodetype):
    """The cutoff vector for 'own timestamp' mode: the row's own ts for
    temporal types, None (no cutoff at all) for non-temporal ones."""
    if not dfscore.is_temporal(graph, nodetype):
        return None
    ts = graph.nodes[nodetype].feat["timestamp"]
    if not torch.is_tensor(ts):
        ts = torch.tensor(ts)
    return ts


def sample_task_cutoffs(path, tasks, n_sample, seed=0):
    """Sample (target_type, nodeidx, timestamp) triples from task tables.

    Used to fit normalization stats on the distribution the task path will
    actually see. Sampling is enough: we need mean/std per column, not the
    exact population.
    """
    import datasets as hds
    with open(osp.join(path, "metatask.yaml")) as f:
        metatask = yaml.safe_load(f)
    rng = np.random.RandomState(seed)
    out = {}
    for taskname in tasks:
        if taskname not in metatask:
            raise KeyError(f"unknown task {taskname!r}; "
                           f"known: {sorted(metatask)}")
        m = metatask[taskname]
        if not m.get("hastimestamp"):
            print(f"  [cutoff-stats] {taskname}: no task timestamp, skipping")
            continue
        nt = m["target_type"]
        ds = hds.load_from_disk(osp.join(path, "task", taskname))
        n = len(ds)
        take = min(n_sample, n)
        sel = np.sort(rng.choice(n, take, replace=False))
        rows = ds.select(sel)
        idx = torch.as_tensor(np.asarray(rows["nodeidx"]), dtype=torch.int64)
        cut = torch.as_tensor(np.asarray(rows["timestamp"]), dtype=torch.int64)
        prev_i, prev_c = out.get(nt, (None, None))
        if prev_i is None:
            out[nt] = (idx, cut)
        else:
            out[nt] = (torch.cat((prev_i, idx)), torch.cat((prev_c, cut)))
        print(f"  [cutoff-stats] {taskname}: sampled {take}/{n} rows of {nt}")
    return out


def resolve_tasks(path, tasks):
    """Expand task group names (others-1, ALLTASK, ...) the same way the
    trainers do."""
    with open(osp.join(path, "metatask.yaml")) as f:
        metatask = yaml.safe_load(f)
    if len(tasks) == 1:
        t = tasks[0]
        if t == "ALLTASK":
            return list(metatask)
        if t in ("commerce-1", "commerce-2", "others-1", "others-2"):
            with open("task_names.yaml") as f:
                return yaml.safe_load(f)[t]
    return tasks


def build_nameemb(all_parts, graph, dfs1_embs, dim, use_nomic, readable):
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
    ap.add_argument("--d1_prims", type=str, default=",".join(DEFAULT_PRIMS),
                    help="comma-separated subset of count,mean,max,min,std. "
                         "Default count,mean,max keeps existing E-cells "
                         "comparable; min,std widen toward the fastdfs set.")
    ap.add_argument("--include_target_rels", action="store_true",
                    help="aggregate relations whose tail is an is_target type. "
                         "Their columns are task labels, so this is only "
                         "defensible under an explicit cutoff — it is refused "
                         "unless --cutoff_stats is also given.")
    ap.add_argument("--d2_nontemporal_hubs", choices=["recompute", "skip"],
                    default="recompute",
                    help="how depth 2 handles a non-temporal hub. recompute "
                         "(default) evaluates the hub's d1 at the querying "
                         "row's cutoff; skip zero-fills. Never falls back to "
                         "the hub's all-time d1, which leaks the future.")
    ap.add_argument("--d2_hub_budget", type=int, default=2_000_000,
                    help="max unique (hub, cutoff) pairs to recompute before "
                         "zero-filling that relation.")
    ap.add_argument("--cutoff_stats", action="store_true",
                    help="additionally fit normalization stats at real task "
                         "cutoffs (needs --tasks). Required for any task-path "
                         "training on a non-temporal target type.")
    ap.add_argument("--tasks", nargs="*", default=["ALLTASK"],
                    help="tasks whose cutoffs to sample for --cutoff_stats. "
                         "Accepts group names (others-1, ALLTASK, ...).")
    ap.add_argument("--cutoff_stats_sample", type=int, default=20000,
                    help="rows sampled per task when fitting cutoff stats.")
    ap.add_argument("--nomic", action="store_true",
                    help="use Nomic to embed feature names (needs GPU + "
                         "sentence_transformers); default is compositional")
    ap.add_argument("--emb_dim", type=int, default=512)
    args = ap.parse_args()

    prims = tuple(p.strip() for p in args.d1_prims.split(",") if p.strip())
    unknown = set(prims) - {"count", *dfscore.VALUE_PRIMS}
    if unknown:
        raise SystemExit(f"unknown --d1_prims: {sorted(unknown)}")
    if args.include_target_rels and not args.cutoff_stats:
        raise SystemExit(
            "--include_target_rels aggregates task-label columns. That is only "
            "leak-free when the features are evaluated at the task cutoff, so "
            "it requires --cutoff_stats (and the task path, which always "
            "passes a cutoff). Re-run with --cutoff_stats --tasks <...>."
        )

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
        if meta.get("__format__", 1) != FORMAT_VERSION:
            print(f"  [format] existing artifacts are v{meta.get('__format__', 1)}, "
                  f"this converter writes v{FORMAT_VERSION}; discarding old "
                  f"metadata (regenerate every node type)")
            meta = {}
        for nt in types:
            meta.pop(nt, None)   # recomputed below
    existing_nameemb = {}
    if osp.exists(osp.join(dfsdir, "dfsfeatnameemb.pt")):
        existing_nameemb = torch.load(
            osp.join(dfsdir, "dfsfeatnameemb.pt"),
            map_location="cpu", weights_only=True,
        )

    colsource = FullColumnCache(graph)

    # ── Layouts (needed for every type reachable as a depth-2 hub) ──
    layout_types = sorted(
        {nt for nt in graph.metanode
         if not graph.nodes[nt].is_target and graph.nodes[nt].adj is not None}
        | set(types)
    )
    layouts = {
        nt: build_d1_layout(graph, nt, args.max_cols_per_rel, prims,
                            args.include_target_rels)
        for nt in layout_types
    }
    for nt in types:
        if layouts[nt]["dropped_target_rels"]:
            print(f"  [{nt}] dropped label relations: "
                  f"{layouts[nt]['dropped_target_rels']}")

    # ── Pass 1: depth 1 at each row's own timestamp ──
    d1_store = {}
    for nt in types:
        layout = layouts[nt]
        if not layout["names"]:
            print(f"[d1] {nt}: no aggregatable relations; skipping")
            continue
        num = graph.metanode[nt]["num"]
        print(f"[d1] {nt} ({num} rows, {len(layout['names'])} feats)")
        feat = compute_d1_at(
            graph, nt, torch.arange(num), own_timestamps(graph, nt), layout,
            colsource, args.batch_rows, progress=nt,
        )
        d1_store[nt] = feat
        for n, p in zip(layout["names"], layout["parts"]):
            all_parts[n] = p
            readable[n] = n.replace("__", " ")

    def d1_raw_of(tail):
        return d1_store.get(tail)

    # ── Pass 2: depth 2 (uses unnormalized d1 of neighbors) ──
    d2_store, d2_plans = {}, {}
    if not args.skip_d2:
        for nt in types:
            if nt not in d1_store:
                continue
            plan = build_d2_plan(graph, nt, layouts)
            plan = [p for p in plan if p[0] in layouts[nt]["relcols"]
                    and edgename2tail(p[0]) in d1_store]
            if not plan:
                continue
            print(f"[d2] {nt} ({len(plan)} candidate feats)")
            num = graph.metanode[nt]["num"]
            feat = compute_d2_at(
                graph, nt, torch.arange(num), own_timestamps(graph, nt), plan,
                layouts, d1_raw_of, colsource, args.batch_rows,
                args.d2_nontemporal_hubs, args.d2_hub_budget,
            )
            # Cap by non-degeneracy, not raw variance: a single epoch-scale
            # column would otherwise crowd out every real feature.
            if feat.shape[1] > args.max_d2_feats:
                keep = torch.topk(nondegeneracy_score(feat),
                                  args.max_d2_feats).indices.sort().values
                feat = feat[:, keep]
                plan = [plan[i] for i in keep.tolist()]
            d2_store[nt], d2_plans[nt] = feat, plan
            for n, p in zip(plan_names(plan), plan_parts(plan)):
                all_parts[n] = p
                readable[n] = n.replace("__", " ")

    # ── Normalization stats at real task cutoffs ──
    cutoff_stats = {}
    if args.cutoff_stats:
        tasks = resolve_tasks(args.dstpath, args.tasks)
        print(f"[cutoff-stats] sampling cutoffs from {len(tasks)} tasks")
        samples = sample_task_cutoffs(args.dstpath, tasks,
                                      args.cutoff_stats_sample)
        for nt, (idx, cut) in samples.items():
            if nt not in d1_store:
                print(f"  [cutoff-stats] {nt}: no d1 layout (not converted); "
                      f"skipping")
                continue
            print(f"  [cutoff-stats] {nt}: {len(idx)} (row, cutoff) pairs")
            s = {}
            f1 = compute_d1_at(graph, nt, idx, cut, layouts[nt], colsource,
                               args.batch_rows)
            m, sd = znorm_stats(f1)
            s["d1"] = {"mean": m.tolist(), "std": sd.tolist()}
            if nt in d2_plans:
                f2 = compute_d2_at(
                    graph, nt, idx, cut, d2_plans[nt], layouts, d1_raw_of,
                    colsource, args.batch_rows, args.d2_nontemporal_hubs,
                    args.d2_hub_budget,
                )
                m, sd = znorm_stats(f2)
                s["d2"] = {"mean": m.tolist(), "std": sd.tolist()}
            cutoff_stats[nt] = s

    # ── Save raw features + metadata ──
    meta["__format__"] = FORMAT_VERSION
    for nt, feat in d1_store.items():
        layout = layouts[nt]
        os.makedirs(osp.join(dfsdir, nt), exist_ok=True)
        torch.save(feat, osp.join(dfsdir, nt, "d1.pt"))
        mean, std = znorm_stats(feat)
        entry = meta.setdefault(nt, {})
        entry["d1"] = {
            "names": layout["names"],
            "rel_of_feat": layout["rel_of_feat"],
            "relcols": {r: list(cs) for r, cs in layout["relcols"].items()},
            "mean": mean.tolist(), "std": std.tolist(),
        }
        entry["args"] = {
            "max_cols_per_rel": args.max_cols_per_rel,
            "prims": list(prims),
            "include_target_rels": bool(args.include_target_rels),
            "d2_nontemporal_hubs": args.d2_nontemporal_hubs,
        }
        entry["temporal"] = dfscore.is_temporal(graph, nt)
        if nt in cutoff_stats and "d1" in cutoff_stats[nt]:
            entry["d1_at"] = cutoff_stats[nt]["d1"]
    for nt, feat in d2_store.items():
        os.makedirs(osp.join(dfsdir, nt), exist_ok=True)
        torch.save(feat, osp.join(dfsdir, nt, "d2.pt"))
        mean, std = znorm_stats(feat)
        entry = meta.setdefault(nt, {})
        entry["d2"] = {
            "names": plan_names(d2_plans[nt]),
            "plan": [[r, int(j)] for r, j, _, _ in d2_plans[nt]],
            "mean": mean.tolist(), "std": std.tolist(),
        }
        if nt in cutoff_stats and "d2" in cutoff_stats[nt]:
            entry["d2_at"] = cutoff_stats[nt]["d2"]

    # ── Name embeddings ──
    d1_names = {n for nt in d1_store for n in layouts[nt]["names"]}
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
    total_d1 = sum(len(layouts[nt]["names"]) for nt in d1_store)
    total_d2 = sum(len(p) for p in d2_plans.values())
    print(f"Done. {len(d1_store)} types, {total_d1} d1 feats, "
          f"{total_d2} d2 feats -> {dfsdir}")
    if cutoff_stats:
        print(f"Cutoff-mode stats fitted for: {sorted(cutoff_stats)}")
    else:
        print("NOTE: no cutoff stats fitted. Task-path training on a "
              "non-temporal target type will refuse to run; re-run with "
              "--cutoff_stats --tasks <group>.")


if __name__ == "__main__":
    main()
