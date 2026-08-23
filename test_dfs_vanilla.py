"""Test battery for the DFS integration against VANILLA Griffin.

Validates the four risk areas of the DFS change without needing a GPU:

  T1  Backward compatibility — dfs off is bit-identical (same RNG seed)
  T2  Shape/name integrity   — appended columns match name embeddings,
                               depth is cumulative, hop-0 leaf path works
  T3  Mask correctness       — target column still masked, DFS columns
                               visible, fewshot similarity mask width safe
  T4  Leakage hand-check     — a d1 feature recomputed by hand from raw
                               adjacency (strict-past only) matches the
                               converter's stored value
  T5  Vanilla model forward  — hmodel.GriffinMod (NOT the SMPNN variant)
                               consumes a DFS-enriched fewshot batch
  T6  Loader end-to-end      — LoaderWrapperTask with dfs_root=2 +
                               dfs_fewshot=1 produces a consistent batch

Run (CPU is fine, needs DFS artifacts for the rel-f1 types):
    python dataconverterdfs.py datasets/joint-v65 --nodetypes \
        rel-f1-drivers rel-f1-results rel-f1-races
    python test_dfs_vanilla.py [--dataset datasets/joint-v65]
"""
import argparse
import math

import torch
import yaml

from hdataset import Graph, Task
from hFloatEmb import SimpleRepeater
from hloaderwrapper import LoaderWrapperTask, pad_feat_mask

INT64_MIN = -9223372036854775808
NT = "rel-f1-drivers"          # small, fully converted type
TASK = "rel-f1-driver-position"


def t1_backward_compat(g, fe):
    """Same seed => dfs-off subgraph is identical, and the native columns
    of a dfs-on subgraph equal the dfs-off features exactly."""
    idx = torch.arange(32)
    torch.manual_seed(1234)
    n_off, adj_off, nne_off, _, _ = g.subgraph(NT, idx, hop=2, floatemb=fe, fanout=5)
    torch.manual_seed(1234)
    n_off2, _, _, _, _ = g.subgraph(NT, idx, hop=2, floatemb=fe, fanout=5, dfs_depth=0)
    assert torch.equal(n_off[NT], n_off2[NT]), "dfs_depth=0 changed sampling!"

    torch.manual_seed(1234)
    n_on, _, nne_on, _, _ = g.subgraph(NT, idx, hop=2, floatemb=fe, fanout=5, dfs_depth=2)
    c_native = n_off[NT].shape[1]
    assert torch.equal(n_on[NT][:, :c_native], n_off[NT]), \
        "native columns differ when DFS is appended!"
    assert torch.equal(nne_on[NT][:c_native], nne_off[NT]), \
        "native name embeddings differ when DFS is appended!"
    print(f"T1 backward-compat: OK (native C={c_native}, "
          f"dfs-on C={n_on[NT].shape[1]})")
    return c_native


def t2_shapes(g, fe, c_native):
    idx = torch.arange(16)
    meta = g.load_dfs()["meta"][NT]
    c1_expect = len(meta["d1"]["names"])
    c2_expect = len(meta.get("d2", {}).get("names", []))

    _, nne1 = g.getdfsfeat(NT, idx, 1, fe)
    f1, _ = g.getdfsfeat(NT, idx, 1, fe)
    f2, nne2 = g.getdfsfeat(NT, idx, 2, fe)
    assert f1.shape == (16, c1_expect, 512), f1.shape
    assert f2.shape == (16, c1_expect + c2_expect, 512), f2.shape
    assert nne2.shape[0] == f2.shape[1], "nameemb misaligned with columns"

    # hop-0 (fewshot leaf) path
    n_leaf, adj_leaf, nne_leaf, _, _ = g.subgraph(
        NT, idx, hop=0, floatemb=fe, fanout=5, dfs_depth=1)
    assert len(adj_leaf) == 0, "hop-0 must produce no edges"
    assert n_leaf[NT].shape[1] == c_native + c1_expect
    assert nne_leaf[NT].shape[0] == n_leaf[NT].shape[1]
    print(f"T2 shapes/names: OK (d1={c1_expect}, d2={c2_expect}, "
          f"leaf C={n_leaf[NT].shape[1]})")


def t3_masks(g, task, fe, taskname=TASK):
    args = {"floatemb": fe, "fanout": 5, "hop": 2,
            "dfs_depth": 2, "dfs_fewshot_depth": 1}
    ds = LoaderWrapperTask(g, batch_size=8, shuffle=False, subgraphargs=args,
                           task=task, tasknames=[taskname], split="train",
                           fewshotfanout=3)
    import accelerate
    acc = accelerate.Accelerator(cpu=True)
    ds.rebuild_indice(acc)
    node, mask, taskfeat, ei, eat, ea, label, y, mapping = ds[0]

    # mask[0] must span all root columns (native + DFS)
    assert mask[0].shape[1] == node[0][1].shape[1], \
        f"mask width {mask[0].shape[1]} != feat width {node[0][1].shape[1]}"

    # DFS columns must never be masked on seed rows
    meta = yaml.safe_load(open(f"{g.path}/metatask.yaml"))[taskname]
    tfm = torch.tensor(
        [f not in meta["masked_feat"]
         for f in g.metanode[meta["target_type"]]["feat"]], dtype=torch.bool)
    c_native = tfm.shape[0]
    assert not mask[0][mapping][:, c_native:].any(), \
        "a DFS column is masked on seed rows — padding broke"
    # native masked columns must still be masked
    expect_native = torch.logical_not(tfm)
    assert torch.equal(mask[0][mapping][0, :c_native], expect_native), \
        "native target-column masking changed!"

    # pad_feat_mask unit checks
    m = torch.tensor([True, False, True])
    assert torch.equal(pad_feat_mask(m, 5),
                       torch.tensor([True, False, True, True, True]))
    assert torch.equal(pad_feat_mask(m, 3), m)
    print(f"T3 masks: OK (root width {mask[0].shape[1]}, "
          f"native {c_native}, masked_feat={meta['masked_feat']})")


def t4_leakage_handcheck(g):
    """Recompute one node's d1 mean feature from raw adjacency with the
    strict-past rule and compare to the converter's stored value."""
    dfs = g.load_dfs()
    meta = dfs["meta"][NT]["d1"]
    # find a mean feature and its relation + source column
    target_feat = None
    for j, name in enumerate(meta["names"]):
        if "__mean__" in name:
            target_feat = (j, name)
            break
    assert target_feat is not None
    j, name = target_feat
    # name: dfs1__mean__<reltag>__<col>; recover the raw relation by tag match
    from dataconverterdfs import sanitize
    rels = g.metanode[NT]["in"] + g.metanode[NT]["out"]
    rel = next(r for r in rels if f"__{sanitize(r)}__" in name)
    col = name.split(f"__{sanitize(rel)}__")[-1]
    from hdataset import edgename2tail, INF
    tail = edgename2tail(rel)

    node = g.nodes[NT]
    ts_all = node.feat["timestamp"]
    if not torch.is_tensor(ts_all):
        ts_all = torch.tensor(ts_all)
    temporal = bool((ts_all != INT64_MIN).any())

    # artifacts store RAW values (format v2); normalization is applied at read
    stored = dfs["feat"][(NT, "d1")][:, j]

    colvals = g.nodes[tail].feat[col]
    if not torch.is_tensor(colvals):
        colvals = torch.tensor(colvals)

    checked = 0
    for i in range(0, len(node), max(1, len(node) // 50)):
        idx = torch.tensor([i])
        ts = ts_all[idx] if temporal else None
        ttadj = node.getedge(idx, INF, ts)
        if rel not in ttadj:
            continue
        src, tar = ttadj[rel]
        if len(tar) == 0:
            continue
        # NaN-skipping mean, matching featuretools/SQL semantics
        v = colvals[tar].to(torch.float32)
        v = v[torch.isfinite(v)]
        if len(v) == 0:
            continue
        hand = v.mean()
        assert torch.allclose(hand, stored[i], atol=1e-3), \
            f"node {i}: hand={hand.item():.5f} stored={stored[i].item():.5f}"
        checked += 1
        if checked >= 5:
            break
    assert checked > 0, "no node with neighbors found to check"
    print(f"T4 leakage hand-check: OK ({checked} nodes, feature '{name}', "
          f"temporal={temporal})")


def t5_vanilla_forward(g, task, fe):
    from hmodel import GriffinMod as VanillaGriffin
    args = {"floatemb": fe, "fanout": 5, "hop": 2,
            "dfs_depth": 2, "dfs_fewshot_depth": 1}
    ds = LoaderWrapperTask(g, batch_size=8, shuffle=False, subgraphargs=args,
                           task=task, tasknames=[TASK], split="train",
                           fewshotfanout=3)
    import accelerate
    acc = accelerate.Accelerator(cpu=True)
    ds.rebuild_indice(acc)
    node, mask, taskfeat, ei, eat, ea, label, y, mapping = ds[0]
    m = VanillaGriffin(hiddim=512, num_mp=4)   # vanilla, NOT the SMPNN file
    out = m(node, mask, taskfeat, ei, eat, ea)
    assert out.shape[1] == 512 and torch.isfinite(out).all()
    # gradient flows
    out.sum().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert any(grads), "no gradients reached the vanilla model"
    print(f"T5 vanilla forward+backward: OK (out {tuple(out.shape)})")


def t6_loader_end_to_end(g, task, fe):
    for dfs_fs, dfs_root in ((0, 0), (1, 0), (0, 2), (1, 2)):
        args = {"floatemb": fe, "fanout": 5, "hop": 2,
                "dfs_depth": dfs_root, "dfs_fewshot_depth": dfs_fs}
        ds = LoaderWrapperTask(g, batch_size=4, shuffle=False,
                               subgraphargs=args, task=task,
                               tasknames=[TASK], split="valid",
                               fewshotfanout=3)
        import accelerate
        acc = accelerate.Accelerator(cpu=True)
        ds.rebuild_indice(acc)
        node, mask, *_ = ds[0]
        assert mask[0].shape == node[0][1].shape[:2]
    print("T6 loader E0/E1/E2/E3 configs: OK")


def _task_pairs(g, taskname, n=5, seed=0):
    """A few (nodeidx, task-timestamp) pairs from a task table."""
    import datasets as hds
    import numpy as np
    meta = yaml.safe_load(open(f"{g.path}/metatask.yaml"))[taskname]
    ds = hds.load_from_disk(f"{g.path}/task/{taskname}")
    rng = np.random.RandomState(seed)
    sel = sorted(rng.choice(len(ds), min(n, len(ds)), replace=False).tolist())
    rows = ds.select(sel)
    idx = torch.as_tensor(np.asarray(rows["nodeidx"]), dtype=torch.int64)
    cut = torch.as_tensor(np.asarray(rows["timestamp"]), dtype=torch.int64)
    return meta["target_type"], idx, cut


def t7_cutoff_handcheck(g, taskname=TASK):
    """Recompute count/mean at a real task cutoff straight from adjacency
    and compare to the online path — then show the own-timestamp store
    disagrees, which is the leak this fix closes."""
    import dfscore
    from hdataset import INF

    nt, idx, cut = _task_pairs(g, taskname, n=8)
    layout = g._dfs_layout(nt)
    node = g.nodes[nt]

    got = dfscore.compute_d1_at(g, nt, idx, cut, layout, dfscore.RowGather(g))

    # hand recomputation, one (row, cutoff) pair at a time
    checked = 0
    for i in range(len(idx)):
        one, tau = idx[i:i + 1], cut[i:i + 1]
        ttadj = node.getedge(one, INF, tau)
        for r, cols in layout["relcols"].items():
            tag = dfscore.sanitize(r)
            tar = ttadj[r][1] if r in ttadj else torch.zeros(0, dtype=torch.int64)
            k = layout["col_index"].get(f"dfs1__count__{tag}")
            if k is not None:
                assert torch.isclose(got[i, k], torch.log1p(torch.tensor(float(len(tar))))), \
                    f"count mismatch for {r} on row {idx[i].item()}"
            for c in cols:
                k = layout["col_index"].get(f"dfs1__mean__{tag}__{c}")
                if k is None or len(tar) == 0:
                    continue
                v = g.nodes[edgename2tail_(r)].feat[c]
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                v = v[tar].to(torch.float32)
                v = v[torch.isfinite(v)]
                hand = v.mean() if len(v) else torch.tensor(0.0)
                assert torch.allclose(hand, got[i, k], atol=1e-4), \
                    f"mean mismatch {r}.{c} row {idx[i].item()}: " \
                    f"hand={hand.item()} got={got[i, k].item()}"
                checked += 1
    assert checked > 0, "no (row, cutoff) pair had neighbors to check"

    # the loader's resolution path must return exactly these numbers
    vals = g._dfs_values(nt, idx, ["d1"], cut)
    assert vals and vals[0][0] == "d1"
    assert torch.allclose(vals[0][1], got, atol=1e-5), \
        "Graph._dfs_values disagrees with dfscore.compute_d1_at"

    # ...and must differ from the own-timestamp store on a non-temporal type
    stored = g.load_dfs()["feat"][(nt, "d1")][idx]
    temporal = g.load_dfs()["meta"][nt].get("temporal", True)
    cnt = [k for k, n in enumerate(layout["names"]) if "__count__" in n]
    if not temporal and cnt:
        assert (stored[:, cnt] >= got[:, cnt] - 1e-5).all(), \
            "own-ts store should be a superset of the at-cutoff window"
        assert not torch.allclose(stored[:, cnt], got[:, cnt]), \
            "non-temporal type: at-cutoff equals all-history — cutoff ignored!"
    print(f"T7 cutoff hand-check: OK ({checked} aggregates on {nt}, "
          f"temporal={temporal}, "
          f"all-history count sum={stored[:, cnt].sum():.2f} vs "
          f"at-cutoff={got[:, cnt].sum():.2f})")


def edgename2tail_(r):
    from hdataset import edgename2tail
    return edgename2tail(r)


def t9_nontemporal_hub_d2(g, taskname=TASK):
    """Depth 2 through a NON-temporal hub must be re-evaluated at the
    querying row's cutoff, not over all history."""
    import dfscore
    nt, idx, cut = _task_pairs(g, taskname, n=8)
    dfs = g.load_dfs()
    if "d2" not in dfs["meta"].get(nt, {}):
        print("T9 non-temporal hub d2: skipped (no d2 for this type)")
        return
    plan = g._dfs_plan(nt)
    layouts = {nt: g._dfs_layout(nt)}
    hubs = set()
    for r, _, _, _ in plan:
        t = edgename2tail_(r)
        layouts[t] = g._dfs_layout(t)
        if not dfscore.is_temporal(g, t):
            hubs.add(t)
    src = dfscore.RowGather(g)
    at = dfscore.compute_d2_at(g, nt, idx, cut, plan, layouts,
                               lambda t: g._dfs_raw(t, "d1"), src)
    skipped = dfscore.compute_d2_at(g, nt, idx, cut, plan, layouts,
                                    lambda t: g._dfs_raw(t, "d1"), src,
                                    nontemporal_hubs="skip")
    assert at.shape == (len(idx), len(plan))
    if hubs:
        assert not torch.allclose(at, skipped), \
            "non-temporal hubs present but recompute == skip"
    print(f"T9 non-temporal hub d2: OK ({len(plan)} d2 feats, "
          f"non-temporal hubs={sorted(hubs) or 'none'})")


def t10_label_relation_guard(g):
    """No DFS column may be built from an is_target relation by default,
    and asking for one without a cutoff must be refused."""
    import subprocess, sys
    from hdataset import edgename2tail
    dfs = g.load_dfs()
    for nt in dfs["meta"]:
        for r in g._dfs_layout(nt)["rels"]:
            assert not g.nodes[edgename2tail(r)].is_target, \
                f"{nt}: label relation {r} present in the default layout"
    p = subprocess.run(
        [sys.executable, "dataconverterdfs.py", g.path,
         "--nodetypes", NT, "--include_target_rels"],
        capture_output=True, text=True,
    )
    assert p.returncode != 0 and "cutoff" in (p.stdout + p.stderr).lower(), \
        f"--include_target_rels without --cutoff_stats was not refused: {p.stdout}{p.stderr}"
    print("T10 label-relation guard: OK (no is_target tails in any layout; "
          "--include_target_rels refused without --cutoff_stats)")


def t11_leaf_cutoff(g, fe, taskname=TASK):
    """A hop-0 (fewshot-leaf) subgraph must carry DFS columns evaluated at
    the cutoff handed to it, matching a direct getdfsfeat call."""
    nt, idx, cut = _task_pairs(g, taskname, n=6)
    n_leaf, adj_leaf, nne_leaf, _, _ = g.subgraph(
        nt, idx, hop=0, floatemb=fe, fanout=5, dfs_depth=1, timestamp=cut)
    assert len(adj_leaf) == 0, "hop-0 must produce no edges"
    direct, dnames = g.getdfsfeat(nt, idx, 1, fe, cutoff=cut)
    c_native = n_leaf[nt].shape[1] - direct.shape[1]
    assert torch.equal(n_leaf[nt][:, c_native:], direct), \
        "leaf DFS columns are not the ones getdfsfeat(cutoff=...) returns"

    # and the cutoff must actually matter: no-cutoff gives different columns
    nocut, _ = g.getdfsfeat(nt, idx, 1, fe, cutoff=None)
    if not g.load_dfs()["meta"][nt].get("temporal", True):
        assert not torch.equal(direct, nocut), \
            "leaf columns identical with and without a cutoff"
    print(f"T11 leaf cutoff: OK ({direct.shape[1]} DFS columns at the "
          f"seed's query time)")


def t12_end_to_end_cutoff(g, task, fe, taskname=TASK):
    """LoaderWrapperTask must drive the whole cutoff path without shape or
    alignment errors, for every E-cell config."""
    import accelerate
    for dfs_fs, dfs_root in ((0, 0), (1, 0), (0, 2), (1, 2)):
        args = {"floatemb": fe, "fanout": 5, "hop": 2,
                "dfs_depth": dfs_root, "dfs_fewshot_depth": dfs_fs}
        ds = LoaderWrapperTask(g, batch_size=8, shuffle=False,
                               subgraphargs=args, task=task,
                               tasknames=[taskname], split="test",
                               fewshotfanout=3)
        acc = accelerate.Accelerator(cpu=True)
        ds.rebuild_indice(acc)
        node, mask, *_ = ds[0]
        assert mask[0].shape == node[0][1].shape[:2]
    print("T12 loader end-to-end at task cutoffs: OK (E0/E1/E2/E3)")


def t11_leaf_cutoff_wiring(g, task, fe):
    """Fewshot leaves' DFS columns are evaluated at the SEED's task
    timestamp (C6+C7): spy on getdfsfeat during a real loader batch."""
    calls = []
    orig = Graph.getdfsfeat

    def spy(self, nodetype, idx, depth, floatemb, cutoff=None):
        calls.append((nodetype, idx.clone(),
                      None if cutoff is None else torch.as_tensor(cutoff).clone()))
        return orig(self, nodetype, idx, depth, floatemb, cutoff)

    args = {"floatemb": fe, "fanout": 5, "hop": 2,
            "dfs_depth": 0, "dfs_fewshot_depth": 1}   # leaf path only
    ds = LoaderWrapperTask(g, batch_size=8, shuffle=False, subgraphargs=args,
                           task=task, tasknames=[TASK], split="train",
                           fewshotfanout=3)
    import accelerate
    acc = accelerate.Accelerator(cpu=True)
    ds.rebuild_indice(acc)
    Graph.getdfsfeat = spy
    try:
        ds[0]
    finally:
        Graph.getdfsfeat = orig

    assert calls, "fewshot leaves made no getdfsfeat call"
    valid_taus = set(int(t) for t in task.tasks[TASK]["timestamp"])
    for nt, idx, cutoff in calls:
        assert cutoff is not None, "leaf DFS evaluated without a cutoff!"
        assert len(cutoff) == len(idx)
        assert all(int(c) in valid_taus for c in cutoff), \
            "leaf cutoff is not a task timestamp of this batch's seeds"
    print(f"T11 leaf cutoff wiring: OK ({len(calls)} call(s), "
          f"{sum(len(c[1]) for c in calls)} leaf rows, all at seed task times)")


def t13_temporal_leaf_own_ts(g, fe):
    """Fewshot leaves of a TEMPORAL root type keep their OWN timestamp as
    DFS cutoff even when the loader hands them the seed's (later) cutoff:
    a temporal leaf's visible outcome belongs to its own row time, and
    aggregates past that time would encode the outcome's consequences.
    Non-temporal leaves keep the seed's cutoff (t11 wiring covers the
    loader path; this checks the override boundary directly)."""
    from hloaderwrapper import LoaderWrapper
    args = {"floatemb": fe, "fanout": 5, "hop": 2,
            "dfs_depth": 0, "dfs_fewshot_depth": 1}
    lw = LoaderWrapper(g, batch_size=4, shuffle=False, subgraphargs=args,
                       fewshotfanout=2)
    calls = []
    orig = Graph.getdfsfeat

    def spy(self, nodetype, idx, depth, floatemb, cutoff=None):
        calls.append((idx.clone(),
                      None if cutoff is None else torch.as_tensor(cutoff).clone()))
        return orig(self, nodetype, idx, depth, floatemb, cutoff)

    nt = "rel-f1-results"
    assert g.dfs_is_temporal(nt), f"{nt} should be temporal"
    own_all = g.nodes[nt].feat["timestamp"]
    own_all = own_all if torch.is_tensor(own_all) else torch.tensor(own_all)
    ind = torch.nonzero(own_all != INT64_MIN).flatten()[:6]
    seed_tau = torch.full((len(ind),), int(own_all.max()) + 10_000,
                          dtype=torch.int64)
    Graph.getdfsfeat = spy
    try:
        lw.fewshotsubgraph(nt, ind, timestamp=seed_tau)
    finally:
        Graph.getdfsfeat = orig
    assert calls, "temporal leaf path made no getdfsfeat call"
    idx, cutoff = calls[-1]
    assert cutoff is not None
    assert torch.equal(cutoff, own_all[idx]), \
        "temporal leaf DFS cutoff should be the leaf's OWN timestamp"
    assert not torch.equal(cutoff, seed_tau), "override did not fire"

    calls.clear()
    ind2 = torch.arange(4)
    tau2 = torch.full((4,), int(own_all.max()), dtype=torch.int64)
    Graph.getdfsfeat = spy
    try:
        lw.fewshotsubgraph(NT, ind2, timestamp=tau2)   # non-temporal type
    finally:
        Graph.getdfsfeat = orig
    idx2, cutoff2 = calls[-1]
    assert cutoff2 is not None and bool((cutoff2 == tau2[0]).all()), \
        "non-temporal leaf must keep the seed's cutoff"
    print("T13 temporal leaf own-ts override: OK "
          f"({len(idx)} temporal leaves at own ts, "
          f"{len(idx2)} non-temporal leaves at seed tau)")


def t14_sentinel_guards(g, task, fe):
    """INT64_MIN cutoff handling. (a) An all-sentinel vector — Completion
    on a non-temporal root, whose 'own timestamp' IS the sentinel — serves
    the all-time store instead of silently zero-filling (getedge at
    INT64_MIN keeps no neighbor). (b) A sentinel mixed into real task
    cutoffs raises. (c) A supervised task without timestamps refuses to
    run with DFS enabled rather than fall back to all-time features."""
    idx = torch.arange(6)
    allmin = torch.full((6,), INT64_MIN, dtype=torch.int64)
    a, _ = g.getdfsfeat(NT, idx, 1, fe, cutoff=allmin)
    b, _ = g.getdfsfeat(NT, idx, 1, fe, cutoff=None)
    assert torch.allclose(a, b), \
        "all-sentinel cutoff must resolve to the all-time store"

    mixed = allmin.clone()
    mixed[0] = 10**9
    try:
        g.getdfsfeat(NT, idx, 1, fe, cutoff=mixed)
        raise AssertionError("mixed sentinel cutoff did not raise")
    except RuntimeError as e:
        assert "INT64_MIN" in str(e)

    meta = task.metatask[TASK]
    old = meta["hastimestamp"]
    meta["hastimestamp"] = False
    args = {"floatemb": fe, "fanout": 5, "hop": 1,
            "dfs_depth": 1, "dfs_fewshot_depth": 0}
    ds = LoaderWrapperTask(g, batch_size=4, shuffle=False, subgraphargs=args,
                           task=task, tasknames=[TASK], split="train",
                           fewshotfanout=0)
    import accelerate
    acc = accelerate.Accelerator(cpu=True)
    ds.rebuild_indice(acc)
    try:
        ds[0]
        raise AssertionError("timestamp-less task with DFS on did not raise")
    except RuntimeError as e:
        assert "cutoff" in str(e) or "timestamp" in str(e)
    finally:
        meta["hastimestamp"] = old
    print("T14 sentinel guards: OK (all-sentinel -> store, mixed raises, "
          "timestamp-less task refused)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/joint-v65")
    a = ap.parse_args()

    g = Graph(a.dataset)
    task = Task(a.dataset)
    fe = SimpleRepeater(512)

    c_native = t1_backward_compat(g, fe)
    t2_shapes(g, fe, c_native)
    t3_masks(g, task, fe)
    t4_leakage_handcheck(g)
    t5_vanilla_forward(g, task, fe)
    t6_loader_end_to_end(g, task, fe)
    t7_cutoff_handcheck(g)
    t9_nontemporal_hub_d2(g)
    t10_label_relation_guard(g)
    t11_leaf_cutoff(g, fe)
    t11_leaf_cutoff_wiring(g, task, fe)
    t12_end_to_end_cutoff(g, task, fe)
    t13_temporal_leaf_own_ts(g, fe)
    t14_sentinel_guards(g, task, fe)

    # Opportunistic: the strongest masking case — virus-wnv-pred has a
    # real masked target column (virus-Virus___WnvPresent). Runs once
    # the DFS converter has processed virus-Virus.
    import os.path as osp
    if osp.exists(osp.join(a.dataset, "dfs", "virus-Virus", "d1.pt")):
        g._dfs = None  # re-load artifacts written after our first load
        t3_masks(g, task, fe, taskname="virus-wnv-pred")
        print("T3b masked-target task (virus-wnv-pred): OK")
    else:
        print("T3b skipped (virus-Virus DFS artifacts not yet computed)")
    print("\nALL DFS x VANILLA-GRIFFIN TESTS PASSED")


if __name__ == "__main__":
    main()
