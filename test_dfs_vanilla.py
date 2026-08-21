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

    # pick a node with >=1 valid neighbor under the rule
    stats_mean = torch.tensor(meta["mean"][j])
    stats_std = torch.tensor(meta["std"][j])
    stored = dfs["feat"][(NT, "d1")][:, j] * stats_std + stats_mean

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
