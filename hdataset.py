import yaml
import math
import os.path as osp
import torch
import torch.nn.functional as F
import datasets as hds
from typing import Optional, Iterable, Union, Literal
from typing import Union
import numpy as np

INF = 100000000
MAXINT64 = 1<<62
TIMESTAMPADJNAME = "___TIMESTAMP"

class Node:
    meta: dict
    feat: hds.Dataset
    textemb: hds.Dataset
    adj: Union[hds.Dataset, None]

    def __init__(self, meta, feat, textemb, adj) -> None:
        self.meta = meta
        self.feat = feat
        self.textemb = textemb
        self.adj = adj
        assert len(self.feat) == self.meta["num"]
        if self.adj is not None:
            assert (
                len(self.adj) == self.meta["num"]
            ), f"adj has {len(self.adj)} while there are {self.meta['num']} nodes"

    def __len__(self):
        return self.meta["num"]

    @property
    def is_target(self):
        return self.meta["is_target"]

    @property
    def featlist(self):
        return self.meta["feat"]

    def getfeat(self, idx: Union[int, Iterable[int]], floatemb) -> torch.Tensor:
        data: dict = self.feat[idx]
        def unique_query_textemb(idx):
            unique_idx, inv = torch.unique(idx, return_inverse=True)
            return self.textemb[unique_idx]["emb"][inv]
        def unique_float_emb(val):
            # return floatemb(val)
            unique_val, inv = torch.unique(val, return_inverse=True)
            return floatemb(unique_val)[inv]
        data = torch.stack(
            [
                (
                    unique_query_textemb(data[_])#self.textemb[data[_]]["emb"]
                    if "Griffin_text_" in _
                    else unique_float_emb(data[_])#floatemb(data[_])
                )
                for _ in self.featlist
            ],
            dim=1,
        )  # (B, num_col, dim)
        return data

    def getedge(self, idx: torch.LongTensor, fanout: int = INF, timestamp: list[int] = None):
        if self.adj is None or self.is_target:
            return {}
        num_q = len(idx)
        # uni_idx, inv = torch.unique(idx, return_inverse=True)
        subadj: dict = self.adj[idx]# [uni_idx]
        subadj.pop("number")
        
        hastimestamp = timestamp is not None
        if hastimestamp:
            assert len(timestamp) == num_q
        ret = {}
        for key in subadj:
            if key.endswith(TIMESTAMPADJNAME):
                continue
            adj = subadj[key]
            if len(adj) == 0:
                continue
            if not isinstance(adj, torch.Tensor):
                assert isinstance(adj, list)
                adj = torch.nn.utils.rnn.pad_sequence(adj, batch_first=True, padding_value=-1)
            # adj = adj[inv]
            if len(adj.flatten()) == 0:
                continue
            assert adj.ndim == 2, f"{adj.shape}"
            assert adj.shape[0] == num_q
            rootnode = torch.arange(num_q).reshape(-1, 1).repeat(1, adj.shape[1])
            if hastimestamp:# and key + TIMESTAMPADJNAME in subadj:
                adjtimestamp = subadj[key + TIMESTAMPADJNAME]
                if not isinstance(adjtimestamp, torch.Tensor):
                    adjtimestamp = torch.nn.utils.rnn.pad_sequence(adjtimestamp, batch_first=True, padding_value=-1)
                # adjtimestamp = adjtimestamp[inv]
                assert adjtimestamp.ndim == 2
                assert adjtimestamp.shape[0] == num_q
                adj.masked_fill_(adjtimestamp>=timestamp.reshape(-1, 1), -1)
            
            if adj.shape[1] > fanout:
                rank_val = torch.rand_like(adj, dtype=torch.float)
                rank_val.masked_fill_(adj<0, -1000.)
                topk_ind = torch.topk(rank_val, fanout, dim=-1)[1]
                adj = torch.gather(adj, 1, topk_ind)
                rootnode = rootnode[:, :fanout]
            
            mask = adj >= 0
            rootnode, adj = rootnode[mask], adj[mask]
            if len(rootnode) == 0:
                continue
            ret[key] = torch.stack(
                (rootnode, adj), dim=0
            )
        return ret


def edgename2tail(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[-1]
    elif edgename.startswith("tail of "):
        return edgename.split(":")[0].removeprefix("tail of ")
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


def edgename2head(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[0].removeprefix("head of ")
    elif edgename.startswith("tail of "):
        return edgename.split(":")[-1]
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


class Graph:

    def __init__(self, path) -> None:
        self.path = path
        self._dfs = None  # lazy: {"meta", "feat": {(nt, depth): tensor}, "nameemb"}
        self._dfs_cache = {}  # bounded memo for online (idx, cutoff) batches
        with open(osp.join(path, "metanode.yaml")) as f:
            metanode = yaml.safe_load(f)
        with open(osp.join(path, "metaadj.yaml")) as f:
            metaadj = yaml.safe_load(f)
        for nodetype in metanode:
            metanode[nodetype].update(metaadj[nodetype])
        self.metanode = metanode
        self.edgenameemb = torch.load(
            osp.join(path, "edgenameemb.pt"), map_location="cpu", weights_only=True
        )
        self.featnameemb = torch.load(
            osp.join(path, "featnameemb.pt"), map_location="cpu", weights_only=True
        )
        self.nodes = {
            nodetype: Node(
                self.metanode[nodetype],
                hds.load_from_disk(osp.join(path, "node", nodetype, "feat")).with_format("torch"),
                (
                    hds.load_from_disk(
                        osp.join(path, "node", nodetype, "textemb")
                    ).with_format("torch")
                    if osp.exists(osp.join(path, "node", nodetype, "textemb"))
                    else None
                ),
                (
                    None
                    if len(
                        self.metanode[nodetype]["in"] + self.metanode[nodetype]["out"]
                    )
                    == 0
                    else hds.load_from_disk(
                        osp.join(path, "edge", nodetype, "adj")
                    ).with_format("torch")
                ),
            )
            for nodetype in self.metanode
        }

    DFS_FORMAT_VERSION = 2

    def load_dfs(self):
        """Lazily load precomputed DFS artifacts (see dataconverterdfs.py).

        Stored values are RAW (unnormalized, counts log1p'd); normalization
        is applied at read time so the stored and online-computed paths
        produce identical numbers.
        """
        if self._dfs is not None:
            return self._dfs
        dfsdir = osp.join(self.path, "dfs")
        metapath = osp.join(dfsdir, "metadfs.yaml")
        if not osp.exists(metapath):
            raise FileNotFoundError(
                f"DFS features requested but {metapath} not found. "
                f"Run: python dataconverterdfs.py {self.path}"
            )
        with open(metapath) as f:
            meta = yaml.safe_load(f) or {}
        version = meta.pop("__format__", 1)
        if version != self.DFS_FORMAT_VERSION:
            raise RuntimeError(
                f"{metapath} is DFS artifact format v{version}, but this code "
                f"requires v{self.DFS_FORMAT_VERSION}. v1 artifacts store "
                f"z-normalized values computed at each row's own timestamp, "
                f"which aggregates ALL history (including the test window) "
                f"for non-temporal node types. Regenerate:\n"
                f"    python dataconverterdfs.py {self.path} "
                f"--cutoff_stats --tasks ALLTASK"
            )
        feat = {}
        for nt in meta:
            for depth in ("d1", "d2"):
                p = osp.join(dfsdir, nt, f"{depth}.pt")
                if depth in meta[nt] and osp.exists(p):
                    feat[(nt, depth)] = torch.load(
                        p, map_location="cpu", weights_only=True
                    )
        nameemb = torch.load(
            osp.join(dfsdir, "dfsfeatnameemb.pt"),
            map_location="cpu", weights_only=True,
        )
        self._dfs = {"meta": meta, "feat": feat, "nameemb": nameemb,
                     "layouts": {}, "plans": {}}
        return self._dfs

    def _dfs_layout(self, nodetype: str):
        """Rebuild a node type's d1 column layout and verify it still matches
        the stored artifacts. A silent mismatch would rebind checkpoint
        columns to different features."""
        dfs = self.load_dfs()
        if nodetype in dfs["layouts"]:
            return dfs["layouts"][nodetype]
        import dfscore
        meta = dfs["meta"][nodetype]
        a = meta.get("args", {})
        layout = dfscore.build_d1_layout(
            self, nodetype,
            a.get("max_cols_per_rel", 8),
            tuple(a.get("prims", dfscore.DEFAULT_PRIMS)),
            a.get("include_target_rels", False),
        )
        stored = meta["d1"]["names"]
        if layout["names"] != stored:
            raise RuntimeError(
                f"DFS layout for {nodetype} no longer matches its artifacts "
                f"({len(layout['names'])} columns recomputed vs "
                f"{len(stored)} stored). The dataset or dfscore changed since "
                f"the artifacts were written; regenerate them."
            )
        dfs["layouts"][nodetype] = layout
        return layout

    def _dfs_plan(self, nodetype: str):
        dfs = self.load_dfs()
        if nodetype in dfs["plans"]:
            return dfs["plans"][nodetype]
        meta = dfs["meta"][nodetype]
        plan = []
        if "d2" in meta:
            for (r, j), name in zip(meta["d2"]["plan"], meta["d2"]["names"]):
                tail_layout = self._dfs_layout(edgename2tail(r))
                plan.append((r, int(j), name, tail_layout["names"][int(j)]))
        dfs["plans"][nodetype] = plan
        return plan

    def _dfs_raw(self, nodetype: str, depth: str):
        return self.load_dfs()["feat"].get((nodetype, depth))

    def dfs_is_temporal(self, nodetype: str) -> bool:
        """Whether this node type has real per-row timestamps. Used by the
        fewshot loader to decide a leaf's DFS cutoff (temporal type -> the
        leaf's own timestamp; non-temporal -> the seed's task cutoff).
        Artifact-independent: derived from the node table itself."""
        import dfscore
        return dfscore.is_temporal(self, nodetype)

    def _dfs_own_timestamp(self, nodetype: str, idx: torch.Tensor):
        ts = self.nodes[nodetype].feat["timestamp"]
        if not torch.is_tensor(ts):
            ts = torch.tensor(ts)
        return ts[idx]

    def _dfs_values(self, nodetype: str, idx: torch.Tensor, depths, cutoff):
        """Raw (unnormalized) DFS columns for `idx` at `cutoff`, plus the
        normalization stats that go with them.

        Resolution:
          1. cutoff is None, or cutoff equals every row's own timestamp
             -> read the precomputed own-timestamp store. The equality case
                covers Completion pretraining and every task whose cutoff IS
                the row's timestamp (amazon-rating, seznam-*, retailrocket,
                stackexchange-upvote, ...), which is also where the number of
                distinct cutoffs makes precomputation hopeless.
          2. otherwise -> compute online at the given cutoff. Exact for any
             cutoff, and the only correct answer for a non-temporal root type,
             whose stored value spans all history.
        """
        import dfscore
        dfs = self.load_dfs()
        meta = dfs["meta"][nodetype]

        use_store = cutoff is None
        if not use_store:
            if not torch.is_tensor(cutoff):
                cutoff = torch.as_tensor(cutoff, dtype=torch.int64)
            cutoff = cutoff.to(torch.int64)
            use_store = bool(
                meta.get("temporal", True)
                and torch.equal(cutoff, self._dfs_own_timestamp(nodetype, idx))
            )
            if not use_store:
                sentinel = cutoff == dfscore.INT64_MIN
                if sentinel.all():
                    # Own timestamps of a non-temporal type (Completion-style
                    # "no query time exists"). Computing online at INT64_MIN
                    # would filter out every neighbor and silently return
                    # all-zero features; the all-time store is the intended
                    # semantics here.
                    cutoff = None
                    use_store = True
                elif sentinel.any():
                    raise RuntimeError(
                        f"{nodetype}: {int(sentinel.sum())}/{len(cutoff)} "
                        f"cutoffs are INT64_MIN mixed with real timestamps. "
                        f"A task cutoff vector must not contain the "
                        f"missing-timestamp sentinel."
                    )

        out = []
        if use_store:
            for d in depths:
                raw = self._dfs_raw(nodetype, d)
                if raw is None:
                    continue
                out.append((d, raw[idx], meta[d]["mean"], meta[d]["std"]))
            return out

        # ── online path ──
        for d in depths:
            if d not in meta:
                continue
            stats = meta.get(f"{d}_at")
            if stats is None:
                raise RuntimeError(
                    f"DFS features for {nodetype} were requested at a task "
                    f"cutoff, but its artifacts carry no cutoff-mode "
                    f"normalization stats ('{d}_at' in metadfs.yaml).\n"
                    f"Reading the own-timestamp store instead would aggregate "
                    f"all history"
                    + ("" if meta.get("temporal", True) else
                       f" — {nodetype} is non-temporal, so that store has no "
                       f"cutoff at all and leaks the evaluation window")
                    + f".\nRegenerate:\n    python dataconverterdfs.py "
                    f"{self.path} --cutoff_stats --tasks ALLTASK"
                )
            out.append((d, self._dfs_compute(nodetype, idx, cutoff, d),
                        stats["mean"], stats["std"]))
        return out

    def _dfs_compute(self, nodetype, idx, cutoff, depth):
        """Online DFS for exactly the (row, cutoff) pairs asked for.

        Deduplicates first: a subgraph batch reaches the same root-type row
        from several seeds, and hop-2 expansion duplicates it further.
        """
        import dfscore
        # exact bytes, not hash(): a hash collision would silently return
        # another batch's features
        key = (nodetype, depth,
               idx.numpy().tobytes(), cutoff.numpy().tobytes())
        cached = self._dfs_cache.get(key)
        if cached is not None:
            return cached

        pairs, inv = torch.unique(
            torch.stack((idx.to(torch.int64), cutoff), dim=1),
            dim=0, return_inverse=True,
        )
        u_idx, u_cut = pairs[:, 0].contiguous(), pairs[:, 1].contiguous()
        colsource = dfscore.RowGather(self)
        layout = self._dfs_layout(nodetype)
        if depth == "d1":
            vals = dfscore.compute_d1_at(self, nodetype, u_idx, u_cut, layout,
                                         colsource)
        else:
            meta = self.load_dfs()["meta"][nodetype]
            layouts = {nodetype: layout}
            for r, _, _, _ in self._dfs_plan(nodetype):
                t = edgename2tail(r)
                if t not in layouts:
                    layouts[t] = self._dfs_layout(t)
            vals = dfscore.compute_d2_at(
                self, nodetype, u_idx, u_cut, self._dfs_plan(nodetype),
                layouts, lambda t: self._dfs_raw(t, "d1"), colsource,
                nontemporal_hubs=meta.get("args", {}).get(
                    "d2_nontemporal_hubs", "recompute"),
            )
        vals = vals[inv]
        if len(self._dfs_cache) >= 32:
            self._dfs_cache.pop(next(iter(self._dfs_cache)))
        self._dfs_cache[key] = vals
        return vals

    def getdfsfeat(self, nodetype: str, idx: torch.Tensor, depth: int, floatemb,
                   cutoff=None):
        """Embedded DFS columns for `idx` rows of `nodetype`, evaluated at
        `cutoff`.

        depth is cumulative: 1 -> d1 columns; 2 -> d1 + d2 columns.
        cutoff: (B,) int64 query times, one per row — the task timestamp of
            the seed each row was reached from. None means "no query time
            exists", which falls back to the row's own timestamp.
        Returns (feat (B, C_dfs, D), nameemb (C_dfs, D)) or (None, None)
        if this node type has no DFS features (e.g. no relations).
        """
        import dfscore
        dfs = self.load_dfs()
        if nodetype not in dfs["meta"]:
            return None, None
        depths = ["d1"] if depth == 1 else ["d1", "d2"]
        cols, names = [], []
        for d, raw, mean, std in self._dfs_values(nodetype, idx, depths, cutoff):
            vals = dfscore.apply_norm(raw, mean, std)  # (B, C_d) z-normed
            for j, name in enumerate(dfs["meta"][nodetype][d]["names"]):
                cols.append(floatemb(vals[:, j].contiguous()))  # (B, D)
                names.append(name)
        if not cols:
            return None, None
        feat = torch.stack(cols, dim=1)  # (B, C_dfs, D)
        nameemb = torch.stack([dfs["nameemb"][n] for n in names], dim=0)
        return feat, nameemb

    def subgraph(
        self,
        root_nodetype: str,
        root_nodeidx: Iterable[int],
        hop: int,
        floatemb,
        fanout: int = INF,
        fanout_decay: float = 1.0,
        dfs_depth: int = 0,
        timestamp: Union[list[int], None] = None,
    ):
        """Sample a subgraph rooted at ``root_nodeidx``.

        Args:
            fanout: maximum neighbours per (source-node, edge-type) at hop 0.
                Set to ``INF`` for no cap (original behaviour).
            fanout_decay: geometric per-hop shrink factor. Hop ``h`` uses
                ``max(1, ceil(fanout * fanout_decay ** h))`` neighbours.
                Default ``1.0`` -> identical to a constant fanout.
                Ignored when ``fanout >= INF``.
            dfs_depth: 0 (default) = off. 1 or 2 appends DFS aggregate
                columns (see dataconverterdfs.py) to the ROOT node type's
                features and column-name embeddings. Depth is cumulative
                (2 = d1 + d2 columns). The columns are evaluated at
                ``timestamp`` — the same query time this call samples the
                MPNN neighborhood at — so the exact DFS summary and the
                sampled subgraph describe the same temporal window. Every
                root-type row carries the cutoff of the seed it was reached
                from, including rows found at deeper hops.
        """
        assert fanout_decay > 0, (
            f"fanout_decay must be > 0, got {fanout_decay}"
        )
        hastimestamp: bool = timestamp is not None
        adj = {}
        node = {}
        nodecutoff = {}
        root = {root_nodetype: root_nodeidx}
        roottimestamp = {root_nodetype: timestamp}

        def dictgetlen(d: dict[str, torch.Tensor], key: str, dim: int = 0):
            if key not in d:
                return 0
            return d[key].shape[dim]

        def dictupdate(
            d: dict[str, torch.Tensor],
            key: str,
            value: torch.Tensor,
            concatdim: int = 0,
        ):
            if key not in d:
                d[key] = value
            else:
                d[key] = torch.concat((d[key], value), dim=concatdim)
            return d

        for h in range(hop):
            # Per-hop decaying fanout. Ring 0 uses the full `fanout`;
            # each outer ring shrinks by `fanout_decay` (ceil, floored at 1).
            # decay == 1.0 -> identical to a constant fanout.
            # INF (no cap) stays uncapped.
            if fanout >= INF:
                hop_fanout = fanout
            else:
                hop_fanout = max(1, math.ceil(fanout * (fanout_decay ** h)))
            nroot, nroottimestamp = {}, {}
            for nodetype in root:
                found_src_num = dictgetlen(node, nodetype)
                # print(list(self.nodes.keys()), list(root.keys()), list(roottimestamp.keys()))
                ttadj = self.nodes[nodetype].getedge(
                    root[nodetype], hop_fanout, roottimestamp[nodetype]
                )
                for edgetype in ttadj:
                    srcidx, taridx = ttadj[edgetype][0], ttadj[edgetype][1]

                    tartype = edgename2tail(edgetype)
                    found_tar_num = (
                        dictgetlen(node, tartype)
                        + dictgetlen(root, tartype)
                        + dictgetlen(nroot, tartype)
                    )

                    if hastimestamp:
                        tartimestamp = roottimestamp[nodetype][srcidx]
                        nroottimestamp = dictupdate(
                            nroottimestamp, tartype, tartimestamp
                        )
                    else:
                        nroottimestamp[tartype] = None
                    nroot = dictupdate(nroot, tartype, taridx)

                    srcidx = srcidx + found_src_num
                    taridx = (
                        torch.arange(taridx.shape[0], device=taridx.device)
                        + found_tar_num
                    )

                    adj = dictupdate(
                        adj, edgetype, torch.stack((srcidx, taridx), dim=0), concatdim=1
                    )

            for nodetype in root:
                node = dictupdate(node, nodetype, root[nodetype])
                if hastimestamp and roottimestamp.get(nodetype) is not None:
                    nodecutoff = dictupdate(
                        nodecutoff, nodetype, roottimestamp[nodetype]
                    )
            del root
            del roottimestamp
            root = nroot
            roottimestamp = nroottimestamp

        for nodetype in root:
            node = dictupdate(node, nodetype, root[nodetype])
            if hastimestamp and roottimestamp.get(nodetype) is not None:
                nodecutoff = dictupdate(
                    nodecutoff, nodetype, roottimestamp[nodetype]
                )

        mapping = torch.arange(len(root_nodeidx), device=root_nodeidx.device)
        # not change, latter code depends on mapping == arange

        dfs_root_idx = None
        for nodetype in node:
            assert len(node[nodetype]), f"{list(node.keys())} {root_nodeidx.shape} {list(adj.keys())}"
            if dfs_depth > 0 and nodetype == root_nodetype:
                dfs_root_idx = node[nodetype]  # raw indices, pre-replacement
            #unique_idx, inv = torch.unique(node[nodetype], return_inverse=True)
            #node[nodetype] = self.nodes[nodetype].getfeat(unique_idx, floatemb)[inv]
            node[nodetype] = self.nodes[nodetype].getfeat(node[nodetype], floatemb)

        edgenameemb = {edgetype: self.edgenameemb[edgetype] for edgetype in adj}
        nodenameemb = {
            nodetype: torch.stack(
                [self.featnameemb[_] for _ in self.nodes[nodetype].featlist], dim=0
            )
            for nodetype in node
        }

        # Append precomputed DFS aggregate columns to the root node type.
        # Column attention is shape-agnostic, so downstream code (masks,
        # scalefeat, the model) adapts automatically; loaders pad
        # target_feat_mask for the extra columns.
        if dfs_depth > 0 and dfs_root_idx is not None:
            dfs_cutoff = nodecutoff.get(root_nodetype) if hastimestamp else None
            if dfs_cutoff is not None:
                assert dfs_cutoff.shape[0] == dfs_root_idx.shape[0], (
                    f"cutoff/{root_nodetype} row misalignment: "
                    f"{dfs_cutoff.shape[0]} vs {dfs_root_idx.shape[0]}"
                )
            dfsfeat, dfsnameemb = self.getdfsfeat(
                root_nodetype, dfs_root_idx, dfs_depth, floatemb,
                cutoff=dfs_cutoff,
            )
            if dfsfeat is not None:
                node[root_nodetype] = torch.concat(
                    (node[root_nodetype], dfsfeat), dim=1
                )
                nodenameemb[root_nodetype] = torch.concat(
                    (nodenameemb[root_nodetype], dfsnameemb), dim=0
                )
        # print("loader: ", len(node), len(adj), list(nodenameemb.keys()), list(edgenameemb.keys()))

        return node, adj, nodenameemb, edgenameemb, mapping
    
    def fewshot(
        self,
        root_nodetype: str,
        root_nodeidx: torch.LongTensor,
        task_mask: torch.BoolTensor,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
        prefetch_factor: int = 10,
    ):
        assert fanout < INF, "alway sample past"
        # assert node idx in data is sorted by time stamp
        hastimestamp: bool = timestamp is not None

        idx = torch.randint(0, MAXINT64, (root_nodeidx.shape[0], prefetch_factor*fanout))
        idx = idx % (root_nodeidx.clamp_min(1)).reshape(-1, 1)
        
        if prefetch_factor > 1:
            rootfeat = self.nodes[root_nodetype].getfeat(root_nodeidx, floatemb) # (B, C, D)
            rootfeat[task_mask.unsqueeze(-1).expand(-1, -1, rootfeat.shape[-1])] = 0.
            feat = self.nodes[root_nodetype].getfeat(idx.flatten(), floatemb).unflatten(0, (-1, prefetch_factor*fanout)) # （B, FANOUT*PREFETCH, C, D)
            score = feat.flatten(-2, -1) @ rootfeat.flatten(-2, -1).unsqueeze(-1) # (B, FANOUT*PREFETCH, 1)
            score = score.squeeze(-1) # (B, FANOUT*PREFETCH)
            idx = torch.gather(idx, 1, torch.topk(score, k=fanout, dim=-1)[1]).flatten() # (B, FANOUT)
        idx = idx.flatten()
        rootnode = torch.arange(root_nodeidx.shape[0]).repeat_interleave(fanout)
        mask = root_nodeidx[rootnode] > 0
        idx, rootnode = idx[mask], rootnode[mask]
        return idx, rootnode


class Task:
    def __init__(self, path) -> None:
        with open(osp.join(path, "metatask.yaml")) as f:
            self.metatask = yaml.safe_load(f)

        self.tasknameemb = torch.load(
            osp.join(path, "tasknameemb.pt"), map_location="cpu", weights_only=True
        )
        self.tasks = {
            taskname: hds.load_from_disk(osp.join(path, "task", taskname)).with_format(
                "torch"
            )
            for taskname in self.metatask
        }

    def get_retrieval(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "retrieval"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (meta["seed_type"], meta["num_class"]),
        )

    def get_regression(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "regression"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (None, None),
        )
