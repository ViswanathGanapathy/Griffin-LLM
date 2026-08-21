"""Bridge: RelBench (v2) dataset -> Griffin raw-source format.

Exports a RelBench dataset (e.g. ``rel-ratebeer``) into the raw src
layout that Griffin's four-stage ETL consumes
(dataconverter.py -> dataconverteredge.py -> dataconvertertask.py ->
dataconverterpost.py), i.e. a ``metadata.yaml`` + per-column ``.npy``
arrays. See sprints/v2/RATEBEER_PLAN.md for the full recipe.

    pip install relbench
    python build_relbench_src.py rel-ratebeer griffin_src/rel-ratebeer \
        [--tasks beer-churn ...] [--skip_text] [--dry_run]

Status: SCAFFOLD, validated for structure but not yet against a live
rel-ratebeer download (schema names are read dynamically, so it should
adapt; run --dry_run first and eyeball the printed schema).

Mapping decisions (mirrors how joint-v65 was built):
  * table -> node type "<prefix>-<TableName>"; row order preserved so
    RelBench row index == Griffin node index.
  * numeric column -> float32 .npy (NaN -> 0.0 with a paired
    "<col>_isnan" indicator column when >1% NaN).
  * low-cardinality (<=200) object/categorical column -> int64 codes.
  * high-cardinality text column -> Nomic-embedded matrix under the
    "Griffin_text_<col>" convention (skipped with --skip_text; strongly
    recommended to start with --skip_text and add text later — ratebeer
    review text is millions of rows).
  * time_col -> "__timestamp__" int64 epoch-seconds array (tables
    without a time_col get zeros; dataconverter.py maps all-zero to
    -inf = "always visible"). Rows of temporal tables MUST be sorted by
    time for Griffin's fewshot past-only sampler — we sort and persist
    the permutation, remapping fkeys accordingly.
  * fkey (child.fkey_col -> parent.pkey) -> edge array (2, E) with type
    string "<child>:<child>-<fkey_col>:<parent>".
  * classification task -> joint-v65 "retrieval" construction: a
    synthetic label-holder node type "<target>_<taskcol>" with
    num_classes rows (is_target semantics), label edges from entity
    rows (timestamped at the task row's timestamp), plus
    train/val/test node_pairs + timestamp arrays.
  * regression task -> seed_nodes + labels + timestamp arrays.
  * name_emb / task_emb -> Nomic (sentence-transformers); CPU works for
    the few hundred names involved.
"""
import argparse
import os
import os.path as osp

import numpy as np
import yaml

TEXT_CARDINALITY_MIN = 200          # above this an object column is "text"
NAN_INDICATOR_THRESHOLD = 0.01


def get_name_embedder(dim=512):
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        cache_folder="cache_data/model", trust_remote_code=True,
        truncate_dim=dim,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    def enc(text: str):
        e = model.encode(text, prompt="clustering: ", convert_to_numpy=True)
        return (e / np.linalg.norm(e)).tolist()
    return enc


def epoch_seconds(series):
    import pandas as pd
    s = pd.to_datetime(series, errors="coerce")
    out = (s.astype("int64") // 10**9).to_numpy()
    out[s.isna().to_numpy()] = 0
    return out.astype(np.int64)


def export_table(name, df, time_col, dstdir, prefix, skip_text, embed_name):
    """Write one table's columns; returns (feature_meta_entries, sort_perm)."""
    ntype = f"{prefix}-{name}"
    feats = []

    # sort by time so Griffin's past-only fewshot sampling is valid
    perm = None
    ts = np.zeros(len(df), dtype=np.int64)
    if time_col is not None and time_col in df.columns:
        ts = epoch_seconds(df[time_col])
        perm = np.argsort(ts, kind="stable")
        df = df.iloc[perm].reset_index(drop=True)
        ts = ts[perm]
    np.save(osp.join(dstdir, f"features/{ntype}___timestamp__.npy"), ts)

    for col in df.columns:
        if col == time_col:
            continue
        s = df[col]
        cname, path = None, None
        if s.dtype.kind in "if":
            vals = s.to_numpy(dtype=np.float32)
            nanfrac = float(np.isnan(vals).mean())
            cname = col
            path = f"features/{ntype}_{col}.npy"
            np.save(osp.join(dstdir, path), np.nan_to_num(vals))
            feats.append({"name": cname, "path": path, "type": ntype,
                          "extra_fields": {"dtype": "float",
                                           "name_emb": embed_name(f"{name} {col}")}})
            if nanfrac > NAN_INDICATOR_THRESHOLD:
                ipath = f"features/{ntype}_{col}_isnan.npy"
                np.save(osp.join(dstdir, ipath),
                        np.isnan(s.to_numpy(dtype=np.float32)).astype(np.int64))
                feats.append({"name": f"{col}_isnan", "path": ipath, "type": ntype,
                              "extra_fields": {"dtype": "category", "num_categories": 2,
                                               "name_emb": embed_name(f"{name} {col} missing")}})
        elif s.dtype.kind in "OUb" or str(s.dtype) == "category":
            nunique = s.nunique(dropna=True)
            if nunique <= TEXT_CARDINALITY_MIN:
                codes = s.astype("category").cat.codes.to_numpy().astype(np.int64) + 1
                path = f"features/{ntype}_{col}.npy"
                np.save(osp.join(dstdir, path), codes)
                feats.append({"name": col, "path": path, "type": ntype,
                              "extra_fields": {"dtype": "category",
                                               "num_categories": int(nunique) + 1,
                                               "name_emb": embed_name(f"{name} {col}")}})
            elif not skip_text:
                raise NotImplementedError(
                    f"text column {name}.{col} ({nunique} uniques): text "
                    f"embedding export not implemented in the scaffold — "
                    f"run with --skip_text or extend export_table()."
                )
            # else: silently dropped under --skip_text
    return feats, perm, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="RelBench dataset name, e.g. rel-ratebeer")
    ap.add_argument("dstpath", help="output raw-src dir for Griffin ETL")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="RelBench task names to export (default: all)")
    ap.add_argument("--skip_text", action="store_true",
                    help="drop high-cardinality text columns (recommended v1)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print schema mapping, write nothing")
    args = ap.parse_args()

    from relbench.datasets import get_dataset
    from relbench.tasks import get_task_names, get_task

    prefix = args.dataset.replace("rel-", "")
    ds = get_dataset(args.dataset, download=True)
    db = ds.get_db()

    print(f"== {args.dataset}: {len(db.table_dict)} tables ==")
    for tname, table in db.table_dict.items():
        print(f"  {tname:24s} rows={len(table.df):>9}  pkey={table.pkey_col} "
              f"time={table.time_col}  fkeys={table.fkey_col_to_pkey_table}")
    tasknames = args.tasks or get_task_names(args.dataset)
    print(f"tasks: {tasknames}")
    if args.dry_run:
        return

    os.makedirs(osp.join(args.dstpath, "features"), exist_ok=True)
    os.makedirs(osp.join(args.dstpath, "edges"), exist_ok=True)
    embed_name = get_name_embedder()

    meta = {"dataset_name": f"{args.dataset}-griffin",
            "graph": {"nodes": [], "edges": [], "feature_data": []},
            "feature_data": [], "tasks": []}
    perms, pkey_index = {}, {}

    # ── tables ──
    for tname, table in db.table_dict.items():
        ntype = f"{prefix}-{tname}"
        feats, perm, ts = export_table(
            tname, table.df, table.time_col, args.dstpath, prefix,
            args.skip_text, embed_name)
        perms[tname] = perm
        # pkey value -> post-sort row index (fkeys reference pkey values)
        pk = table.df[table.pkey_col].to_numpy() if table.pkey_col else None
        if pk is not None and perm is not None:
            pk = pk[perm]
        pkey_index[tname] = ({v: i for i, v in enumerate(pk)}
                             if pk is not None else None)
        meta["graph"]["nodes"].append({"type": ntype, "num": len(table.df)})
        meta["graph"]["feature_data"].append({
            "domain": "node", "name": "__timestamp__", "type": ntype,
            "path": f"features/{ntype}___timestamp__.npy",
            "extra_fields": {}, "format": "numpy", "in_memory": False})
        meta["feature_data"].extend(
            {**f, "domain": "node", "format": "numpy", "in_memory": False}
            for f in feats)
        print(f"exported {ntype}: {len(feats)} feature columns")

    # ── fkey edges ──
    for tname, table in db.table_dict.items():
        child = f"{prefix}-{tname}"
        cdf = table.df
        if perms[tname] is not None:
            cdf = cdf.iloc[perms[tname]].reset_index(drop=True)
        for fkey_col, parent_t in table.fkey_col_to_pkey_table.items():
            parent = f"{prefix}-{parent_t}"
            idxmap = pkey_index[parent_t]
            child_rows, parent_rows = [], []
            for i, v in enumerate(cdf[fkey_col].to_numpy()):
                j = idxmap.get(v)
                if j is not None:
                    child_rows.append(i)
                    parent_rows.append(j)
            etype = f"{child}:{child}-{fkey_col}:{parent}"
            arr = np.stack([np.array(child_rows, dtype=np.int64),
                            np.array(parent_rows, dtype=np.int64)])
            path = f"edges/{etype.replace(':', '__')}_edges.npy"
            np.save(osp.join(args.dstpath, path), arr)
            meta["graph"]["edges"].append(
                {"type": etype, "path": path, "format": "numpy"})
            print(f"edge {etype}: E={arr.shape[1]}")

    # ── tasks ── (regression path; classification needs the synthetic
    # label-holder construction — see RATEBEER_PLAN.md §4 before using)
    for taskname in tasknames:
        task = get_task(args.dataset, taskname, download=True)
        ttype = f"{prefix}-{task.entity_table}"
        entity_map = pkey_index[task.entity_table]
        tdir = f"{taskname}"
        entry = {"extra_fields": {
            "name": f"{prefix}-{taskname}",
            "target_type": ttype, "seed_type": f"x:{ttype}",
            "task_type": "regression",  # TODO classification: see plan §4
            "evaluation_metric": "mae",
            "task_emb": embed_name(f"{args.dataset} {taskname}")}}
        for split, key in (("train", "train_set"), ("val", "validation_set"),
                           ("test", "test_set")):
            tab = task.get_table(split)
            ent = tab.df[task.entity_col].map(entity_map).to_numpy(dtype=np.int64)
            lab = tab.df[task.target_col].to_numpy(dtype=np.float32)
            tsv = epoch_seconds(tab.df[tab.time_col]) if tab.time_col else None
            os.makedirs(osp.join(args.dstpath, tdir, key), exist_ok=True)
            datalist = []
            for nm, arr in (("seed_nodes", ent), ("labels", lab),
                            ("timestamp", tsv)):
                if arr is None:
                    continue
                p = f"{tdir}/{key}/{nm}.npy"
                np.save(osp.join(args.dstpath, p), arr)
                datalist.append({"name": nm, "path": p, "format": "numpy",
                                 "in_memory": True})
            entry[key] = [{"data": datalist, "type": ttype}]
        meta["tasks"].append(entry)
        print(f"task {taskname}: exported (regression form)")

    with open(osp.join(args.dstpath, "metadata.yaml"), "w") as f:
        yaml.dump(meta, f)
    print(f"\nWrote {args.dstpath}/metadata.yaml — now run the 4-stage ETL "
          f"(see sprints/v2/RATEBEER_PLAN.md §5).")


if __name__ == "__main__":
    main()
