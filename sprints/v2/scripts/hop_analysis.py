"""Generic 1/2/3-hop neighbor counts for a RelBench v2 dataset.

Usage: python hop_analysis.py <dataset> <seed_table> [n_sample] [example_idx]
Nodes are (table, row); edges are FK links, traversed both directions.
"""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from relbench.datasets import get_dataset

ds_name, seed_table = sys.argv[1], sys.argv[2]
n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 500
example_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0

db = get_dataset(ds_name).get_db()
tables = db.table_dict

# ---- build positional FK arrays and reverse indices ----
pk_pos = {}  # table -> {pkey_value: row_pos}
for name, t in tables.items():
    if t.pkey_col is not None:
        pk_pos[name] = pd.Series(np.arange(len(t.df)), index=t.df[t.pkey_col].values)

fwd = {}   # (child, fk_col) -> (parent, np.array child_pos -> parent_pos or -1)
rev = {}   # (child, fk_col) -> (parent_pos-sorted child order, starts, ends)
for cname, t in tables.items():
    for fk_col, pname in t.fkey_col_to_pkey_table.items():
        mapped = pk_pos[pname].reindex(t.df[fk_col].values).values
        arr = np.where(np.isnan(mapped), -1, mapped).astype(np.int64)
        fwd[(cname, fk_col)] = (pname, arr)
        valid = arr >= 0
        order = np.argsort(arr[valid], kind="stable")
        child_pos = np.flatnonzero(valid)[order]
        parent_sorted = arr[valid][order]
        npar = len(tables[pname].df)
        starts = np.searchsorted(parent_sorted, np.arange(npar))
        ends = np.searchsorted(parent_sorted, np.arange(npar), side="right")
        rev[(cname, fk_col)] = (pname, child_pos, starts, ends)

def expand(frontier):
    """frontier: dict table -> np.array of rows. Returns dict of neighbor sets."""
    out = defaultdict(list)
    for tname, rows in frontier.items():
        # child -> parent
        for fk_col, pname in tables[tname].fkey_col_to_pkey_table.items():
            _, arr = fwd[(tname, fk_col)]
            v = arr[rows]
            out[pname].append(v[v >= 0])
        # parent -> children
        for (cname, fk_col), (pname, child_pos, starts, ends) in rev.items():
            if pname != tname:
                continue
            chunks = [child_pos[starts[r]:ends[r]] for r in rows]
            if chunks:
                out[cname].append(np.concatenate(chunks) if len(chunks) > 1 else chunks[0])
    return {k: np.unique(np.concatenate(v)) for k, v in out.items() if v}

def hops(seed_row, detail=False):
    visited = {t: set() for t in tables}
    visited[seed_table].add(seed_row)
    frontier = {seed_table: np.array([seed_row])}
    counts, breakdowns = [], []
    for _ in range(3):
        nb = expand(frontier)
        new = {}
        for t, rows in nb.items():
            fresh = np.array([r for r in rows if r not in visited[t]], dtype=np.int64)
            if len(fresh):
                new[t] = fresh
                visited[t].update(fresh.tolist())
        counts.append(sum(len(v) for v in new.values()))
        breakdowns.append({t: len(v) for t, v in new.items()})
        frontier = new
        if not frontier:
            break
    while len(counts) < 3:
        counts.append(0); breakdowns.append({})
    return counts, breakdowns

n_seed = len(tables[seed_table].df)
print(f"dataset={ds_name} seed_table={seed_table} rows={n_seed}")
for name, t in tables.items():
    print(f"  {name}: {len(t.df)} rows, fkeys={t.fkey_col_to_pkey_table}")

c, b = hops(example_idx, detail=True)
print(f"\nexample row #{example_idx} of {seed_table}:")
for i in range(3):
    print(f"  hop{i+1}: {c[i]}  {b[i]}")

rng = np.random.default_rng(0)
samp = rng.choice(n_seed, min(n_sample, n_seed), replace=False)
res = np.array([hops(int(s))[0] for s in samp])
print(f"\nsample of {len(samp)} rows of {seed_table}:")
for i in range(3):
    col = res[:, i]
    print("  hop%d: mean %.0f median %d p90 %d max %d" %
          (i + 1, col.mean(), np.median(col), np.percentile(col, 90), col.max()))

# median-ish example: row whose hop3 is closest to the sample median
med_i = int(samp[np.argsort(np.abs(res[:, 2] - np.median(res[:, 2])))[0]])
c2, b2 = hops(med_i)
print(f"\nmedian-like row #{med_i}: hops={c2}")
for i in range(3):
    print(f"  hop{i+1}: {c2[i]}  {b2[i]}")
