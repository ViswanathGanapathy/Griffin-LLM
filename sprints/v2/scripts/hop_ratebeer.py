"""1/2/3-hop neighbor counts for rel-ratebeer, seeded at `users`.

Reads only pkey/fkey columns from the cached parquet files (11.8M-row
beer_ratings on a 15GB machine) and uses boolean visited masks.
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

DB = os.path.expanduser("~/.cache/relbench/rel-ratebeer/db")

# table -> (pkey or None, {fk_col: parent_table})
SCHEMA = {
    "brewers":       ("brewer_id",   {"country_id": "countries", "state_id": "states", "type_id": "place_types"}),
    "beer_styles":   ("style_id",    {}),
    "place_types":   ("type_id",     {}),
    "place_ratings": ("rating_id",   {"place_id": "places", "user_id": "users"}),
    "users":         ("user_id",     {}),
    "places":        ("place_id",    {"state_id": "states", "type_id": "place_types", "country_id": "countries"}),
    "availability":  ("avail_id",    {"beer_id": "beers", "place_id": "places", "country_id": "countries", "user_id": "users"}),
    "countries":     ("country_id",  {}),
    "beers":         ("beer_id",     {"brewer_id": "brewers", "style_id": "beer_styles"}),
    "beer_upcs":     (None,          {"beer_id": "beers"}),
    "states":        ("state_id",    {"country_id": "countries"}),
    "beer_ratings":  ("rating_id",   {"user_id": "users", "beer_id": "beers", "availability_id": "availability"}),
    "favorites":     ("favorite_id", {"user_id": "users", "beer_id": "beers"}),
}

seed_table = sys.argv[1] if len(sys.argv) > 1 else "users"
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 200

nrows, pk_pos, cols = {}, {}, {}
for t, (pk, fks) in SCHEMA.items():
    want = ([pk] if pk else []) + list(fks)
    df = pd.read_parquet(f"{DB}/{t}.parquet", columns=want)
    nrows[t] = len(df)
    if pk:
        pk_pos[t] = pd.Series(np.arange(len(df)), index=df[pk].values)
    cols[t] = df
    print(f"loaded {t}: {len(df)} rows")

fwd, rev = {}, {}
for t, (pk, fks) in SCHEMA.items():
    for fk_col, parent in fks.items():
        mapped = pk_pos[parent].reindex(cols[t][fk_col].values).values
        arr = np.where(np.isnan(mapped), -1, mapped).astype(np.int64)
        fwd[(t, fk_col)] = (parent, arr)
        valid = arr >= 0
        order = np.argsort(arr[valid], kind="stable")
        child_pos = np.flatnonzero(valid)[order].astype(np.int64)
        parent_sorted = arr[valid][order]
        starts = np.searchsorted(parent_sorted, np.arange(nrows[parent]))
        ends = np.searchsorted(parent_sorted, np.arange(nrows[parent]), side="right")
        rev[(t, fk_col)] = (parent, child_pos, starts, ends)
for t in cols:
    cols[t] = None  # free the dataframes; only index arrays needed now

def hops(seed_row, max_hop=3):
    visited = {t: np.zeros(n, dtype=bool) for t, n in nrows.items()}
    visited[seed_table][seed_row] = True
    frontier = {seed_table: np.array([seed_row], dtype=np.int64)}
    counts, breakdowns = [], []
    for _ in range(max_hop):
        nb = defaultdict(list)
        for tname, rows in frontier.items():
            for fk_col, parent in SCHEMA[tname][1].items():
                _, arr = fwd[(tname, fk_col)]
                v = arr[rows]
                nb[parent].append(v[v >= 0])
            for (cname, fk_col), (parent, child_pos, starts, ends) in rev.items():
                if parent != tname:
                    continue
                chunks = [child_pos[starts[r]:ends[r]] for r in rows]
                if chunks:
                    nb[cname].append(np.concatenate(chunks) if len(chunks) > 1 else chunks[0])
        new = {}
        for t, parts in nb.items():
            u = np.unique(np.concatenate(parts))
            fresh = u[~visited[t][u]]
            if len(fresh):
                visited[t][fresh] = True
                new[t] = fresh
        counts.append(sum(len(v) for v in new.values()))
        breakdowns.append({t: len(v) for t, v in sorted(new.items())})
        frontier = new
        if not frontier:
            break
    while len(counts) < max_hop:
        counts.append(0); breakdowns.append({})
    return counts, breakdowns

print(f"\nseed_table={seed_table} rows={nrows[seed_table]}")
c, b = hops(0)
print(f"example row #0:")
for i in range(3):
    print(f"  hop{i+1}: {c[i]}  {b[i]}")

rng = np.random.default_rng(0)
samp = rng.choice(nrows[seed_table], n_sample, replace=False)
res = np.array([hops(int(s))[0] for s in samp])
print(f"\nsample of {n_sample} rows of {seed_table}:")
for i in range(3):
    col = res[:, i]
    print("  hop%d: mean %.0f median %d p90 %d max %d  (frac with 0: %.2f)" %
          (i + 1, col.mean(), np.median(col), np.percentile(col, 90), col.max(),
           (col == 0).mean()))

med_i = int(samp[np.argsort(np.abs(res[:, 2] - np.median(res[:, 2])))[0]])
c2, b2 = hops(med_i)
print(f"\nmedian-like row #{med_i}:")
for i in range(3):
    print(f"  hop{i+1}: {c2[i]}  {b2[i]}")
