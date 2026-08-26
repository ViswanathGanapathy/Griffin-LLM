"""Dump per-seed hop1/2/3 counts for rel-salt (salesdocument) and rel-ratebeer (users)."""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

CACHE = os.path.expanduser("~/.cache/relbench")

SCHEMAS = {
    "rel-salt": {
        "customer":         ("CUSTOMER",      {"ADDRESSID": "address"}),
        "salesdocument":    ("SALESDOCUMENT", {}),
        "address":          ("ADDRESSID",     {}),
        "salesdocumentitem":("ID",            {"SALESDOCUMENT": "salesdocument", "SOLDTOPARTY": "customer",
                                               "SHIPTOPARTY": "customer", "BILLTOPARTY": "customer",
                                               "PAYERPARTY": "customer"}),
    },
    "rel-ratebeer": {
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
    },
}

def build(ds_name):
    schema = SCHEMAS[ds_name]
    nrows, pk_pos = {}, {}
    raw = {}
    for t, (pk, fks) in schema.items():
        want = ([pk] if pk else []) + list(fks)
        df = pd.read_parquet(f"{CACHE}/{ds_name}/db/{t}.parquet", columns=want)
        nrows[t] = len(df)
        if pk:
            pk_pos[t] = pd.Series(np.arange(len(df)), index=df[pk].values)
        raw[t] = df
    fwd, rev = {}, {}
    for t, (pk, fks) in schema.items():
        for fk_col, parent in fks.items():
            mapped = pk_pos[parent].reindex(raw[t][fk_col].values).values
            arr = np.where(np.isnan(mapped), -1, mapped).astype(np.int64)
            fwd[(t, fk_col)] = (parent, arr)
            valid = arr >= 0
            order = np.argsort(arr[valid], kind="stable")
            child_pos = np.flatnonzero(valid)[order].astype(np.int64)
            ps = arr[valid][order]
            starts = np.searchsorted(ps, np.arange(nrows[parent]))
            ends = np.searchsorted(ps, np.arange(nrows[parent]), side="right")
            rev[(t, fk_col)] = (parent, child_pos, starts, ends)
    return schema, nrows, fwd, rev

def hops(schema, nrows, fwd, rev, seed_table, seed_row):
    visited = {t: np.zeros(n, dtype=bool) for t, n in nrows.items()}
    visited[seed_table][seed_row] = True
    frontier = {seed_table: np.array([seed_row], dtype=np.int64)}
    counts = []
    for _ in range(3):
        nb = defaultdict(list)
        for tname, rows in frontier.items():
            for fk_col, parent in schema[tname][1].items():
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
        frontier = new
        if not frontier:
            break
    while len(counts) < 3:
        counts.append(0)
    return counts

out = {}
for ds_name, seed_table, n_samp in [("rel-salt", "salesdocument", 4000),
                                    ("rel-ratebeer", "users", 1500)]:
    schema, nrows, fwd, rev = build(ds_name)
    rng = np.random.default_rng(0)
    samp = rng.choice(nrows[seed_table], n_samp, replace=False)
    res = np.array([hops(schema, nrows, fwd, rev, seed_table, int(s)) for s in samp])
    out[ds_name.replace("rel-", "")] = res
    print(ds_name, "done", res.shape)

np.savez(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'hops_relbench.npz'), **out)
print("saved")
