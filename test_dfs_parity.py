"""Numerical parity test: Griffin's DFS aggregation vs featuretools.

Validates that dataconverterdfs.py's aggregation core produces the same
numbers as featuretools' DFS on an equivalent synthetic 3-table RDB
(users ← orders ← items), including NaN handling and temporal cutoffs.

Two phases, so featuretools never touches the griffin env:

  Phase A (griffin env, torch):
      python test_dfs_parity.py --phase a --out /tmp/dfs_parity
    Builds the synthetic RDB (seeded), runs the ACTUAL aggregate_batch
    from dfscore.py with the strict-past rule, dumps
    ours.json + the raw tables as CSV.

  Phase B (venv with featuretools + pandas, NO torch):
      python test_dfs_parity.py --phase b --out /tmp/dfs_parity
    Loads the CSVs, runs featuretools dfs() with per-row cutoff times
    aligned to our strict-past rule (cutoff = ts - 1s, since
    featuretools' cutoff is inclusive while ours is strict), and
    compares Count / Mean / Max at depth 1 and Mean-of-Mean at depth 2.

Alignment notes (the deliberate semantic mapping):
  * cutoff: ours is strict (nb_ts < node_ts); featuretools includes
    rows at time <= cutoff_time -> we pass cutoff_time = node_ts - 1s.
  * NaN: both skip NaNs in Mean/Max; Count counts rows.
  * depth 2: compared on the NON-temporal edge (orders->items has no
    timestamps beyond the order's own), where our per-neighbor-snapshot
    semantics and featuretools' single-cutoff semantics coincide. The
    temporal d2 deviation is documented in DFS_GRIFFIN_DESIGN.md.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

SEED = 7
N_USERS, N_ORDERS, N_ITEMS = 12, 60, 200
NAN_FRAC = 0.25


def build_tables():
    rng = np.random.RandomState(SEED)
    users = {
        "user_id": np.arange(N_USERS),
        "signup_ts": np.sort(rng.randint(1_000, 500_000, N_USERS)).astype("int64"),
    }
    orders = {
        "order_id": np.arange(N_ORDERS),
        "user_id": rng.randint(0, N_USERS, N_ORDERS).astype("int64"),
        "order_ts": rng.randint(1_000, 1_000_000, N_ORDERS).astype("int64"),
        # negative-mean column with NaNs — exercises the max-with-NaN case
        "amount": rng.randn(N_ORDERS).astype("float64") - 3.0,
    }
    nanmask = rng.rand(N_ORDERS) < NAN_FRAC
    orders["amount"][nanmask] = np.nan
    items = {
        "item_id": np.arange(N_ITEMS),
        "order_id": rng.randint(0, N_ORDERS, N_ITEMS).astype("int64"),
        "price": np.abs(rng.randn(N_ITEMS)).astype("float64") * 20,
    }
    nanmask_i = rng.rand(N_ITEMS) < NAN_FRAC
    items["price"][nanmask_i] = np.nan
    return users, orders, items


def phase_a(outdir):
    import torch
    from dfscore import aggregate_batch

    users, orders, items = build_tables()
    os.makedirs(outdir, exist_ok=True)
    # persist tables for phase B
    import csv
    for name, tab in (("users", users), ("orders", orders), ("items", items)):
        with open(osp.join(outdir, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            keys = list(tab.keys())
            w.writerow(keys)
            for i in range(len(tab[keys[0]])):
                w.writerow([tab[k][i] for k in keys])

    # ── depth 1: users <- orders (temporal, strict past) ──
    src_list, tar_list = [], []
    for oid in range(N_ORDERS):
        u = orders["user_id"][oid]
        if orders["order_ts"][oid] < users["signup_ts"][u]:   # strict
            src_list.append(u)
            tar_list.append(oid)
    src = torch.tensor(src_list, dtype=torch.int64)
    tar = torch.tensor(tar_list, dtype=torch.int64)
    vals = {"amount": torch.tensor(orders["amount"])[tar]}
    d1_users = aggregate_batch(src, tar, vals, N_USERS)

    # ── depth 1: orders <- items (non-temporal) ──
    src_o = torch.tensor(items["order_id"], dtype=torch.int64)
    tar_i = torch.tensor(np.arange(N_ITEMS), dtype=torch.int64)
    vals_i = {"price": torch.tensor(items["price"])[tar_i]}
    d1_orders = aggregate_batch(src_o, tar_i, vals_i, N_ORDERS)

    # ── depth 2: users <- orders <- items, MEAN of order's MEAN(price)
    #    (same strict-past order set; d1 of orders is NaN-free) ──
    o_mean_price = d1_orders["mean__price"]
    n = torch.zeros(N_USERS).scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    s = torch.zeros(N_USERS).scatter_add_(0, src, o_mean_price[tar])
    d2_users_mean_mean = s / n.clamp_min(1.0)

    ours = {
        "d1_count": d1_users["count"].tolist(),
        "d1_mean_amount": d1_users["mean__amount"].tolist(),
        "d1_max_amount": d1_users["max__amount"].tolist(),
        "d2_mean_mean_price": d2_users_mean_mean.tolist(),
        "included_orders": src_list and [int(t) for t in tar_list] or [],
    }
    with open(osp.join(outdir, "ours.json"), "w") as f:
        json.dump(ours, f)
    print(f"Phase A done -> {outdir}/ours.json "
          f"({int(d1_users['count'].sum())} strict-past user-order pairs)")


def phase_b(outdir):
    import pandas as pd
    import featuretools as ft

    users = pd.read_csv(osp.join(outdir, "users.csv"))
    orders = pd.read_csv(osp.join(outdir, "orders.csv"))
    items = pd.read_csv(osp.join(outdir, "items.csv"))
    ours = json.load(open(osp.join(outdir, "ours.json")))

    # timestamps as datetimes for featuretools
    users["signup_time"] = pd.to_datetime(users["signup_ts"], unit="s")
    orders["order_time"] = pd.to_datetime(orders["order_ts"], unit="s")

    es = ft.EntitySet(id="parity")
    es = es.add_dataframe(dataframe_name="users", dataframe=users.drop(columns=["signup_ts"]),
                          index="user_id")
    es = es.add_dataframe(dataframe_name="orders",
                          dataframe=orders.drop(columns=["order_ts"]),
                          index="order_id", time_index="order_time")
    es = es.add_dataframe(dataframe_name="items", dataframe=items,
                          index="item_id")
    es = es.add_relationship("users", "user_id", "orders", "user_id")
    es = es.add_relationship("orders", "order_id", "items", "order_id")

    # our strict `nb_ts < node_ts` == featuretools' inclusive `<= ts - 1s`
    cutoffs = pd.DataFrame({
        "user_id": users["user_id"],
        "time": users["signup_time"] - pd.Timedelta(seconds=1),
    })

    fm, _ = ft.dfs(
        entityset=es, target_dataframe_name="users",
        agg_primitives=["count", "mean", "max"],
        trans_primitives=[], max_depth=2,
        cutoff_time=cutoffs, cutoff_time_in_index=False,
    )
    fm = fm.sort_index()

    def compare(name, ft_col, ours_key, empty_fill=0.0):
        ft_vals = fm[ft_col].to_numpy(dtype="float64")
        ft_vals = np.nan_to_num(ft_vals, nan=empty_fill)  # empty aggregates
        our_vals = np.asarray(ours[ours_key], dtype="float64")
        ok = np.allclose(ft_vals, our_vals, atol=1e-6, equal_nan=True)
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {name:28s} max|Δ|={np.abs(ft_vals-our_vals).max():.2e}")
        if not ok:
            for i, (a, b) in enumerate(zip(ft_vals, our_vals)):
                if not np.isclose(a, b, atol=1e-6):
                    print(f"      user {i}: featuretools={a:.6f} ours={b:.6f}")
        return ok

    print("Phase B: featuretools", ft.__version__)
    print("feature columns:", [c for c in fm.columns][:10])
    results = [
        compare("COUNT(orders)", "COUNT(orders)", "d1_count"),
        compare("MEAN(orders.amount)", "MEAN(orders.amount)", "d1_mean_amount"),
        compare("MAX(orders.amount)", "MAX(orders.amount)", "d1_max_amount"),
        compare("MEAN(orders.MEAN(items.price))",
                "MEAN(orders.MEAN(items.price))", "d2_mean_mean_price"),
    ]
    if all(results):
        print("\nPARITY: ALL AGGREGATES MATCH FEATURETOOLS")
    else:
        raise SystemExit("\nPARITY FAILURE — see deltas above")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--out", default="/tmp/dfs_parity")
    a = ap.parse_args()
    (phase_a if a.phase == "a" else phase_b)(a.out)
