# Dataset Size Inventory: joint-v65 Families & RelBench v2

**Date:** 2026-08-25
**Purpose:** Record-keeping for dataset footprints — disk, node rows, FK links,
and supervision labels — per task family in `datasets/joint-v65`, plus the
RelBench v2 registry for a comprehensive external view. Companion to
`NEIGHBORHOOD_STATS.md` (per-row neighborhood structure), `COMMERCE_DFS_RUNBOOK.md`
(Step 0 conversion), and `RATEBEER_PLAN.md`.

Measurement notes: disk = `du` of `node/` + `edge/` dirs (includes text
embeddings); rows = feat-dataset lengths over ALL node types incl. label-holder
types; FK links = sum of adjacency-list lengths over all edge columns
(directed FK endpoints, so one relational edge counts once per FK). Task labels
from `metatask.yaml` splits.

---

## 1. joint-v65 corpus totals

| component | size |
|---|---|
| `node/` | 78 GB |
| `edge/` | 12 GB |
| `task/` | 325 MB |
| `dfs/` | 9.1 MB (rel-f1 only — see §4) |
| **whole corpus** | **≈ 90 GB**, 19 dataset families, 131 node types |

The four experiment families below cover ~81 GB (~90%) of the corpus; the
remainder is facebook-recruiting + the `single`-family tabular datasets.

## 2. Per-family inventory (joint-v65)

### commerce-1 (train family) — 4 datasets, 32 node types, ≈ 11.1 GB

| dataset | node types | rows | FK links | disk | largest tables |
|---|---|---|---|---|---|
| retailrocket | 8 | 24.9M | 53.1M | 6.2 GB | ItemProperty 18.0M, View 2.7M |
| rel-hm | 4 | 16.7M | 60.7M | 3.2 GB | transactions 15.2M, customer 1.37M |
| diginetica-downsample | 14 | 14.4M | 53.8M | 1.4 GB | QueryResult 9.2M, View 1.2M |
| seznam | 6 | 2.7M | 5.2M | 0.19 GB | Probehnuto 1.5M |
| **total** | **32** | **58.6M** | **172.8M** | **≈ 11.1 GB** | |

Task labels (6 tasks): train 11,102,951 / val 397,530 / test 396,093 =
**11.90M** — dominated by rel-hm-item-sales (5.7M) and rel-hm-user-churn (4.0M).

### commerce-2 (eval family) — 3 datasets, 29 node types, ≈ 52.9 GB

| dataset | node types | rows | FK links | disk | largest tables |
|---|---|---|---|---|---|
| amazon | 4 | 16.1M | 54.9M | 43.3 GB | Review 13.7M, Customer 1.85M |
| rel-avito | 15 | 20.8M | 101.9M | 9.2 GB | SearchStream 7.1M, AdsInfo 6.0M |
| outbrain-small | 10 | 4.8M | 8.6M | 0.4 GB | User 2.0M, Pageview 2.0M |
| **total** | **29** | **41.6M** | **165.4M** | **≈ 52.9 GB** | |

amazon dominates disk but not rows: 13.7M Review nodes carry text embeddings.
Task labels (6 tasks): **1.82M** total — amazon-churn 1,347K
(1,045,568/149,205/152,486), user-visits 153K, user-clicks 129K,
amazon-rating 100K, outbrain-small-ctr 87K, rel-avito-ad-ctr 8.7K.

### others-1 (train family, historical default) — 3 datasets, 28 node types, ≈ 6.2 GB

| dataset | node types | rows | FK links | disk | largest tables |
|---|---|---|---|---|---|
| stackexchange | 11 | 6.1M | 18.3M | 6.1 GB | Vote 1.7M, PostHistory 1.5M |
| rel-f1 | 11 | 74K | 339K | 16 MB | standings 28K, results 20K |
| virus | 6 | 30K | 78K | 8 MB | Spray 15K |
| **total** | **28** | **6.2M** | **18.7M** | **≈ 6.2 GB** | |

Task labels (6 tasks): train 480,825 / val 129,248 / test 147,017 = **757K**.

### others-2 (eval family) — 4 datasets, 37 node types, ≈ 11.0 GB

| dataset | node types | rows | FK links | disk | largest tables |
|---|---|---|---|---|---|
| talkingdata | 8 | 36.7M | 138.6M | 3.5 GB | App_events 32.5M, Events 3.3M |
| airbnb | 7 | 10.8M | 21.9M | 1.0 GB | Session 10.6M |
| rel-trial | 16 | 5.4M | 15.3M | 6.5 GB | facilities_studies 1.8M |
| telstra | 6 | 0.15M | 0.26M | 12 MB | Log_feature 59K |
| **total** | **37** | **53.1M** | **176.1M** | **≈ 11.0 GB** | |

Task labels (6 tasks): **553K** total — airbnb-destination 213.5K,
rel-trial-site-success 193.8K, talkingdata-demo-pred 74.6K,
study-adverse 50.0K, study-outcome 13.8K, telstra-severity 7.4K.

### Combined views

| combination | datasets | node types | rows | FK links | disk | task labels |
|---|---|---|---|---|---|---|
| commerce-2 + others-2 (both eval families) | 7 | 66 | 94.7M | 341.5M | ≈ 64 GB | 2.38M (12 tasks) |
| all four families | 14 | 126 | 159.6M | 533.0M | ≈ 81 GB | ≈ 15.0M (24 tasks) |

## 3. RelBench v2 registry (relbench 2.1.1, `llm` conda env)

Registry: rel-amazon, rel-avito, rel-event, rel-f1, rel-hm, rel-stack,
rel-mimic, rel-trial, rel-arxiv, rel-salt, rel-ratebeer (+ dbinfer-* ports and
tgb*/thgl* temporal sets). Locally cached under `~/.cache/relbench/`:
rel-f1, rel-salt, rel-ratebeer.

### Fully measured (cached; `get_db()` = temporally clipped, as training sees it)

| dataset | tables | rows | FK links | parquet disk | largest tables |
|---|---|---|---|---|---|
| rel-ratebeer | 13 | 13.8M | 27.0M | 2.73 GB | beer_ratings 11.8M, beers 752K |
| rel-salt | 4 | 4.26M | 9.7M | 60 MB | salesdocumentitem 1.92M, address 1.79M |
| rel-f1 | 9 | 74K | 169K | 2 MB | standings 28K |

(rel-ratebeer raw parquet before clipping: ~16.7M rows incl. 14.3M ratings.)

### Whole registry — official `db.zip` sizes (HTTP HEAD, no download) + tasks

| dataset | zip size | tasks | note |
|---|---|---|---|
| rel-amazon | **6.40 GB** | 8 | ported into joint-v65 as `amazon` (16.1M rows / 54.9M links) |
| rel-ratebeer | 2.32 GB | 8 | cached ✓ — new in v2 |
| rel-stack | 0.88 GB | 6 | joint-v65 has the dbinfer stackexchange port instead |
| rel-trial | 0.57 GB | 9 | in joint-v65 (5.4M rows / 15.3M links) |
| rel-avito | 0.36 GB | 6 | in joint-v65 (20.8M rows / 101.9M links) |
| rel-arxiv | 0.15 GB | 4 | new in v2 |
| rel-hm | 0.14 GB | 4 | in joint-v65 (16.7M rows / 60.7M links) |
| rel-event | 0.11 GB | 6 | NOT in joint-v65 (would need re-processing) |
| rel-mimic | 0.05 GB | 1 | new in v2; tiny — likely demo subset of MIMIC |
| rel-salt | 0.04 GB | 8 | cached ✓ — new in v2 |
| rel-f1 | 2 MB | 6 | cached ✓ |

Whole-registry download ≈ **11 GB compressed** (rel-amazon 58%, rel-ratebeer 21%).

### Cross-connection notes

- **Bytes are not comparable across the two worlds.** RelBench ships raw
  compressed tables; joint-v65 stores Griffin features + precomputed text
  embeddings, inflating the same data ~15–25× (rel-avito 0.36 GB → 9.2 GB;
  rel-trial 0.57 GB → 6.5 GB). Compare row/link counts, not disk.
- **New-in-v2 material not yet in joint-v65:** rel-salt, rel-ratebeer,
  rel-arxiv, rel-mimic, rel-event. The substantial ones are rel-salt (4.3M
  rows, SAP sales, 8 tasks) and rel-ratebeer (13.8M rows, 8 tasks) — see
  `RATEBEER_PLAN.md`. Converting them means a full dataconverter pass; by the
  observed inflation ratio expect roughly 1–2 GB (salt) and 15–30 GB
  (ratebeer) on disk, driven by how much text gets embedded.

## 4. Neighborhood statistics (1/2/3-hop reach per seed row)

Full analysis in `NEIGHBORHOOD_STATS.md`; distribution histograms in
`figs/neighborhood_hop_distributions.png` (log-x, median and p90 marked).
Hop-k = unique nodes first reachable at FK-distance exactly k, both edge
directions, seed excluded.

| dataset (seed row) | hop1 med / mean | hop2 med / mean | hop3 med / mean | hop3 p90 | hop3 max |
|---|---|---|---|---|---|
| rel-hm (customer) | 8 / 15.2 | 8 / ~14 | ~11,900 / ~22,700 | ~56,900 | ~298K |
| rel-hm (article) | 29 / 213 | 30 / ~206 | ~1,500 / ~9,100 | ~25,700 | ~719K |
| rel-salt (salesdocument) | 2 / 4 | 1 / 2 | ~1,380 / ~5,000 | ~13,000 | ~47K |
| rel-ratebeer (user) | 2 / 8 | 2 / 8 | ~1,540 / ~9,700 | ~31,300 | ~219K |

What each hop is:
- **rel-hm customer** → transactions → unique articles → other customers'
  transactions on those articles. (Applies to rel-hm-user-churn.)
- **rel-hm article** → its transactions → unique buying customers → their
  other transactions. (Applies to rel-hm-item-sales; the seed IS the hub —
  the most popular article reaches 719K transactions ≈ 5% of the table in
  3 hops.)
- **rel-salt salesdocument** → line items → customers (4 party-role FKs) →
  customers' addresses + all their other line items.
- **rel-ratebeer user** → ratings/favorites → beers/places → everything on
  those beers (other ratings dominate). Dimension tables (157 styles, 251
  countries, 8 place types) sit 2–3 hops out and connect to everything —
  hop 4 through them sweeps in essentially the whole database.

Shared signature: **hops 1–2 are single-digit; the blowup arrives at hop 3
through popular-entity hubs** (except article-rooted rel-hm, where the seed is
the hub and hop 1 is already large). Degree distributions are heavily
right-skewed (mean ≫ median) — size sampling/memory budgets from p90/p99, not
the median. This is the regime `--hop 2 --fanout 20` and dfscore's
hub-dedup/`hub_budget=2M` are built for; deterministic DFS aggregates matter
most exactly where fanout sampling truncates hardest (hub-rooted seeds like
rel-hm articles: 31K transactions sampled to 20 ≈ 1,500× information loss that
a count/mean aggregate recovers).

Not yet measured this way: the other joint-v65 families (retailrocket,
diginetica, seznam, amazon, rel-avito, talkingdata, …) — the generic script
`scripts/hop_analysis.py` (RelBench) and the §6 method cover how to add them.

## 5. DFS artifact status (as of 2026-08-25)

`datasets/joint-v65/dfs/` contains artifacts for **rel-f1 only** (8.7 MB +
metadfs.yaml). Every other dataset — including all of commerce-1/2 and the
rest of others-1/2 — has none. Consequences:

- E0 / vanilla runs are unaffected everywhere.
- Any DFS cell (E1–E3) on a family other than rel-f1-only tasks requires the
  Step 0 converter pass first (`COMMERCE_DFS_RUNBOOK.md`); node types missing
  from `metadfs.yaml` produce **silent no-DFS behavior** in training.
- Conversion cost scales with the tables above — amazon (13.7M reviews) and
  talkingdata (32.5M App_events) are the long poles for commerce-2/others-2.

## 6. Reproduction

- Disk: `du -sc node/<prefix>-* edge/<prefix>-*` under `datasets/joint-v65`.
- Rows/links: load each `node/<type>/feat` and `edge/<type>/adj` with
  `datasets.load_from_disk` (griffin env); links = sum of
  `pyarrow.compute.list_value_length` over non-timestamp adjacency columns.
- Task labels: `split` field per task in `metatask.yaml`.
- RelBench: relbench 2.1.1 in the `llm` env; zip sizes via
  `curl -sIL https://relbench.stanford.edu/download/<name>/db.zip`.
