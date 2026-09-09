# Multi-Hop Neighborhood Statistics: rel-hm, rel-salt, rel-ratebeer

**Date:** 2026-08-24
**Purpose:** Record-keeping for k-hop neighborhood size analysis used to reason about
fanout sampling (`--hop 2 --fanout 20`) and DFS aggregate cost (`dfscore.py` hub
dedup / `hub_budget`). Companion to `DFS_GRIFFIN_DESIGN.md` and
`COMMERCE_DFS_RUNBOOK.md`.

---

## 1. Method

- **Node/edge model:** each table row is a node; FK links are edges, traversed in
  both directions. Hop-k = nodes first reachable at distance exactly k from the
  seed row (unique nodes, seed and already-visited nodes excluded).
- **Sources:**
  - `rel-hm`: Griffin corpus `datasets/joint-v65` (Arrow adjacency in
    `edge/rel-hm-transactions/adj`), 15,187,287 transactions,
    1,371,980 customers (999,345 with ≥1 transaction), 105,540 articles
    (71,375 with ≥1 transaction).
  - `rel-salt`, `rel-ratebeer`: **RelBench v2** (relbench 2.1.1, installed in the
    `llm` conda env — NOT in `griffin`), cached under `~/.cache/relbench/`.
    rel-salt ≈ 138 MB; rel-ratebeer ≈ 2.32 GB (downloaded 2026-08-24).
  - RelBench numbers below are computed on the **raw parquet** tables;
    relbench's `get_db()` temporally clips to pre-test-timestamp rows
    (e.g. 11.8M of 14.3M ratebeer ratings), so task-time neighborhoods are
    ~20% smaller with the same shape.
- **Stats:** hop-1 exact over all active seeds where noted; hop-2/3 from random
  samples (2,000 seeds for rel-hm, 500 for rel-salt, 200 for rel-ratebeer,
  fixed RNG seed 0). p90/p99 = 90th/99th percentiles.
- **Scripts** (in `sprints/v2/scripts/`, run in the `llm` env):
  `hop_analysis.py` — generic RelBench version, takes
  `<dataset> <seed_table> [n_sample]`; `hop_ratebeer.py` — column-pruned
  parquet + boolean-mask BFS for the 14M-row ratings table on the 15 GB
  machine. rel-hm was computed with a bincount/searchsorted one-off over the
  joint-v65 Arrow adjacency (griffin env).

**Figure:** `figs/neighborhood_hop_distributions.png` — histograms of the
hop-1/2/3 neighborhood-size distributions for all four seed types (log-x,
median and p90 marked; 20,000 seeds for each rel-hm view, 4,000 rel-salt
documents, 1,500 rel-ratebeer users). Regenerate with
`scripts/dump_hm_hops.py` (griffin env) + `scripts/dump_relbench_hops.py`
(llm env), then `scripts/plot_hops.py`.

---

## 2. rel-hm — customer-rooted (task: rel-hm-user-churn)

Schema path: customer → transactions (hop 1) → unique articles (hop 2) →
other customers' transactions on those articles (hop 3). A customer's ONLY
edge type is to `rel-hm-transactions`; each transaction has exactly one
customer and one article FK.

| hop | neighbor type | median | mean | p90 | max |
|---|---|---|---|---|---|
| 1 | transactions | 8 | 15.2 | 37 | 1,039 |
| 2 | unique articles | 8 | ~14 | 32 | — |
| 3 | other transactions | ~11,900 | ~22,700 | ~56,900 | ~298,000 |

Examples: customer #123 (runbook Step-0 example): 4 / 3 / 2,774.
Median customer #41: 8 / 4 / 4,213.

Hop 4 (other customers) would reach hundreds of thousands of distinct
customers — why nothing goes past 2 message-passing hops.

## 3. rel-hm — article-rooted (task: rel-hm-item-sales)

Schema path: article → its transactions (hop 1) → unique buying customers
(hop 2) → those customers' other transactions (hop 3).

| hop | neighbor type | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 1 | transactions | 29 | 213 | 575 | 2,486 | 31,458 |
| 2 | unique customers | 30 | ~206 | ~558 | ~2,424 | ~11,400 |
| 3 | other transactions | ~1,500 | ~9,100 | ~25,700 | ~100,000 | 719,400 (top article) |

Examples: article #0: 107 / 85 / 3,219. Median-degree article #44:
29 / 21 / 1,624. Most-popular article #53892: 31,458 / 21,775 / **719,400**
(≈5% of ALL transactions within 3 hops of one node).

Hop 2 ≈ hop 1 because same-customer repeat purchases of one article are rare.

### Which one applies? Both — per task (from `metatask.yaml`)

- `rel-hm-user-churn`: `seed_type: rel-hm-customer_churn` →
  `target_type: rel-hm-customer` — effectively **customer-rooted** (§2),
  shifted one hop through the label-holder node.
- `rel-hm-item-sales`: `seed_type: rel-hm-article` — **article-rooted** (§3).

| | user-churn (customer root) | item-sales (article root) |
|---|---|---|
| hop-1 size | small, mild skew (med 8, mean 15) | large, extreme skew (med 29, mean 213, max 31K) |
| fanout=20 truncates | rarely at hop 1, heavily at hop 2→3 | immediately at hop 1 for most rows |
| DFS depth-1 aggregate | ~8–15 transactions | hundreds–thousands of transactions |
| DFS depth-2 cost driver | popular articles among purchases | customers' full purchase histories |

Implications:
1. For **item-sales**, depth-1 DFS (E1/E3) summarizes exactly what fanout-20
   sampling throws away (a popular article's 31K transactions are sampled to
   20 — ~1,500× loss the deterministic count/mean recovers). For
   **user-churn**, hop-1 survives sampling mostly intact, so depth-1 DFS adds
   less. Concrete instance of the registered E2 variance-reduction prediction:
   strongest where sampling truncation is severest.
2. Online cutoff-DFS cost is cheap for churn seeds, expensive for item-sales
   seeds — item-sales is the task to watch in conversion logs (hub dedup /
   `hub_budget=2M` exist for the 719K-row hop-3 sets).

---

## 4. rel-salt (RelBench v2) — salesdocument-rooted

4 tables: `salesdocument` 411,966 (root of sales-shipcond / sales-incoterms
tasks) · `salesdocumentitem` 1,916,685 · `customer` 139,607 ·
`address` 1,788,887. Items → document (1 FK) and → customer via **4**
party-role FKs (SOLDTOPARTY, SHIPTOPARTY, BILLTOPARTY, PAYERPARTY);
customer → address. (Raw parquet: 500,908 docs / 2,319,540 items before
temporal clipping.)

Path: document → line items (hop 1) → customers on those items (hop 2) →
customers' addresses + all their OTHER line items (hop 3).

| hop | neighbor type | median | mean | p90 | max |
|---|---|---|---|---|---|
| 1 | items | 2 | 4 | 9 | 167 |
| 2 | unique customers | 1 | 2 | 2 | 3 |
| 3 | addresses + other items | ~1,380 | ~5,000 | ~13,000 | 46,572 |

Example row #0: 2 / 2 / 1,385 (2 addresses + 1,383 items).

Notes:
- Documents are tiny (median 2 items, 1 customer); the blowup is entirely
  **customer hubs** (median customer ≈ 1,400 line items, heavy ≈ 47K).
- The 4 parallel party FKs mean one item can contribute the same customer up
  to 4×: unique-node counts here, but edge-wise fanout is up to 4× larger,
  and per-FK DFS features would see 4 separate aggregation channels.

## 5. rel-ratebeer (RelBench v2) — user-rooted

13 tables; big: `beer_ratings` 14,344,128 · `beers` 1,191,662 ·
`users` 236,969 · `place_ratings` 385,707 · `beer_upcs` 249,348 ·
`favorites` 212,273 · `availability` 117,424 · `places` 86,061 ·
`brewers` 50,013. Tiny dimension tables: `beer_styles` 157 ·
`countries` 251 · `states` 622 · `place_types` 8.
A user connects out through 4 child tables: beer_ratings, place_ratings,
availability, favorites.

Path: user → ratings/favorites (hop 1) → beers/places rated (hop 2) →
everything attached to those beers (hop 3: other ratings, brewers, styles,
UPCs, availability).

| hop | neighbor type | median | mean | p90 | max |
|---|---|---|---|---|---|
| 1 | ratings/favorites | 2 | 8 | 20 | 148 |
| 2 | beers/places | 2 | 8 | 20 | 129 |
| 3 | mostly other beer_ratings | ~1,540 | ~9,700 | ~31,300 | 218,642 |

Example user #0 (heavy, 248 ratings): 248 / 248 / **432,910**
(417K other ratings + 14.2K favorites + …) ≈ 3% of the database in 3 hops.
Median-like user #47265: 5 / 5 / 1,537.

Hazard note: the dimension tables (157 styles, 251 countries, 8 place types)
sit 2–3 hops from any seed and connect to EVERYTHING — hop 4 through them
sweeps in essentially the whole database. Any future Griffin conversion
should treat them like the telstra/virus dimension types.

---

## 6. Cross-dataset comparison

| dataset (seed) | hop1 med / mean | hop2 med / mean | hop3 med / mean | hop3 max | hub mechanism |
|---|---|---|---|---|---|
| rel-hm (customer) | 8 / 15.2 | 8 / ~14 | ~11,900 / ~22,700 | ~298K | popular articles → their transactions |
| rel-hm (article) | 29 / 213 | 30 / ~206 | ~1,500 / ~9,100 | ~719K | customer purchase histories |
| rel-salt (salesdocument) | 2 / 4 | 1 / 2 | ~1,380 / ~5,000 | ~47K | customers → all their line items |
| rel-ratebeer (user) | 2 / 8 | 2 / 8 | ~1,540 / ~9,700 | ~219K | popular beers → their ratings |

Shared structural signature (except article-rooted rel-hm, where the seed IS
the hub): **hops 1–2 are single-digit; the blowup arrives at hop 3 through
popular-entity hubs.** This is exactly the regime Griffin's
`--hop 2 --fanout 20` sampling and dfscore's hub-dedup/`hub_budget` target:

- Degree distributions are heavily right-skewed everywhere (mean >> median);
  p90/p99, not the median, size the sampling and memory budgets.
- Deterministic DFS aggregates are most valuable precisely where fanout
  sampling truncates hardest (hub-rooted seeds like rel-hm articles).
- 3 hops from a single hub node can cover 3–5% of an entire dataset —
  full-neighborhood expansion past hop 2 is never viable.

## 7. Local RelBench v2 status (as checked 2026-08-24)

- relbench **2.1.1** in conda env `llm` (also 2.0.2 in `clavaddpm`); NOT
  installed in `griffin`. Registry includes `rel-salt`, `rel-ratebeer`,
  `rel-mimic`, `rel-arxiv`, the dbinfer-* ports, and tgb*/thgl* temporal sets.
- Cache `~/.cache/relbench/`: rel-f1 (2.2 MB), rel-salt (138 MB),
  rel-ratebeer (2.32 GB). Neither salt nor ratebeer exists in
  `datasets/joint-v65` — using them in Griffin would require a full
  dataconverter pass first (raw → joint format), which is a separate task.
