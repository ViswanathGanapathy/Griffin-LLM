"""Dump per-seed hop1/2/3 counts for rel-hm (customer- and article-rooted)."""
import numpy as np
from datasets import load_from_disk

ds = load_from_disk('/home/pviswanath/Griffin/datasets/joint-v65/edge/rel-hm-transactions/adj')
tbl = ds.data
cust = tbl.column('head of rel-hm-transactions:rel-hm-transactions-customer_id:rel-hm-customer').combine_chunks().flatten().to_numpy()
art  = tbl.column('head of rel-hm-transactions:rel-hm-transactions-article_id:rel-hm-article').combine_chunks().flatten().to_numpy()

cust_deg = np.bincount(cust)
art_deg = np.bincount(art)

def make_index(key):
    order = np.argsort(key, kind='stable')
    sk = key[order]
    n = key.max() + 1
    return order, np.searchsorted(sk, np.arange(n)), np.searchsorted(sk, np.arange(n), side='right')

c_order, c_s, c_e = make_index(cust)
a_order, a_s, a_e = make_index(art)
art_by_cust = art[c_order]
cust_by_art = cust[a_order]

rng = np.random.default_rng(0)

def sample_hops(seed_kind, n_samp):
    if seed_kind == 'customer':
        active = np.flatnonzero(cust_deg > 0)
        s, e, other, other_deg = c_s, c_e, art_by_cust, art_deg
    else:
        active = np.flatnonzero(art_deg > 0)
        s, e, other, other_deg = a_s, a_e, cust_by_art, cust_deg
    samp = rng.choice(active, min(n_samp, len(active)), replace=False)
    out = np.empty((len(samp), 3), dtype=np.int64)
    for i, x in enumerate(samp):
        nb = other[s[x]:e[x]]
        u = np.unique(nb)
        out[i] = (len(nb), len(u), int(other_deg[u].sum()) - len(nb))
    return out

np.savez(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'hops_hm.npz'),
         customer=sample_hops('customer', 20000),
         article=sample_hops('article', 20000))
print("done")
