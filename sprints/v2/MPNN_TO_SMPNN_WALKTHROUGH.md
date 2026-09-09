# From MPNN to SMPNN to SMPNN+Transformer: A Step-by-Step Architecture Walkthrough

Three architectures, three layers of evolution. Each section breaks down
the **per-layer block** in detail (equations, what each operation does,
why it's there, how it's initialised) and shows the cumulative diff
relative to the previous architecture.

Companion to [SMPNN_GRIFFIN_DESIGN.md](SMPNN_GRIFFIN_DESIGN.md) and
[ARCHITECTURE_VARIANTS.md](ARCHITECTURE_VARIANTS.md). This doc is the
"how does the per-layer block actually work" reference.

---

## 0. Conventions used in this doc

| Symbol | Meaning |
|---|---|
| `X ∈ R^{N×D}` | Per-layer node features. N nodes in the current subgraph batch, D = hiddim (=512 in our runs). |
| `Ã` | Heterogeneous relational message-passing operator. Concretely: for each (target node, edge type) bucket, mean-aggregate sources, then max-aggregate across edge types. Implemented in `RMPNN`. |
| `H₁, H₂, ...` | Intermediate hidden states inside a layer. |
| `α` | Learnable scalar scaling factor (DiT-style identity-init residual). |
| `W₁, W₂, W_Q, ...` | Learnable weight matrices. |
| `LN` | LayerNorm with learnable affine (γ, β). |
| `SiLU` | x · sigmoid(x). Smooth nonlinearity. |
| `gate(x)` | Per-node scalar gate ∈ R^N, Griffin-specific, zero-initialised. |
| `MLP_pre` | Pre-aggregation projection: in our code `Linear(hiddim, hiddim, bias=False) → SiLU`. |
| `MLP_ff` | Post-aggregation feedforward: `Linear → SiLU → Linear`. |

All architectures process a heterogeneous relational subgraph (a batch
of FK/PK-linked rows extracted from a RelBench database). The
*per-layer block* below is the unit that gets stacked `num_mp` times.

---

## 1. MPNN architecture (vanilla Griffin)

Reference: vanilla Griffin block, as in [hmodel.py](../../hmodel.py) when
`use_smpnn=False`. This is the architecture every other variant is
measured against.

### 1.1 Block structure

A single MPNN layer applies **one** block of work and produces a residual
update to the node features:

```
                    ┌──────────────────────────────────┐
   X ─────────────► │     PARALLEL UPDATE (Eq. 1.1)    │ ─────► X_next
                    └──────────────────────────────────┘
                                    │
                                    │ X_next = X + LN-then-aggregate
                                    │            + LN-then-FF
                                    │            ───────────────
                                    │            (added in parallel)
```

There is **no internal residual between sub-blocks** — the FF and the
GNN aggregation both feed off the *same* normalised input and their
outputs are summed into a single residual update.

### 1.2 Mathematical specification

```
Layer i:
    H₁         = LN(X)                                                  (Eq. 1.1a)
    gate_fwd   = gate_i(H₁)                                             (Eq. 1.1b — Griffin add)
    H_fwd      = gate_fwd ⊙ Ã · MLP_pre(H₁)                             (Eq. 1.1c)
    gate_rev   = revgate_i(H₁)                                          (Eq. 1.1d — Griffin add)
    H_rev      = gate_rev ⊙ Ã^T · MLP_pre(H₁)                          (Eq. 1.1e)
    H_ff       = MLP_ff(H₁)                                             (Eq. 1.1f)
    X_next     = X + H_fwd + H_rev + H_ff                              (Eq. 1.1g)
```

Note that `LN` here is a *shared, non-affine* LayerNorm (i.e. with
`elementwise_affine=False`). It's the same instance reused across all
layers.

### 1.3 What each step does and why

| Step | Operation | Purpose |
|---|---|---|
| Eq. 1.1a | `H₁ = LN(X)` | Stabilise activations before all three downstream paths. |
| Eq. 1.1b | `gate_i(H₁) ∈ R^N` | Per-node forward-edge gate. Zero-init so contribution is 0 at step 0 — lets a freshly initialised network warm-start cleanly. |
| Eq. 1.1c | `Ã · MLP_pre(H₁)` | The actual message-passing: project each node, then aggregate via the relational mean+max operator. |
| Eq. 1.1d/e | Symmetric reverse-edge path | Captures information flowing *into* a node from incoming FK references. |
| Eq. 1.1f | `MLP_ff(H₁)` | Pointwise feedforward — refines each node's features without using any graph structure. |
| Eq. 1.1g | Sum into residual | Standard residual update. |

### 1.4 What's NOT in the MPNN block

- ❌ No per-layer affine LayerNorm. Just one shared non-affine LN.
- ❌ No α scaling factors. Each sub-block contributes with weight 1.
- ❌ No separate residual between the FF and the MP. They're parallel,
  added together once at the end.
- ❌ No global attention. Information only flows along the actual graph
  edges.

### 1.5 Initialization and warm-start

- Gates `gate_i` and `revgate_i`: last linear layer's weight zero-init.
  At step 0 the gate output is 0, so MP contributions are 0.
- Linear weights in `MLP_pre`, `MLP_ff`: default PyTorch init
  (kaiming-uniform).
- LayerNorm has no learnable params (non-affine), so nothing to init.

Net effect at step 0: gates are 0, so `H_fwd = H_rev = 0`. The only
contribution is `H_ff`. The network is *not* purely identity at init —
the FF sub-block is fully active. This is fine because `H_ff` is small
relative to `X`'s scale early in training.

### 1.6 Parameter count per layer (hiddim=512)

| Component | Params |
|---|---|
| `MLP_pre` (Linear, no bias) | 512² = 262,144 |
| `MLP_ff` (Linear → Linear, no bias) | 2 × 262,144 = 524,288 |
| `gate_i` and `revgate_i` (Linear → Linear) | small, ~25K |
| RMPNN's `rellin` (Linear) | 262,144 |
| Reverse-RMPNN's `rellin` | 262,144 |
| **Total per layer** | ~1.34M |

At num_mp=4 the model totals ~11.4M params (some overhead from the
per-table cross-attention aggregators).

---

## 2. SMPNN update

Reference: SMPNN paper (Saez de Ocariz Borde et al. 2024) Eqs. 5-8.
Implemented in [hmodel_smpnn.py](../../hmodel_smpnn.py).

### 2.1 What changes vs MPNN

| Aspect | MPNN | SMPNN |
|---|---|---|
| Block structure | 1 sub-block (parallel) | **2 sub-blocks (sequential)** |
| Residual addition | 1 residual at end | **2 residuals (one per sub-block)** |
| LayerNorm | 1 shared non-affine | **Per-layer affine `ln_gnn` + `ln_ff`** |
| Scaling | Unscaled (×1) | **Learnable `α_gnn`, `α_ff` scalars (init 1e-6)** |
| FF input | LN(X) (same as MP input) | **LN(X + α_gnn · GNN_out)** — sees the GNN's update |
| MLP_pre activation | inside Linear | unchanged (still Linear → SiLU) |
| Gates | unchanged | unchanged (still zero-init) |
| Reverse edges | unchanged | unchanged |

### 2.2 Block structure (visual)

```
                ┌────────────────────────────────┐
                │ Sub-block 1: GNN               │
   X ─┬──────► LN_gnn ──► MLP_pre ──► Ã (mean+max)
      │            ▲          │             │
      │            └─ gate ◄──┘             │ ───► gnn_out
      │                                     │
      ▼                                     ▼
      X ──────────────────── + α_gnn · ──────  ──► X' (intermediate)
                                                          │
                ┌────────────────────────────────┐
                │ Sub-block 2: FF                │
      X' ─┬──► LN_ff ──► MLP_ff ──► ff_out
          │                                     ▼
          └────────────────── + α_ff · ───────────► X_next
```

The key new feature: **sub-block 2 receives X', not X**. The FF refines
the embedding *after* it has been updated by the GNN, instead of acting
in parallel.

### 2.3 Mathematical specification

```
Layer i:
    # ─── Sub-block 1: GNN ───
    H₁     = LN_gnn_i(X)                                                (Eq. 2.1)
    gnn    = gate_i(H₁) ⊙ Ã · MLP_pre(H₁)                              (Eq. 2.2)
    gnn   += revgate_i(H₁) ⊙ Ã^T · MLP_pre(H₁)         [use_rev=True]  (Eq. 2.3)
    X'     = X + α_gnn_i · gnn                                           (Eq. 2.4)

    # ─── Sub-block 2: FF (Pre-LN, residual) ───
    H₂     = LN_ff_i(X')                                                (Eq. 2.5)
    X_next = X' + α_ff_i · MLP_ff(H₂)                                   (Eq. 2.6)
```

Eqs. 2.4 and 2.6 are both `x + α · f(LN(x))` residual updates — the
classic Pre-LN Transformer pattern, but with the attention replaced by
relational message-passing.

### 2.4 What each step does and why

| Step | Operation | Purpose |
|---|---|---|
| Eq. 2.1 | `LN_gnn_i(X)` | Per-layer affine LN before sub-block 1. Affine = each layer can rescale/shift independently. |
| Eq. 2.2-2.3 | Gated forward + reverse MPNN | Same as MPNN. Carries the relational information. |
| Eq. 2.4 | `X + α_gnn_i · gnn` | **First residual**. α_gnn_i is a learnable scalar initialised at 1e-6 (= near-identity). At step 0 the GNN contribution is ~zero, so X' ≈ X. As training proceeds, the model **learns how much GNN signal each layer should let through**. |
| Eq. 2.5 | `LN_ff_i(X')` | Per-layer affine LN before sub-block 2. Critically: this is LN of X', not X. The FF gets the post-GNN representation. |
| Eq. 2.6 | `X' + α_ff_i · MLP_ff(H₂)` | **Second residual**. α_ff_i is the second learnable scalar, also init 1e-6. The FF starts near-zero and grows as needed. |

### 2.5 Why two residuals (instead of one parallel sum)?

This is the central insight of the SMPNN paper. With one residual:

- The FF and the GNN are **co-active** at every layer.
- An over-active FF can wash out the GNN's signal.
- Deeper networks suffer from oversmoothing because each layer mixes
  features uniformly.

With two sequential residuals:

- Each sub-block has its **own α scaling**, so the network can
  selectively dampen or amplify each path per-layer.
- The FF sees the GNN's update, allowing it to *refine* relational
  features instead of competing.
- Identity-init (α≈0) means **a fresh deep network behaves like a
  shallow one** — there's no gradient catastrophe to overcome. The
  network discovers depth as needed.

This is what enables Griffin at `num_mp=6` (SMPNN-6) to outperform
vanilla Griffin at `num_mp=4` (vanilla-4) without collapsing — the
extra two layers contribute only as much as their learned α allows.

### 2.6 What "α scaling" really does

Each `α_gnn_i` and `α_ff_i` is a single scalar parameter, initialised at
1e-6. During training, gradient descent adjusts these. Three regimes
observed in our runs:

1. **α stays at 1e-6 or smaller**: that layer is effectively dormant.
   The model has chosen to skip it. Useful safety property at depth 8.
2. **α grows to ~0.1-1**: meaningful contribution. The layer is "on".
3. **α grows past 1**: layer dominates the residual. Rare; usually
   indicative of the model finding a critical layer.

The `alpha_init` hyperparameter (V5 = 1e-2 vs V2 = 1e-6) determines how
quickly the α terms can grow during training under a fixed gradient
budget. Aggressive init (1e-2) means the layer is "on" from epoch 0,
yielding earlier convergence on tasks that benefit from depth.

### 2.7 Initialization and warm-start

- LN affine: `γ=1, β=0` → identity-equivalent at init.
- `α_gnn = α_ff = alpha_init = 1e-6` → near-zero contribution from each
  sub-block at step 0.
- Gates still zero-init from the vanilla MPNN side.

Net effect at step 0:

```
gnn ≈ 0 (gates zero) → X' = X + 1e-6 · 0 = X
ff_out = MLP_ff(LN_ff(X)), but multiplied by 1e-6
X_next = X + 1e-6 · ff_out ≈ X
```

The whole network is essentially identity at step 0. **This is what
enables warm-starting a deep SMPNN network from a shallower vanilla
Griffin checkpoint** without any loss spike — the new layers don't
contribute until training has had time to grow their αs.

### 2.8 Parameter count per layer (hiddim=512)

| Component | Params (delta vs MPNN) |
|---|---|
| MPNN components (MLP_pre, MLP_ff, gates, RMPNNs) | 1.34M (unchanged) |
| `ln_gnn_i` (affine LayerNorm) | 1,024 (γ + β) |
| `ln_ff_i` (affine LayerNorm) | 1,024 |
| `α_gnn_i`, `α_ff_i` (two scalars) | 2 |
| **Total per layer** | ~1.34M + 2,050 |

Essentially no new parameters — SMPNN packaging is almost free. At
num_mp=6, the model is ~17.2M params vs vanilla-4's 11.4M; the +5.8M
difference comes from the **2 extra layers**, not from the SMPNN
machinery.

### 2.9 The α-init flag (V2 vs V5 vs V6)

The single hyperparameter we tune:

| Variant | `alpha_init` | Why |
|---|---|---|
| V2 (D1) | 1e-6 | SMPNN paper default. Conservative: each new layer is dormant at init. |
| V6 (D2) | 1e-4 | Intermediate. Marginal practical effect. |
| V5 (D3) | 1e-2 | Aggressive: each layer is meaningfully active from epoch 0. Best on others-cluster, worst on commerce regression. |

Empirically, V5's aggressive init helps when the task benefits from
fully-trained depth (e.g. others-1 reaches 0.5650 vs V2's 0.5572). It
hurts on commerce regression where slower ramp-up may give MLP-style
features a chance to dominate.

---

## 3. SMPNN with global attention (transformer add-on)

Reference: SMPNN paper Appendix A, Eqs. 13-21. Implemented in
[hmodel_smpnn.py:LinearGlobalAttention](../../hmodel_smpnn.py) +
the `use_attention=True` branch of `_gnn_ff_step`.

### 3.1 What changes vs SMPNN

| Aspect | SMPNN | SMPNN + global attention |
|---|---|---|
| Sub-block 1 paths | Local (RMPNN) only | **Local AND global, summed inside the same residual** |
| Pre-LN for global path | n/a | **Separate `ln_global` per layer (NOT shared with `ln_gnn`)** |
| Global attention | none | **Linear global attention via a virtual-node query** |
| α scaling | one α per sub-block | unchanged — the same `α_gnn` scales `(local + global)` |
| Parameter count | 17.2M (num_mp=6) | **21.9M (+4.72M for W_Q/W_K/W_V across 6 layers)** |

The transformer add-on is **only added to sub-block 1**. The FF
sub-block is unchanged.

### 3.2 Block structure (visual)

```
                                    LOCAL PATH (RMPNN)
                                 ┌─────────────────────────┐
                                 ▼                          ▼
   X ─┬──► LN_gnn  ──► MLP_pre ─►Ã                          │
      │                                                     │
      │       GLOBAL PATH (linear attention)                │
      │       ┌─────────────────────────────┐               │
      │       ▼                              ▼              ▼
      ├──► LN_global ──► LinearGlobalAttn ────►─── sum ──────────►
      │                                                     │
      ▼                                                     ▼
      X ────────────────────────────────────── + α_gnn · ───────► X' (intermediate)
                                                                       │
                       (Sub-block 2 unchanged — see §2.3)               │
                                                                       ▼
                                                                    X_next
```

### 3.3 Mathematical specification

```
Layer i (with attention):
    # ─── Sub-block 1 — LOCAL path ───
    H₁_local  = LN_gnn_i(X)                                             (Eq. 3.1)
    gnn_local = gate_i(H₁_local) ⊙ Ã · MLP_pre(H₁_local)               (Eq. 3.2)
    gnn_local += revgate_i(H₁_local) ⊙ Ã^T · MLP_pre(H₁_local)         (Eq. 3.3)

    # ─── Sub-block 1 — GLOBAL path (the transformer) ───
    H₁_global = LN_global_i(X)                                          (Eq. 3.4 — SEPARATE LN)
    Q = H₁_global · W_Q_i                                                (Eq. 3.5)
    K = H₁_global · W_K_i                                                (Eq. 3.6)
    V = H₁_global · W_V_i                                                (Eq. 3.7)

    Q_sn = Σ_n Q_n / ||Σ_n Q_n||₂                                       (Eq. 3.8)
    K_n  = K / ||K||₂  (per-row L2 normalize)                            (Eq. 3.9)

    a = softmax(Q_sn · K_n^T)  ∈ R^N                                    (Eq. 3.10)
    g = Σ_n a_n · V_n  ∈ R^D                                            (Eq. 3.11)
    gnn_global = broadcast(g, N) ∈ R^{N×D}                              (Eq. 3.12)

    # ─── Sub-block 1 — combine and residual ───
    X' = X + α_gnn_i · (gnn_local + gnn_global)                         (Eq. 3.13)

    # ─── Sub-block 2: FF (unchanged from SMPNN §2) ───
    H₂     = LN_ff_i(X')                                                (Eq. 3.14)
    X_next = X' + α_ff_i · MLP_ff(H₂)                                   (Eq. 3.15)
```

### 3.4 What each new step does and why

| Step | Operation | Purpose |
|---|---|---|
| Eq. 3.4 | `LN_global_i(X)` | **Separate LN for the global path**. Why separate? The local path's LN learns to normalize for relational features; the global path's LN learns to normalize for "what should be globally attended to". Two different distributions, two different LNs. |
| Eq. 3.5-3.7 | Q, K, V projections | Standard transformer-style query/key/value computation. Bias-free (matches paper). |
| Eq. 3.8 | `Q_sn = sum(Q) / ||sum(Q)||₂` | **The "virtual-node query"**. Instead of an O(N²) all-pairs attention, the entire batch is summarised into a single direction in feature space. **This is the linearisation trick** — what makes the attention O(N) instead of O(N²). |
| Eq. 3.9 | `K_n = K / ||K||₂` | Normalise each key independently. Combined with Eq. 3.8, this means the dot product `Q_sn · K_n^T` becomes a cosine-similarity-like quantity, well-bounded and stable to softmax. |
| Eq. 3.10 | `softmax(Q_sn · K_n^T)` | Compute a single distribution `a` over the N nodes. Tells us *"how much each node contributes to the global summary"*. |
| Eq. 3.11 | `g = Σ a_n V_n` | Aggregate the V values weighted by `a`. Produces a single D-dim summary of the entire batch. |
| Eq. 3.12 | `broadcast(g, N)` | **Every node in the batch sees the same global summary**. This is the "1_N ⊗ V" operation in the paper. |
| Eq. 3.13 | Combined residual | The local (graph-edge-restricted) and global (batch-summary) contributions are summed INSIDE a single α-scaled residual. One α, not two. |

### 3.5 Why this is "linear" attention

Standard multi-head attention computes an N×N matrix of pairwise scores
— O(N²) memory and compute. For a batch with N=100,000 nodes that's
40 GB at 32-bit precision. Infeasible.

Linear global attention sidesteps this:

1. **Summarise** all N queries into a single direction Q_sn ∈ R^D. (Eq.
   3.8) Cost: O(N·D).
2. **Score** each of the N keys against the single Q_sn. (Eq. 3.10) Cost:
   O(N·D).
3. **Weighted-aggregate** the V matrix using the scores. (Eq. 3.11) Cost:
   O(N·D).

Total: O(N·D). Linear in N, constant relative to the batch size.

### 3.6 What the global path actually achieves on a heterogeneous graph

The local path can only propagate information along actual FK/PK edges
in `num_mp` hops. For deep relational structures (e.g., "user → click →
ad → campaign → advertiser"), 4-6 layers may not be enough to thread
information across the entire schema.

The global path bypasses this: every node sees the **batch-level
summary** of every other node, regardless of edge structure. In
principle this should help when:

- Cross-table information matters but isn't directly edge-linked.
- Distribution shifts in the batch carry signal (e.g. "is this a bot
  user?" — the answer depends on what other users in the batch look
  like).

**Empirically on RDB**: the attention path **does not help** (in fact
hurts by 2-3% on native training). Hypothesis: heterogeneous RDB
subgraphs already carry sufficient cross-table signal via the explicit
FK/PK edges, so the global summary mostly injects noise.

### 3.7 Multi-head attention

When `num_heads > 1`, the D dimension is split into H heads of size
d = D/H. Each head has its own (W_Q, W_K, W_V) and computes its own
Q_sn, K_n, a, g. The final `g` for the layer is the concatenation of
per-head g's, reshaped to D. Mathematically equivalent to summing
per-head contributions when V_head ∈ R^{N×d}.

Default `num_heads=1` works for our hiddim=512 (one head of size 512).
Paper finds little benefit from multi-head on the homogeneous
benchmarks, consistent with our RDB results.

### 3.8 Initialization

| Param | Init | Notes |
|---|---|---|
| `LN_global_i` (affine) | γ=1, β=0 | Identity-equivalent |
| `W_Q_i`, `W_K_i`, `W_V_i` | Kaiming-uniform | Random non-zero |
| `α_gnn_i` (shared with local path) | 1e-6 | At step 0, *both* local AND global outputs are scaled to near-zero |

So at step 0 the entire sub-block 1 (local + global combined) is
near-identity. The model has to *learn* to use the global path
non-trivially, just like it has to learn to use the local path. The
α_gnn gates them together — they can't decouple at the layer level.

### 3.9 Parameter cost analysis

Per layer with attention:

| Component | Params |
|---|---|
| All SMPNN components (from §2) | 1.34M |
| `LN_global_i` | 1,024 |
| `W_Q_i` (Linear, no bias) | 512² = 262,144 |
| `W_K_i` | 262,144 |
| `W_V_i` | 262,144 |
| **Per-layer attention overhead** | ~787K |

At num_mp=6: 6 × 787K = **+4.72M params**. Total ~21.9M vs SMPNN-6's
17.2M. Matches the empirical Param counts in our runs exactly.

For a paper that argues attention isn't worth it on RDB, this 27%
parameter overhead is part of the case: not only does it not help, it's
expensive in memory and compute.

### 3.10 Why combine inside one residual instead of two

Could have chosen:

```
X' = X + α_local · gnn_local + α_global · gnn_global    (alternative, NOT used)
```

But the paper (Eq. 21) uses:

```
X' = X + α_gnn · (gnn_local + gnn_global)                (Eq. 3.13, what we use)
```

The single-α formulation forces the model to **co-modulate** the two
paths. If global attention is helpful only when paired with strong
local signal, a shared α lets them grow together. If the model wanted
to use only global, it would have to push α_gnn up, which would also
let local through.

This is a deliberate design choice in the paper. We follow it. (An
alternative would be worth ablating, but adds parameters and is out of
scope for our current sweep.)

---

## 4. Cumulative diff summary

| Component | MPNN (vanilla) | SMPNN | SMPNN + transformer |
|---|---|---|---|
| Block sub-blocks | 1 (parallel) | 2 (sequential) | 2 (sequential) |
| Per-layer LNs | 1 shared non-affine | `ln_gnn`, `ln_ff` per layer | + `ln_global` per layer |
| α scaling | none | `α_gnn`, `α_ff` per layer | unchanged |
| Identity-init capability | partial (gates only) | **full** (gates + α) | full |
| Global information flow | along edges only | along edges only | + virtual-node attention |
| Per-layer params (hiddim=512) | ~1.34M | ~1.34M + 2K LN/α | ~1.34M + 2K + 787K attention |
| At num_mp=6 total | n/a (vanilla uses 4) | **17.2M** | **21.9M** |
| RDB headline result | 0.5476 (V0, num_mp=4) | **0.5650 (V5, α=1e-2)** | 0.5241 (V7, hurts!) |

---

## 5. What each architecture is good for

| If you want... | Use | Why |
|---|---|---|
| The simplest baseline | **MPNN (V0)** | Works well as a TabPFN embedding (rank 1 under TabPFN ZS no-proj on others-1!) |
| Best native-head training on others-cluster | **SMPNN-6 + α=1e-2 (V5)** | +1.7% over vanilla-4 in-distribution |
| Best on commerce-cluster | **SMPNN-6 default (V2)** | Paper default is correct here; aggressive α hurts regression |
| Smaller model that still performs | **SMPNN-6 no-FF (V8b)** | 14M params (3M less than V5/V2) with ~0.02 native drop |
| **Avoid** | SMPNN + attention (V7), vanilla-6 (V4) | V7 wastes 5M params for no gain; V4 looks fine on native but collapses on embedding quality |

---

## 6. Two implementation details worth knowing

### 6.1 SiLU placement (deviation from paper, documented)

The paper's Eq. 6 applies SiLU **after** aggregation:
`H₂ = α₁ · SiLU(Ã · H₁ · W₁) + X`. Our `hmodel_smpnn.py` applies SiLU
**before** aggregation, via `mlp = Linear → SiLU` then aggregate. The
two are mathematically different but the paper's "where to put the
nonlinearity" is a known design choice (Pre/Post-aggregation activation
is debated in the GNN literature).

We don't fix this because our current backbone checkpoints were trained
with the pre-aggregation SiLU, and fixing would require retraining all
ablation variants. For the paper we document the choice in the methods
section.

### 6.2 Reverse edges and gates are unchanged across all three architectures

`gate_i`, `revgate_i`, `Ã^T` (reverse RMPNN) are Griffin-specific
additions to the base relational MPNN. They survive the SMPNN
repackaging and the attention augmentation unchanged. This means: the
SMPNN block isn't a *replacement* for Griffin's gating mechanism — it's
a different way of **composing** the gated MP with the rest of the
layer (Pre-LN, sub-blocks, residuals, α).

---

## 7. References

- Saez de Ocariz Borde et al. (2024), *Scalable Message Passing Neural
  Networks: No Need for Attention in Large Graph Representation
  Learning*. arXiv:2411.00835. Section 3.2 (block), Appendix A
  (attention).
- Peebles & Xie (2023), *Scalable Diffusion Models with Transformers*
  (DiT). Source of the identity-init α-scaling pattern.
- Xiong et al. (2020), *On Layer Normalization in the Transformer
  Architecture*. Pre-LN vs Post-LN motivation.
- Griffin original implementation (this repository, `hmodel.py`).
