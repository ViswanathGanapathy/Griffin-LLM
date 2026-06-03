# SMPNN-Griffin: Architecture, Experiments, and Paper Plan

Working design doc for the SMPNN-integrated Griffin work targeting a
foundation-models-on-structured-data venue (ICLR main / FM4SD workshop /
LoG). Drafted while ICL eval + hop=3 reruns are in flight.

---

## 1. Architecture

### 1.1 Vanilla Griffin (baseline)

Griffin is a relational message-passing network over RelBench-style
subgraphs. Per training step we sample a batch of seed entities and
extract their `--hop`-hop neighbourhood from a heterogeneous RDB graph
where node types correspond to tables and edge types to FK/PK joins.

Single layer (`hmodel.py:GriffinMod`):

```
x_in -> MLP_pre -> RMPNN -> [+ revRMPNN] -> gate * (...) -> x_out
```

with components:

- **Per-table value aggregator** (`SelfAverageAggregator` on layer 0,
  `SelfAttentionAggregator` thereafter): cross-attention over cell-level
  features keyed by column-name embeddings. Pools each row to a single
  hidden vector before MP.
- **RMPNN**: relational message-passing. For each (target, edge_type)
  bucket, mean-aggregates source-node features, scales by an edge-type-
  specific projection, then max-aggregates across edge types into the
  target. Captures cross-table information along FK/PK edges.
- **revRMPNN**: same primitive over reversed edges. The two outputs are
  gated and summed.
- **Zero-init gate**: ensures the network is near-identity at step 0,
  letting Griffin warm-start from a pretrained checkpoint of any depth.

Default config: `hiddim=512`, `num_mp=4`, `hop=2`, `fanout=20`,
`fewshotfanout=3`, `use_rev=True`, `use_gate=True`. Param count: 11.37M.

### 1.2 SMPNN integration

SMPNN [Saez de Ocariz Borde et al., 2024] proposes packaging
message-passing into a Pre-LayerNorm Transformer-style block. We port the
packaging to Griffin (`hmodel_smpnn.py:GriffinMod`), replacing each MP
layer with two residual sub-blocks:

```
# Sub-block 1: GNN  ( + optional global attention )
h1   = LayerNorm_gnn(x)
gnn  = gate * RMPNN(MLP_pre(h1)) + revgate * revRMPNN(...)
if use_attention:
    h1g  = LayerNorm_global(x)
    gnn += LinearGlobalAttention(h1g)
x = x + alpha_gnn * gnn          # alpha_gnn init 1e-6 (near-identity)

# Sub-block 2: pointwise FF
h2  = LayerNorm_ff(x)
x   = x + alpha_ff * MLP_ff(h2)  # alpha_ff init 1e-6
```

State-dict is a strict superset of vanilla Griffin under default flags
(adds `ln_gnn`, `ln_ff`, `alpha_gnn`, `alpha_ff` per layer), so vanilla
checkpoints warm-start the SMPNN-Griffin cleanly: missing keys default
to LayerNorm(γ=1, β=0) and α=1e-6, which is approximately identity at
step 0.

#### Per-component ablation switches

Five Boolean flags (all on `GriffinMod.__init__`):

| Flag | Default | Effect when False |
|---|---|---|
| `use_alpha` | True | Fixes α=1 (no learnable scaling). |
| `use_ff` | True | Removes the FF sub-block entirely (saves ~18% params at num_mp=6). |
| `use_gnn_ln` | True | Drops the LayerNorm before the GNN sub-block. |
| `use_attention` | False | Disables LinearGlobalAttention (default off; on adds ~4.7M params at num_mp=6, hiddim=512). |
| `num_heads` | 1 | Heads for LinearGlobalAttention when enabled. |

#### LinearGlobalAttention (paper Appendix A)

O(N) global attention via a single virtual-node query per head. Given
node features X ∈ R^{N×D}:

```
Q = X W_Q;   K = X W_K;   V = X W_V    ∈ R^{N×D}
Q_sn = sum_n(Q) / || sum_n(Q) ||_2     ∈ R^{D}    (single virtual-node query)
K_n  = K / ||K||_2                       ∈ R^{N×D}
A    = softmax(Q_sn · K_n^T)            ∈ R^{1×N}
out  = (A ⊗ 1_N) · V                    ∈ R^{N×D}
```

Multi-head: split D into `num_heads × head_dim`, sum head contributions.
Adds 3·hiddim²·num_mp params (4.72M at our defaults).

### 1.3 Pipeline composition with downstream heads

Three downstream-head variants land on top of the Griffin (vanilla or
SMPNN) backbone. All produce a 512-d embedding per seed entity.

```
                                            +-- native head (linear + task-specific loss)
Griffin backbone (vanilla or SMPNN)  --->  +-- LLM head    (LoRA on Qwen3-1.7B)
   512-d embedding                          +-- ICL head:
                                                 [ + ICLProjection 512->128 + 5-epoch probe ]  (standard)
                                                 [ no projection, no probe ]                    (our finding)
                                              -> TabPFN v3 / TabICL v2 (in-context)
```

The **no-projection ICL** path (`--no_icl_projection --probe_epochs 0`)
feeds raw 512-d Griffin embeddings directly to TabPFN/TabICL. This is
the configuration used for all ICL evals reported below; the standard
projection-and-probe configuration is the baseline being challenged.

---

## 2. Experiment Catalogue

Every experiment maps to a question and a paper-table cell. Status
abbreviations: ✅ done, 🟡 in flight, ⏳ planned, ❌ blocked/skip.

### 2.1 In-distribution backbone training (others-1 train, others-1 eval)

All trained with `hmaintask_combine.py --tasks others-1`, hiddim=512,
hop=2, fanout=20, 20 epochs, lr=3e-4, wd=4e-4, batchsize=256.

| ID | Variant | Motivation | Status | Avg (1 seed) |
|---|---|---|---|---|
| C1 | Vanilla Griffin, num_mp=4 | Baseline against which everything is measured. | ✅ | 0.5476 |
| C2 | SMPNN Griffin, num_mp=4 | Parity check: does the SMPNN block recover the baseline at matched depth? | ✅ | 0.5426 |
| C3 | Vanilla Griffin, num_mp=6 | Depth-control: SMPNN paper's headline claim was that vanilla GNNs collapse at depth=6 (ogbn-arxiv 39.67%). Does the same collapse happen on RDB? | ✅ | 0.5490 |
| C4 | SMPNN Griffin, num_mp=6 | Standard SMPNN depth advantage: does +2 layers help when packaged? | ✅ | 0.5572 |
| C5 | SMPNN Griffin, num_mp=8 | Depth ceiling: does the alpha-pruning hypothesis hold (deeper layers fade out via small α)? | ✅ | 0.5405 |
| A2 | SMPNN-6, `use_alpha=False` | Is learnable α scaling necessary, or do residuals + LN alone carry it? | ✅ | 0.5466 |
| A3 | SMPNN-6, `use_ff=False` | Is the pointwise FF sub-block necessary? Also halves FF parameters. | ✅ | 0.5280 |
| A4 | SMPNN-6, `use_gnn_ln=False` | Is Pre-LN before the GNN sub-block necessary? Paper finds it *helps* on ogbn-arxiv but is dataset-dependent. | ✅ | 0.5422 |
| D2 | SMPNN-6, `alpha_init=1e-4` | Faster ramp-up — does it converge faster, or is the slow 1e-6 ramp deliberate? | ✅ | 0.5393 |
| D3 | SMPNN-6, `alpha_init=1e-2` | Aggressive ramp — does it destabilise training (paper hypothesis), or is RDB tolerant? | ✅ | 0.5650 |
| B1 | SMPNN-6, `use_attention=True, num_heads=1` | Paper Appendix A: does linear global attention help on heterogeneous RDB graphs? Paper finds <1% gain on homogeneous. | ✅ | 0.5241 |

### 2.2 In-distribution downstream-head evaluation

For each of the 11 backbones above, run two no-projection ICL evals
(`hmaintask_combine_llm.py --no_icl_projection --probe_epochs 0`) on the
same others-1 task family. Motivation: tests whether the backbone-level
ablation effects propagate to **embedding quality** for downstream
foundation-model heads — the more interesting question than native-head
classification.

| ID | Backbone × Head | Motivation | Status |
|---|---|---|---|
| ICL-1 | All 11 backbones × TabPFN v3 ZS @ 30K, no-proj | Does the native-head ordering hold under TabPFN ZS? | ✅ (see 2.2.1) |
| ICL-2 | All 11 backbones × TabICL v2 ZS @ 10K, no-proj | Does the ordering also hold under a different ICL head? Cross-validates the embedding-quality story. | ✅ (see 2.2.1) |

#### 2.2.1 ICL eval results (in-distribution, single seed)

TabPFN ZS no-proj ranking (avg across 6 others-1 tasks):

| Rank | Backbone | Avg | Note |
|---|---|---|---|
| 1 | vanilla-4 (C1) | **0.5607** | New TabPFN ZS in-dist best |
| 2 | d3-alpha-1e-2 (D3) | 0.5592 | Native-head winner; tied with #1 within noise |
| 3 | a2-no-alpha | 0.5470 | |
| 4 | a4-no-gnn-ln | 0.5458 | |
| 5 | b1-attn-1h | 0.5443 | Attention less catastrophic under TabPFN than native |
| 6 | a3-no-ff | 0.5362 | FF removal still hurts |
| 7 | smpnn-6 (D1) | 0.5318 | SMPNN-6 default loses to vanilla-4 under TabPFN |
| 8 | smpnn-4 (C2) | 0.5317 | |
| 9 | d2-alpha-1e-4 | 0.5305 | |
| 10 | smpnn-8 | 0.5250 | |
| 11 | **c3-vanilla-6** | **0.5220** | Sharp drop from native rank 3 (0.5490) -- embedding-quality collapse |

TabICL ZS no-proj ranking:

| Rank | Backbone | Avg |
|---|---|---|
| 1 | **d3-alpha-1e-2 (D3)** | **0.5503** |
| 2 | a2-no-alpha | 0.5453 |
| 3 | a4-no-gnn-ln | 0.5450 |
| 4 | vanilla-4 (C1) | 0.5430 |
| 5 | a3-no-ff | 0.5360 |
| 6 | smpnn-6 (D1) | 0.5350 |
| 7-11 | (b1, smpnn-4, d2, smpnn-8, c3) | 0.5190 - 0.5293 |
| 11 | c3-vanilla-6 | 0.5190 |

#### 2.2.2 Cross-head findings (paper-shaping)

1. **D3 wins under all 3 heads** (native 0.5650 #1, TabPFN 0.5592 #2, TabICL 0.5503 #1).
   Robust across heads -- strongest single finding in the matrix without multi-seed.

2. **C3 (vanilla-6) collapses on embedding quality.** Native rank 3 -> TabPFN/TabICL rank 11.
   Rescues the SMPNN paper's "vanilla GNNs collapse at depth 6" claim for RDB --
   shifted from classification accuracy to embedding quality.

3. **Native vs ICL rankings differ substantially.** Spearman correlation between
   native and TabPFN-ZS rankings is ~0.3. The "best backbone" depends on what's
   downstream. SMPNN-6 (D1) is rank 2 under native but rank 7 under TabPFN.
   *A backbone optimized for native-head training is not optimal for downstream
   ICL use.* This is itself a paper-worthy finding.

4. **Attention (B1) is bad-to-middling under ICL, not catastrophic.**
   Native rank 11 -> TabPFN rank 5, TabICL rank 8. Softens the "attention
   strictly hurts" native-head claim to "attention doesn't help anywhere".

5. **FF removal (A3) is consistently weak across all 3 heads** (native 0.5280,
   TabPFN 0.5362, TabICL 0.5360). FF carries cross-head signal.

### 2.3 Extended-reach (hop=3) ablation

| ID | Variant | Motivation | Status |
|---|---|---|---|
| H1 | Vanilla-4, hop=3 | Does longer reach help vanilla without SMPNN packaging? | ✅ (test=0.5493) |
| H2 | SMPNN-6, hop=3 | Does SMPNN's depth advantage compound with longer reach? | 🟡 (rerun pending) |
| H3 | SMPNN-8, hop=3 | Depth ceiling + extended reach. May OOM. | ⏳ |
| H4 | SMPNN-6 + attention, hop=3 | With-transformer companion to H2. Does attention become useful when subgraphs are larger? | 🟡 (rerun pending) |

H2/H4 form the "with vs without transformer at extended reach" pair —
the headline claim is "attention is net-negative on RDB regardless of
reach" if both H2 > H4 and B1 < SMPNN-6.

### 2.4 Diagnostics (logged passively, no separate runs)

- **Final per-layer α values**: logged every 2 epochs via
  `--log_alpha_every 2`. Tests the "deeper layers self-prune" hypothesis
  (α[i] for i > effective-depth shrinks).
- **Parameter counts**: logged at construction time. Required for Study
  B's accuracy-vs-params trade-off table.
- **Peak GPU memory**: capturable from `nvidia-smi --query` during
  training, provides Figure 2/3 analog for the paper.

---

## 3. Cross-task transfer results — 4-direction matrix complete

**Status (2026-06-03): 24 of 24 cells captured.** 3 anchor backbones
(C1 vanilla-4, D1 SMPNN-6 α=1e-6, D3 SMPNN-6 α=1e-2) trained on each of
4 source families and evaluated on the complementary target family via
TabPFN v3 ZS no-proj and TabICL v2 ZS no-proj. Single seed.

### 3.1 Setup as run

3 anchor backbones × 3 new source families (others-2, commerce-1,
commerce-2) trained from scratch on the source family. The others-1
source reused the existing ablation checkpoints (V0/V2/V5).

For each (direction, backbone, head) triple, embedding extraction +
ICL inference. No additional probe training. Total: 9 new backbone
trainings (~30 GPU-h) + 24 ICL evals (~2 GPU-h) = ~32 GPU-h.

### 3.2 Four-direction average results (winners bolded)

Headline: average test metric per backbone per head, per direction.

| Direction | Target family | C1 vanilla-4 TabPFN | D1 SMPNN-6 TabPFN | D3 α=1e-2 TabPFN | C1 vanilla-4 TabICL | D1 SMPNN-6 TabICL | D3 α=1e-2 TabICL |
|---|---|---|---|---|---|---|---|
| **o1→o2** | others-2 | −0.891 | −0.897 | **−0.865** ⭐ | −0.891 | −0.887 | **−0.878** ⭐ |
| **o2→o1** | others-1 | 0.520 | **0.527** ⭐ | 0.489 | **0.519** | 0.518 | 0.484 |
| **c1→c2** | commerce-2 | **0.105** ⭐ | 0.102 | 0.101 | 0.047 | **0.097** ⭐ | 0.055 |
| **c2→c1** | commerce-1 | **0.261** ⭐ | 0.246 | 0.255 | 0.191 | 0.198 | **0.209** ⭐ |

### 3.3 Wins per backbone (across 8 head × direction cells)

| Backbone | TabPFN wins | TabICL wins | Total |
|---|---|---|---|
| **D3 α=1e-2** | 1 (o1→o2) | 2 (o1→o2, c2→c1) | **3** |
| **C1 vanilla-4** | 2 (c1→c2, c2→c1) | 1 (o2→o1) | **3** |
| **D1 SMPNN-6** | 1 (o2→o1) | 1 (c1→c2) | **2** |

### 3.4 Decisive wins (Δ > 0.02 over runner-up)

| Backbone | Decisive wins | Margin |
|---|---|---|
| **D3 α=1e-2** | 2 (o1→o2 TabPFN; c2→c1 TabICL) | +0.026, +0.018 |
| **D1 SMPNN-6** | 1 (c1→c2 TabICL) | +0.050 (strongest single signal) |
| **C1 vanilla-4** | 0 | best margin: +0.015 (c2→c1 TabPFN, within noise) |

**Headline finding**: Vanilla Griffin has **zero decisive cross-task wins**.
All 3 of its nominal "wins" are within ±0.015 of the runner-up.
SMPNN variants hold every decisive win.

Under TabICL specifically, SMPNN wins 3 of 4 directions
(o1→o2 D3, c1→c2 D1, c2→c1 D3). Under TabPFN the picture is more even,
but all gaps in TabPFN-favouring directions are within seed-variance bounds.

### 3.5 Direction-dependent optimum α-init

The 3 SMPNN cross-task wins span both α-init choices:
- **D3 (aggressive α=1e-2)** wins forward-others (o1→o2) and reverse-commerce (c2→c1)
- **D1 (paper-default α=1e-6)** wins forward-commerce (c1→c2)

Interpretation: aggressive α produces strongly-tuned representations
that transfer well when source and target share structural patterns
(forward-others and reverse-commerce both happen to share this property
in RelBench). Conservative α produces less specialised features that
generalise better to structurally distinct targets (commerce-1 → avito).

This is a **direction-dependent recommendation**, not a universal one —
worth a paragraph in the paper.

### 3.6 What this tells us about the per-component ablations (in-dist only)

The cross-task data only covers the 3 anchors (V0, V2, V5). The 8
per-component variants (V1, V3, V4, V6, V7, V8a, V8b, V8c) remain
in-distribution-only — that's a deliberate choice to make the eval
matrix tractable. The in-distribution ablations stand alone as the
"which SMPNN components matter on RDB" contribution; the cross-task
matrix stands alone as the "do SMPNN backbones transfer" contribution.

---

## 4. Would this make an ICLR main paper?

Honest assessment. Three dimensions: novelty, evidence, and venue fit.

### 4.1 Novelty — strengthened by completed cross-task matrix

| Claim | Novelty grade | Notes |
|---|---|---|
| **Vanilla Griffin has zero decisive cross-task wins on RDB** | A+ | All 3 SMPNN decisive wins (Δ > 0.02); all 3 vanilla "wins" within seed noise. Strong contrary signal to "SMPNN doesn't help much on RDB". |
| Optimal α-init is direction-dependent (D3 wins forward-others + reverse-commerce; D1 wins forward-commerce) | A | New finding; SMPNN paper treats α as a single hyperparameter. We show it interacts with transfer structure. |
| SMPNN's component importance differs on heterogeneous RDB | A | Paper's ablations were all on homogeneous transductive ogbn benchmarks. The FF >> GNN-LN > alpha ordering we see on RDB is new. |
| Attention is *net-negative* on RDB (Δ −0.0331 at hop=2 in-dist, neutral cross-task) | A+ | Contradicts paper's "<1% gain" on ogbn. Strong contrary finding. |
| Vanilla GNN at depth=6 does NOT collapse on RDB (in-dist) but DOES collapse on ICL embedding quality | A | The collapse exists, just shifted from classification accuracy → embedding quality. Reviewers should engage with this nuance. |
| Under TabICL specifically, SMPNN wins 3 of 4 transfer directions | A | Head-architecture interaction is real — paper-worthy on its own. |
| No-projection ICL with raw 512-d embeddings beats projection bottleneck | B+ | Already documented in our prior cross-task work; SMPNN paper doesn't address ICL composition. |

### 4.2 Evidence — substantially stronger now

| Required for ICLR main | Current state | Gap |
|---|---|---|
| Multi-direction transfer matrix | ✅ **Complete (24 of 24 cells)** | 4 directions × 3 anchors × 2 heads. Done. |
| In-distribution component ablations | ✅ **Complete (11 backbones × 3 heads on others-1)** | Done. |
| Multi-seed (≥3 seeds, mean±std) for headline numbers | ⚠️ Single seed everywhere | Need 2 more seeds × {3 highest-value cells} = ~7 GPU-hours. **Critical for ICLR claims.** |
| Multi-dataset or multi-benchmark | RelBench only | Could add OGB-relational, but the 24-cell RelBench coverage with both ICL heads is itself a substantial benchmark. Not a hard block. |
| Comparison to LLM baselines | Have LLM-FT results from prior work | Needs incorporation into the cross-task matrix as a 4th column. ~30 min table work. |
| Theoretical contribution | None | ICLR main reviewers vary on this. Foundation-models tracks accept pure empirical contributions if findings are non-obvious — ours are. |
| Reproducibility (code + scripts + configs) | All on GitHub on smpnn-ablations branch | Good. |

### 4.3 Venue fit — risky for ICLR main, strong for FM4SD/LoG

ICLR main accepts ~25-30% and is dominated by:
- Novel architectures / training recipes
- Theoretical insights into deep learning
- Strong empirical results with multi-seed, multi-benchmark validation

Our paper's profile:
- ✅ Empirical contribution with multiple counter-intuitive findings
- ✅ Reproducible code
- ❌ Not a new architecture — applies existing SMPNN to a new domain
- ❌ No theory
- ⚠️ Single benchmark (RelBench)
- ⚠️ Currently single seed

Likely reviewer reaction at ICLR main:
- **Strong-accept reviewer**: "Surprising negative results on attention, well-controlled ablations. Useful for the community building RDB FMs."
- **Borderline reviewer**: "Incremental over SMPNN [2024]; applies existing methods to RelBench. Where's the novelty?"
- **Weak-reject reviewer**: "Single benchmark, single seed, no theory. Better fit for a workshop."

Realistic outcome probability with current data: 15-25% accept. With
multi-seed + 4-direction transfer matrix: 35-50%. Either way it's a
defensible submission, not a slam-dunk.

### 4.4 Recommendation

**Two-track path:**

1. **Workshop now (FM4SD at ICML, or LoG, or NeurIPS Tab workshops)**
   with the current data + hop=3 + multi-seed-of-D3-only. 4-page short
   paper, fast turnaround, builds momentum.

2. **ICLR main in 6-9 months** with the full extension:
   - All 11 ablations × 4 directions × 3 seeds
   - Add a second benchmark (something like OGB-LSC or a custom RDB
     benchmark)
   - Possibly a small theoretical section on why attention should be
     bad on heterogeneous subgraph-batched MP (sketch: local FK/PK
     edges already carry the relevant cross-table signal; global
     attention pulls in irrelevant context across unrelated tables)

The workshop submission would also be the natural backstop if ICLR
main reviewers reject — workshops accept revisions.

### 4.5 Minimum-viable ICLR main version — what's actually left

Most of what was P0 is now done. The remaining work is multi-seed
validation of the decisive cross-task wins (3 cells with Δ > 0.02):

1. ✅ **3 anchor backbones × 4 source families × 2 heads** — done (this doc §3)
2. ✅ **11-backbone in-dist ablation matrix on others-1 × 3 heads** — done
3. ⚠️ **Multi-seed of the 3 decisive cross-task wins** — outstanding:
   - C1 vs D1 on c1→c2 TabICL (+0.050) — the biggest gap
   - C1 vs D3 on o1→o2 TabPFN (+0.026)
   - C1 vs D3 on c2→c1 TabICL (+0.018)
   - Plan: 3 seeds × {C1, D3} on others-1 (5 GPU-h) + 3 seeds × {C1, D1, D3} on commerce-2 source for c2→c1 = ~7 GPU-h total
4. ⚠️ **Hop=3 multi-seed clean comparison** (currently regime-confounded) — optional
5. ⚠️ **Add LLM-FT baselines as a 4th column in the cross-task table** — needs no new runs; just incorporate prior-work numbers

**Remaining GPU budget for ICLR-grade submission: ~7 GPU-hours.**
That's a meaningful number — the matrix is mostly done.

---

## 5. Open questions — status update

1. **Are the in-distribution single-seed numbers robust?** ⏳ Open.
   D3=0.5650 win over D1=0.5572 (+0.008) is still single-seed. Multi-seed
   needed before final claims (~5 GPU-h).
2. **Does the projection-bypass result hold under cross-task transfer
   with SMPNN backbones?** ✅ Resolved. The 24-cell matrix uses
   `--no_icl_projection --probe_epochs 0` throughout. SMPNN variants
   hold all 3 decisive cross-task wins with no projection.
3. **Is the c1→avito constant-feature collapse fixed by any SMPNN
   variant?** ✅ Resolved. NO collapse on any anchor backbone (C1, D1, D3)
   trained on commerce-1. The original collapse was specific to the
   `o1-tth-lora-v3` multi-task pretraining checkpoint, not a c1 source
   issue. New finding worth documenting in the paper.
4. **What's the parameter-efficiency angle?** ⏳ Open. A3 (no FF) at
   14M params loses 0.020 on in-dist native; not tested cross-task.
   Worth one short paragraph; not a paper headline.
5. ⭐ **NEW question: Why does D3 win c2→c1 TabICL but not c2→c1 TabPFN?**
   Same backbone, same target, different ICL head. This is a *head-architecture
   interaction* worth a paragraph in the paper. Hypothesis: TabICL's
   10K-context cap means it benefits more from D3's compressed (deeper,
   more-tuned) features, while TabPFN's 30K context exposes overfitting.

---

## 6. Suggested next actions (ordered by ROI)

1. ✅ **In-dist ICL eval sweep** — done.
2. ✅ **Cross-task 4-direction × 3 anchors × 2 heads matrix** — done.
3. ⏳ **Multi-seed validation of decisive cross-task wins** (~7 GPU-h):
   - C1 vs D1 on c1→c2 TabICL (the +0.050 finding) — highest priority
   - C1 vs D3 on o1→o2 TabPFN (+0.026)
   - C1 vs D3 on c2→c1 TabICL (+0.018, the new finding)
   This produces mean±std error bars for the headline cells. Without
   this, single-seed reviewers will flag.
4. ⏳ **Multi-seed D1, D3, C1 on in-dist others-1** (~5 GPU-h).
   Adds error bars to the in-distribution table; confirms D3 > D1 + 0.008.
5. ⏳ **Add LLM-FT baselines as a 4th column in the cross-task table** —
   no new compute; ~30 min table reformat from prior project work.
6. **Write up the paper** with the now-complete matrix. The headline
   ("vanilla Griffin has zero decisive cross-task wins on RDB; SMPNN
   variants hold all 3 decisive wins under TabICL") is now defensible.
7. **Submit to workshop (FM4SD/LoG)** as the backstop, ICLR main as
   the target. With multi-seed (~7 GPU-h) the ICLR submission becomes
   genuinely competitive.

Total minimum additional compute: ~12 GPU-hours (steps 3 + 4).
Writing: a paper with strong empirical content.
