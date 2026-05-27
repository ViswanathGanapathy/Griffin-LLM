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
| ICL-1 | All 11 backbones × TabPFN v3 ZS @ 30K, no-proj | Does the native-head ordering hold under TabPFN ZS? | 🟡 |
| ICL-2 | All 11 backbones × TabICL v2 ZS @ 10K, no-proj | Does the ordering also hold under a different ICL head? Cross-validates the embedding-quality story. | 🟡 |

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

## 3. Cross-task transfer experiments (next phase)

The in-distribution experiments answer "does ablation X improve backbone
quality on the task family it was trained on?" The cross-task
experiments answer "do those improvements propagate to unseen RDB task
families?" — the more impactful claim for a transfer / foundation-model
paper.

### 3.1 Setup

Reuse the existing 11 in-distribution backbone checkpoints (Section 2.1)
trained on the source task family. For each target direction, evaluate
all 11 backbones with TabPFN v3 ZS no-proj and TabICL v2 ZS no-proj on
the target task family. No additional training — just embedding
extraction + ICL inference.

Wall-time estimate: 11 backbones × 2 heads × 4 directions = 88 eval runs,
~5 min each = ~7.5 GPU-hours total. Trivially parallelisable across GPUs.

### 3.2 Four-direction matrix

| Direction | Source (backbone trained on) | Target (eval tasks) | Why this matters |
|---|---|---|---|
| **o1→o2** | others-1 (f1, stack, virus) | others-2 (airbnb, trial-site, trial-adverse, trial-outcome, talkingdata, telstra) | Standard cross-family transfer. The original no-projection paper-bests are here — does adding SMPNN help further? |
| **o2→o1** | others-2 | others-1 | Reverse direction. Tests if the projection-bypass finding is symmetric, or direction-dependent. |
| **c1→c2** | commerce-1 (hm, retailrocket, seznam) | commerce-2 (avito, diginetica, ...) | Within the commerce cluster. Last attempt hit the c1→avito constant-feature collapse — does any SMPNN variant resist this collapse? |
| **c2→c1** | commerce-2 | commerce-1 | Reverse commerce direction. Robust under the prior no-proj recipe; does SMPNN add value? |

### 3.3 Per-direction script plan

For each direction, one master eval script:

```
run_smpnn_xtask_o1_to_o2.sh    # uses checkpoints/smpnn-ablation-* + smpnn-depth-*
run_smpnn_xtask_o2_to_o1.sh    # NEEDS new o2 backbones first!
run_smpnn_xtask_c1_to_c2.sh    # NEEDS new c1 backbones first!
run_smpnn_xtask_c2_to_c1.sh    # NEEDS new c2 backbones first!
```

**Important blocker:** the existing 11 backbones are all trained on
others-1. For the o2→o1, c1→c2, c2→c1 directions, we need to retrain
the 11 ablations on those source task families. That's another 11 × 3 =
33 training runs. Wall time ~22 GPU-hours per task family on a single A100
(roughly the same as the others-1 sweep took), so ~66 GPU-hours total.

If GPU budget is tight, the **minimum** version of this phase is:
- Train only **SMPNN-6 (D1)** on each of {others-2, commerce-1, commerce-2}
  → 3 additional backbone trainings, ~6 GPU-hours
- Eval all 4 directions × 2 heads × 1 backbone = 8 eval runs
- Loses the per-ablation-component cross-task story but keeps the
  headline transfer matrix.

The **full** version (11 ablations × 4 source families) is the proper
ICLR table.

### 3.4 What each cell in the 4-direction × 11-backbone × 2-head matrix tells us

- **D3 (alpha=1e-2) winning the in-dist sweep** — does it also win
  cross-task? If yes, it's a robust finding worth multi-seeding.
- **B1 (attention) losing in-dist** — does attention become *useful*
  when generalising to unseen task families? Hypothesis: maybe yes,
  because global attention could help with distribution shift. If no,
  the "attention bad on RDB" claim becomes much stronger.
- **A3 (no FF) losing badly in-dist** — confirms FF carries cross-task
  signal too, or only in-distribution?
- **C3 (vanilla-6) being competitive in-dist** — does the gap to SMPNN-6
  widen under cross-task transfer? If yes, SMPNN's value is more about
  generalisation than in-distribution accuracy.

---

## 4. Would this make an ICLR main paper?

Honest assessment. Three dimensions: novelty, evidence, and venue fit.

### 4.1 Novelty — borderline-strong

| Claim | Novelty grade | Notes |
|---|---|---|
| SMPNN's component importance differs on heterogeneous RDB | A | Paper's ablations were all on homogeneous transductive ogbn benchmarks. The FF >> GNN-LN > alpha ordering we see on RDB is new. |
| Attention is *net-negative* on RDB (Δ −0.0331 at hop=2) | A+ | Contradicts paper's "<1% gain" on ogbn. Strong contrary finding. |
| Vanilla GNN at depth=6 does NOT collapse on RDB | A | Paper had ogbn-arxiv crashing to 39.67% at depth=6. Heterogeneity + subgraph batching naturally limit oversmoothing. |
| Aggressive α=1e-2 init beats paper default 1e-6 | B | Single-seed result; could be noise. Needs multi-seed to claim. |
| No-projection ICL with raw 512-d embeddings beats projection bottleneck | B+ | Already documented in our prior cross-task work; SMPNN paper doesn't address ICL composition. |

### 4.2 Evidence — currently weak, fixable

| Required for ICLR main | Current state | Gap |
|---|---|---|
| Multi-seed (≥3 seeds, mean±std) for headline numbers | 1 seed only | Need 2 more seeds × ~11 runs × ~30 min = ~11 GPU-hours |
| Multi-direction transfer matrix | 1 direction (others-1 in-dist) | Need 3 more source families: ~22 GPU-hours per family if running 11 ablations each, or ~6 if running only the headline backbone |
| Multi-dataset or multi-benchmark | RelBench only | Could add OGB-relational or another RDB benchmark, but not strictly required if RelBench coverage is good (it is — 4 directions × 6+ tasks). |
| Comparison to LLM baselines | Have LLM-FT results from prior work; need to harmonise into the same table | Light lift — table reformat |
| Ablations | Strong: A/B/C/D suite of 11 backbones | Adequate |
| Theoretical contribution | None | ICLR main reviewers vary on this. Many accepted ICLR papers are pure empirical, especially in foundation-models tracks. Not a hard block. |
| Reproducibility (code + scripts + configs) | All on GitHub, branch ready for PR | Good |

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

### 4.5 Minimum-viable ICLR main version (12-18 GPU-hours to complete)

If you're committed to going straight to ICLR main:

1. Pick **D3 (alpha=1e-2), SMPNN-6 (D1), and Vanilla-4 (C1)** as the three
   "headline backbones" worth promoting to multi-seed.
2. Run **3 seeds of each** on in-dist others-1 → 9 trainings, ~5 GPU-hours
3. Train the same 3 backbones on **the other 3 source families** (o2, c1, c2)
   → 9 trainings, ~6 GPU-hours
4. Run **TabPFN no-proj eval in all 4 directions × 3 backbones × 3 seeds**
   → 36 evals, ~3 GPU-hours
5. Keep the broader 8 ablations (A/B/D) as in-dist-only, 1 seed each (already done)

This produces:
- Headline Table: 3 backbones × 4 directions × 2 heads × 3 seeds = 72
  cells with mean±std
- Ablation Table: 11 backbones × 1 direction (in-dist) × 1 seed (already
  in hand)
- Hop=3 Table: 4 cells (vanilla-4-hop3, SMPNN-6-hop3, SMPNN-6-hop3-attn,
  optionally SMPNN-8-hop3)

That's defensible for ICLR main. ~14 GPU-hours of additional compute.

---

## 5. Open questions to resolve before any submission

1. **Are the in-distribution single-seed numbers robust?** Especially
   the D3=0.5650 win over D1=0.5572 (+0.008 avg). Run 3 seeds of D1, D3,
   C1 to get a variance baseline.
2. **Does the projection-bypass result hold under cross-task transfer
   when paired with SMPNN backbones?** All 5 of our prior project-bests
   were either vanilla-4 or SMPNN-6 — but with the original probe
   pipeline, not the no-projection one. Need to recompose: SMPNN-6 (D1)
   + TabPFN no-proj + cross-task.
3. **Is the c1→avito constant-feature collapse fixed by any SMPNN
   variant?** D3's aggressive alpha or A3's removed FF might produce
   non-collapsing features. Single eval per backbone would tell us.
4. **What's the parameter-efficiency angle?** A3 (no FF) at 14M params
   beats vanilla-4 at 11M while losing 0.020 on avg — is that an
   acceptable trade-off for a deployment claim?

---

## 6. Suggested next actions (ordered by ROI)

1. **Let current ICL eval sweep complete** (in flight). Will resolve open
   question #2 in-dist.
2. **Multi-seed D1, D3, C1 in-dist** (3 runs each, ~5 GPU-hours).
   Resolves open question #1; if D3 > D1 holds with error bars, it's
   the new project-best to advocate for in any paper.
3. **Train D1 (SMPNN-6) on the 3 other source families** (others-2,
   commerce-1, commerce-2). ~6 GPU-hours. This is the minimum to enable
   the 4-direction transfer matrix.
4. **Run the 4-direction × {D1, C1} × {TabPFN, TabICL} eval matrix**.
   ~3 GPU-hours.
5. **Write up workshop short paper** (4 pages, FM4SD/LoG). This is the
   natural backstop and provides a forcing function for organising the
   results.
6. **Decide whether to extend to ICLR main** based on whether multi-seed
   D3 survives and whether cross-task results corroborate the in-dist
   findings.

Total minimum additional compute: ~14 GPU-hours. Total minimum
additional writing: 4-page workshop draft.
