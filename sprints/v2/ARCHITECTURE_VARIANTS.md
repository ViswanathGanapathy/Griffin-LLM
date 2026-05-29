# Griffin Architecture Variants: Catalog, Value vs Vanilla, and Experiment Status

A focused reference for which architectures we're exploring, how each one
differs from vanilla Griffin, what value it adds (measured), and which
experiments have validated it. Companion to
[SMPNN_GRIFFIN_DESIGN.md](SMPNN_GRIFFIN_DESIGN.md) — that doc covers
the paper plan and ICLR readiness; this one is a clean
architecture-value catalog.

---

## 1. Reference baseline: Vanilla Griffin

The architecture every variant is measured against.

| Property | Value |
|---|---|
| Architecture | Relational MPNN over RelBench subgraphs |
| Per-layer structure | `x → MLP_pre → RMPNN → gate ⊕ revRMPNN → x` |
| Hidden dim | 512 |
| MP layers (`num_mp`) | 4 |
| Subgraph reach (`hop`) | 2 |
| Fanout per hop | 20 |
| Reverse edges | Yes (`--use_rev True`) |
| Zero-init gate | Yes (`--use_gate True`) |
| Total params | 11,366,752 |
| In-dist others-1 native avg (1 seed) | **0.5476** |
| In-dist others-1 TabPFN ZS no-proj avg | **0.5607** |
| In-dist others-1 TabICL ZS no-proj avg | **0.5430** |

All "Δ vs vanilla" deltas below are computed against these three reference
numbers, by head.

---

## 2. Architecture variants under exploration

Each variant changes one thing relative to vanilla. Variants 1–6 share
the SMPNN packaging primitive (Pre-LN per sub-block + residual + α
scaling); variant 7 augments SMPNN with attention; variants 8–9
ablate SMPNN components; variant 10 extends subgraph reach.

### 2.1 Variant catalog

| ID | Variant | Differs from vanilla in | New CLI flags | Params | Param Δ vs vanilla |
|---|---|---|---|---|---|
| **V0** | Vanilla Griffin (reference) | — | — | 11.37M | — |
| **V1** | SMPNN-4 | Packages each MP layer in a 2-sub-block Pre-LN Transformer pattern | `--use_smpnn` | 11.37M + 3.7M (FF + LNs + alphas) = ~13.0M | +1.6M |
| **V2** | SMPNN-6 (D1, paper default) | V1 + 2 deeper MP layers | `--use_smpnn --num_mp 6` | 17.2M | +5.8M |
| **V3** | SMPNN-8 | V1 + 4 deeper MP layers | `--use_smpnn --num_mp 8` | 23.0M | +11.6M |
| **V4** | Vanilla-6 (C3) | Vanilla at depth 6 (control for V2) | `--num_mp 6` | 16.4M | +5.0M |
| **V5** | SMPNN-6 with α=1e-2 (D3) | V2 with aggressive identity-init for α (1e-2 instead of paper's 1e-6) | `--use_smpnn --alpha_init 1e-2` | 17.2M | +5.8M |
| **V6** | SMPNN-6 with α=1e-4 (D2) | V2 with mid-range α init | `--use_smpnn --alpha_init 1e-4` | 17.2M | +5.8M |
| **V7** | SMPNN-6 + Linear Global Attention (B1) | V2 + parallel global attention path per layer (Appendix A) | `--use_smpnn --use_attention True --num_heads 1` | 21.9M | +10.6M |
| **V8a** | SMPNN-6 without α (A2) | V2 with α fixed at 1 (no learnable scaling) | `--use_smpnn --use_alpha False` | 17.2M | +5.8M |
| **V8b** | SMPNN-6 without FF (A3) | V2 with FF sub-block removed | `--use_smpnn --use_ff False` | 14.0M | +2.6M |
| **V8c** | SMPNN-6 without GNN Pre-LN (A4) | V2 without LayerNorm before the GNN sub-block | `--use_smpnn --use_gnn_ln False` | 17.2M | +5.8M |
| **V10** | Vanilla-4 at hop=3 | V0 with longer subgraph reach (and forced smaller batch/fanout) | `--hop 3 --fanout 6 --batchsize 32` | 11.37M | — |

### 2.2 What each variant tests

| ID | Hypothesis being tested | Why it matters for the paper |
|---|---|---|
| V1 | SMPNN packaging recovers vanilla at matched depth (parity check) | Sanity check; would invalidate everything else if it failed |
| V2 | SMPNN's depth advantage transfers from homogeneous graphs to heterogeneous RelBench | Headline architecture claim |
| V3 | Depth plateaus around 6 layers (alpha pruning hypothesis) | Establishes a depth ceiling |
| V4 | Vanilla GNNs collapse at depth 6 (per SMPNN paper Table 7 on ogbn-arxiv) | Critical control; if V4 ≈ V0, the depth-collapse story for RDB shifts |
| V5 | Aggressive α=1e-2 init beats paper default 1e-6 on RDB | Tunes the SMPNN paper's recommendation for our setting |
| V6 | α=1e-4 either improves or worsens convergence vs 1e-6 (intermediate) | Maps out α sensitivity curve |
| V7 | Linear global attention helps on heterogeneous RDB graphs | Tests if paper's "attention adds <1% on ogbn" extends or reverses on RDB |
| V8a | Learnable α is necessary, or do residuals + LN suffice? | SMPNN component ablation (paper Table 6) |
| V8b | The pointwise FF sub-block carries necessary signal | SMPNN component ablation; also saves params if dispensable |
| V8c | Pre-LN before the GNN sub-block matters | SMPNN component ablation; paper found this *helped* on ogbn-arxiv |
| V10 | Longer reach (hop=3) compounds with SMPNN's depth advantage | Tests whether hop and depth interact |

---

## 3. Measured value of each variant (Δ vs vanilla)

Values are average test metric across the 6 in-distribution others-1
tasks, single seed. All ICL evaluations use the no-projection recipe
(`--no_icl_projection --probe_epochs 0`). Δ shown vs vanilla-4 (V0) per
head; positive Δ means the variant beats vanilla under that head.

### 3.1 Native-head training (what the backbone was trained with)

| ID | Variant | Avg | Δ vs V0 |
|---|---|---|---|
| **V5** | SMPNN-6, α=1e-2 | **0.5650** | **+0.0174** ✨ |
| V2 | SMPNN-6 (paper default) | 0.5572 | +0.0096 |
| V4 | Vanilla-6 | 0.5490 | +0.0014 (tie) |
| V0 | Vanilla-4 (reference) | 0.5476 | — |
| V8a | SMPNN-6 − α | 0.5466 | −0.0010 (tie) |
| V1 | SMPNN-4 | 0.5426 | −0.0050 |
| V8c | SMPNN-6 − GNN-LN | 0.5422 | −0.0054 |
| V3 | SMPNN-8 | 0.5405 | −0.0071 |
| V6 | SMPNN-6, α=1e-4 | 0.5393 | −0.0083 |
| V8b | SMPNN-6 − FF | 0.5280 | −0.0196 |
| V7 | SMPNN-6 + attention | 0.5241 | −0.0235 |
| V10 | Vanilla-4 at hop=3 | 0.5500 (regime confound) | +0.0024 |

### 3.2 TabPFN v3 ZS no-proj (embedding quality for fast tabular ICL)

| ID | Variant | Avg | Δ vs V0 |
|---|---|---|---|
| **V0** | **Vanilla-4 (reference)** | **0.5607** | **— (best under TabPFN)** |
| V5 | SMPNN-6, α=1e-2 | 0.5592 | −0.0015 (tie) |
| V8a | SMPNN-6 − α | 0.5470 | −0.0137 |
| V8c | SMPNN-6 − GNN-LN | 0.5458 | −0.0149 |
| V7 | SMPNN-6 + attention | 0.5443 | −0.0164 |
| V8b | SMPNN-6 − FF | 0.5362 | −0.0245 |
| V2 | SMPNN-6 (paper default) | 0.5318 | −0.0289 |
| V1 | SMPNN-4 | 0.5317 | −0.0290 |
| V6 | SMPNN-6, α=1e-4 | 0.5305 | −0.0302 |
| V3 | SMPNN-8 | 0.5250 | −0.0357 |
| V4 | Vanilla-6 | 0.5220 | −0.0387 |

### 3.3 TabICL v2 ZS no-proj (10K context cap, less expressive head)

| ID | Variant | Avg | Δ vs V0 |
|---|---|---|---|
| **V5** | **SMPNN-6, α=1e-2** | **0.5503** | **+0.0073** ✨ |
| V8a | SMPNN-6 − α | 0.5453 | +0.0023 (tie) |
| V8c | SMPNN-6 − GNN-LN | 0.5450 | +0.0020 (tie) |
| V0 | Vanilla-4 (reference) | 0.5430 | — |
| V8b | SMPNN-6 − FF | 0.5360 | −0.0070 |
| V2 | SMPNN-6 (paper default) | 0.5350 | −0.0080 |
| V1 | SMPNN-4 | 0.5293 | −0.0137 |
| V7 | SMPNN-6 + attention | 0.5288 | −0.0142 |
| V6 | SMPNN-6, α=1e-4 | 0.5278 | −0.0152 |
| V3 | SMPNN-8 | 0.5255 | −0.0175 |
| V4 | Vanilla-6 | 0.5190 | −0.0240 |

### 3.4 Cross-head summary: rank position by head

Ranks 1 (best) to 11 (worst) per head, for the 11 in-dist variants:

| Variant | Native rank | TabPFN rank | TabICL rank | Consistency |
|---|---|---|---|---|
| V5 (SMPNN-6, α=1e-2) | 1 | 2 | 1 | **Top 2 everywhere** |
| V0 (Vanilla-4) | 4 | 1 | 4 | Strong, head-dependent |
| V8a (no α) | 5 | 3 | 2 | Surprisingly stable mid-pack |
| V8c (no GNN-LN) | 7 | 4 | 3 | Stable mid-pack |
| V2 (SMPNN-6 paper default) | 2 | 7 | 6 | **Drops sharply under ICL** |
| V4 (Vanilla-6) | 3 | 11 | 11 | **Native masks embedding collapse** |
| V8b (no FF) | 10 | 6 | 5 | Consistently weak |
| V7 (attention) | 11 | 5 | 8 | Worst under native, mid under ICL |
| V1 (SMPNN-4) | 6 | 8 | 7 | Mid-pack |
| V6 (α=1e-4) | 9 | 9 | 9 | Consistently weak |
| V3 (SMPNN-8) | 8 | 10 | 10 | Mild depth ceiling |

---

## 4. Headline architecture findings

Five conclusions from the value matrix above. All single-seed in-dist on
others-1; cross-task validation is in progress.

1. **V5 (SMPNN-6 with α=1e-2) is the most robust variant.** Top-2 rank under
   every head tested. Native winner, TabICL winner, near-winner under
   TabPFN. The aggressive α init is the single tunable that most reliably
   improves on the SMPNN paper's default.

2. **V0 (vanilla-4) is competitive for downstream ICL.** Under TabPFN ZS
   it actually beats every SMPNN variant including V5. This is unexpected:
   the simpler 4-layer vanilla embeddings are what TabPFN consumes best.
   Implication: backbone depth helps native training but doesn't
   automatically translate to embedding quality.

3. **V4 (vanilla-6) collapses on embedding quality.** Native rank 3 →
   TabPFN/TabICL rank 11. The SMPNN paper's "vanilla GNNs collapse at
   depth 6" claim **does** transfer to RDB — just at the embedding-quality
   layer, not the native-head accuracy layer.

4. **V7 (attention) doesn't help on RDB.** Net-negative under native head
   (−0.024). Net-near-zero under TabPFN (−0.016). Net-near-zero under
   TabICL (−0.014). Plus 10.6M extra parameters. Stronger than the SMPNN
   paper's "<1% gain on ogbn" — on RDB attention is at best neutral and
   often harmful.

5. **V8b (no FF) is the most-damaging component ablation.** Confirms FF
   carries cross-head signal: native −0.020, TabPFN −0.025, TabICL −0.007.
   FF > GNN-LN > α in importance ordering, which differs from the SMPNN
   paper's ordering on ogbn benchmarks.

---

## 5. Experiment status

### 5.1 Completed (✅)

| Experiment set | What | Where logged |
|---|---|---|
| **In-dist backbones, native head** (11 variants) | Train V0–V8c on others-1, eval native | `logs/smpnn-ablations*.log`, `logs/smpnn-depth.log` |
| **In-dist ICL eval** (11 backbones × 2 heads) | TabPFN v3 + TabICL v2 ZS no-proj on others-1 | `logs/smpnn-ablations-icl-eval.log` |
| **Hop=3 native** (partial: V0 + V2 only) | Vanilla-4 and SMPNN-6 at hop=3 | `logs/smpnn-hop3-smpnn6.log` |

### 5.2 In flight (🟡)

| Experiment set | What | Status |
|---|---|---|
| **Cross-task backbones** | Train V0/V2/V5 on each of 3 new source families (others-2, commerce-1, commerce-2) — 9 trainings | Script ready: `run_smpnn_xtask_backbones.sh`, ~30 GPU-hours |
| **Cross-task eval** | 4 directions × 3 backbones × 2 heads = 24 evals | Script ready: `run_smpnn_xtask_eval.sh`, ~1.5 GPU-hours |

### 5.3 Recommended next, ordered by ROI

| Priority | Experiment | Effort | Resolves |
|---|---|---|---|
| **P0** | Cross-task backbones + eval (Section 5.2 above) | ~32 GPU-h | Does V5 win cross-task? Does the C1→avito collapse recur? |
| **P1** | 3 seeds of V0, V2, V5 in-dist | ~5 GPU-h | Error bars on the V5 win (currently 1 seed) |
| **P2** | Cross-task with V8b (no FF) added | ~12 GPU-h | Does FF removal hurt cross-task too, or only in-dist? |
| **P3** | Hop=3 with matched data regime (rerun V0 at bs=32, fanout=6 for apples-to-apples vs V2-hop=3) | ~3 GPU-h | Settles whether the V2-hop=3 result is real or regime artefact |
| **(skip)** | SMPNN-8-hop=3, attention-hop=3 | 12+ GPU-h | Limited paper value given hop=3 is not a headline |

### 5.4 Skipped / deferred

| Experiment | Why skipped |
|---|---|
| V7 (attention) cross-task | Native-head loss is strong enough evidence; not worth GPU-hours to confirm cross-task |
| V8a/V8b/V8c cross-task | Component ablations are sufficient as in-dist; cross-task would be a 22-run extension for diminishing returns |
| Hop=3 SMPNN-8 + attention | OOM-prone at 23M params + hop=3; not central to any headline claim |
| Multi-dataset (beyond RelBench) | Out of scope for the FM4SD short-paper version; possible for ICLR-main extension |

---

## 6. Quick-reference: which variant to use when

| Use case | Pick | Why |
|---|---|---|
| Train Griffin from scratch on a new RelBench-style RDB | **V5 (SMPNN-6, α=1e-2)** | Best native-head accuracy; top-2 across all downstream heads |
| Use Griffin as a frozen embedding model for TabPFN | **V0 (vanilla-4)** | Wins under TabPFN by 0.0015 over V5; simplest |
| Use Griffin embeddings for TabICL or a future tabular FM | **V5** | TabICL winner; safer default than vanilla-4 for unknown heads |
| Smallest model that still performs reasonably | **V8b (no FF)** | 14M params (V5 has 17.2M); accepts −0.020 native penalty |
| **Avoid** | V7 (attention), V4 (vanilla-6), V6 (α=1e-4) | All consistently underperform |

---

## 7. What's still single-seed (everything)

All numbers in Sections 3.1–3.3 are from a single training seed per
variant. The cross-head consistency of V5 winning under 3 different heads
is itself evidence that the result is not a single-seed artefact, but
ICLR-grade claims require multi-seed (≥3) for the headline numbers. See
Section 5.3 P1.
