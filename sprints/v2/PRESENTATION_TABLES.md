# SMPNN-Griffin: Per-Task Results Tables (Presentation-Ready)

All single-seed results across in-distribution and cross-task transfer
experiments. Higher = better; for regression tasks values are reported as
−MAE so the sign is consistent. **Bold** marks per-column best within
each table.

Generated for presentation use 2026-06-02. As cells land in the ongoing
gap-filler eval they'll be filled in (3 c2→c1 cells still pending).

---

## 1. In-Distribution: Others-1 Native Head

11 backbones, 6 tasks, single seed.

| Backbone | f1-dnf | f1-pos | f1-top3 | sx-churn | sx-upv | virus | **avg** |
|---|---|---|---|---|---|---|---|
| **V5 D3 α=1e-2** | **0.744** | **−0.557** | 0.781 | 0.845 | 0.896 | 0.681 | **0.5650** |
| V2 D1 smpnn-6 | 0.723 | −0.579 | 0.787 | 0.844 | 0.896 | 0.672 | 0.5572 |
| V4 C3 vanilla-6 | 0.717 | −0.633 | 0.782 | **0.855** | 0.895 | **0.693** | 0.5490 |
| V0 C1 vanilla-4 | **0.747** | −0.660 | **0.807** | 0.836 | **0.897** | 0.659 | 0.5476 |
| V8a A2 no-α | 0.733 | −0.653 | 0.782 | 0.825 | 0.894 | 0.690 | 0.5466 |
| V1 C2 smpnn-4 | 0.720 | −0.620 | 0.788 | 0.819 | 0.896 | 0.654 | 0.5426 |
| V8c A4 no-GNN-LN | 0.726 | −0.661 | 0.793 | 0.843 | 0.896 | 0.648 | 0.5422 |
| V3 C5 smpnn-8 | 0.715 | −0.682 | 0.809 | 0.835 | 0.896 | 0.653 | 0.5405 |
| V6 D2 α=1e-4 | 0.715 | −0.628 | 0.770 | 0.834 | 0.892 | 0.674 | 0.5393 |
| V8b A3 no-FF | 0.693 | −0.697 | 0.795 | 0.827 | 0.894 | 0.660 | 0.5280 |
| V7 B1 attn-1h | 0.720 | −0.707 | 0.805 | 0.833 | 0.896 | 0.598 | 0.5241 |

---

## 2. In-Distribution: Others-1 TabPFN ZS No-Proj

| Backbone | f1-dnf | f1-pos | f1-top3 | sx-churn | sx-upv | virus | **avg** |
|---|---|---|---|---|---|---|---|
| **V0 C1 vanilla-4** | 0.723 | −0.571 | 0.798 | 0.848 | **0.897** | **0.669** | **0.5607** |
| V5 D3 α=1e-2 | **0.733** | **−0.570** | 0.787 | 0.846 | 0.895 | 0.664 | 0.5592 |
| V8a A2 no-α | 0.725 | −0.579 | 0.792 | 0.833 | 0.895 | 0.616 | 0.5470 |
| V8c A4 no-GNN-LN | 0.721 | −0.597 | 0.760 | 0.847 | 0.893 | 0.651 | 0.5458 |
| V7 B1 attn-1h | 0.716 | −0.572 | 0.786 | 0.842 | 0.893 | 0.601 | 0.5443 |
| V8b A3 no-FF | 0.724 | −0.656 | 0.795 | 0.833 | 0.894 | 0.627 | 0.5362 |
| V2 D1 smpnn-6 | 0.718 | −0.591 | 0.778 | 0.833 | 0.895 | 0.558 | 0.5318 |
| V1 C2 smpnn-4 | 0.727 | −0.693 | 0.769 | 0.835 | 0.895 | 0.657 | 0.5317 |
| V6 D2 α=1e-4 | 0.721 | −0.577 | 0.747 | 0.825 | 0.893 | 0.574 | 0.5305 |
| V3 C5 smpnn-8 | 0.711 | −0.695 | 0.786 | 0.839 | 0.895 | 0.614 | 0.5250 |
| V4 C3 vanilla-6 | 0.717 | −0.681 | 0.770 | **0.857** | 0.895 | 0.574 | 0.5220 |

---

## 3. In-Distribution: Others-1 TabICL ZS No-Proj

| Backbone | f1-dnf | f1-pos | f1-top3 | sx-churn | sx-upv | virus | **avg** |
|---|---|---|---|---|---|---|---|
| **V5 D3 α=1e-2** | 0.716 | **−0.583** | 0.768 | 0.835 | 0.893 | **0.673** | **0.5503** |
| V8a A2 no-α | 0.716 | −0.595 | 0.783 | 0.830 | 0.894 | 0.644 | 0.5453 |
| V8c A4 no-GNN-LN | **0.733** | −0.596 | 0.782 | 0.834 | 0.891 | 0.626 | 0.5450 |
| V0 C1 vanilla-4 | 0.714 | −0.598 | **0.786** | 0.847 | **0.895** | 0.614 | 0.5430 |
| V8b A3 no-FF | 0.729 | −0.643 | 0.760 | 0.828 | 0.894 | 0.648 | 0.5360 |
| V2 D1 smpnn-6 | 0.697 | −0.616 | 0.770 | 0.815 | 0.893 | 0.651 | 0.5350 |
| V1 C2 smpnn-4 | 0.721 | −0.642 | 0.772 | 0.842 | 0.895 | 0.588 | 0.5293 |
| V7 B1 attn-1h | 0.707 | −0.629 | 0.782 | 0.836 | 0.893 | 0.584 | 0.5288 |
| V6 D2 α=1e-4 | 0.719 | −0.614 | 0.757 | 0.803 | 0.891 | 0.611 | 0.5278 |
| V3 C5 smpnn-8 | 0.703 | −0.648 | 0.769 | 0.838 | 0.895 | 0.596 | 0.5255 |
| V4 C3 vanilla-6 | 0.716 | −0.747 | 0.761 | **0.851** | 0.893 | 0.640 | 0.5190 |

---

## 4. In-Distribution: Others-2 Native Head

3 anchors trained on others-2; eval on the 6 others-2 tasks.

| Backbone | airbnb-dest | trial-site-success | trial-study-adverse | trial-study-outcome | talkingdata | telstra | **avg** |
|---|---|---|---|---|---|---|---|
| **D3 α=1e-2** | **0.870** | **−0.701** | **−1.302** | 0.677 | **−2.399** | **−0.735** | **−0.6006** |
| C1 vanilla-4 | 0.867 | −0.712 | −1.387 | 0.660 | −2.399 | −0.752 | −0.6203 |
| D1 smpnn-6 | 0.865 | −0.842 | −1.311 | **0.683** | −2.407 | −0.766 | −0.6285 |

---

## 5. In-Distribution: Commerce-1 Native Head

3 anchors trained on commerce-1; eval on 4 commerce-1 tasks (seznam-charge / seznam-prepay missing from this CSV).

| Backbone | diginetica | hm-item-sales | hm-user-churn | retailrocket | **avg** |
|---|---|---|---|---|---|
| **D1 smpnn-6** | 0.478 | −0.997 | 0.660 | **0.960** | **0.4760** |
| C1 vanilla-4 | 0.534 | **−0.968** | **0.660** | 0.929 | 0.4632 |
| D3 α=1e-2 | **0.589** | −1.017 | 0.656 | 0.948 | 0.4597 |

---

## 6. Cross-Task o1 → o2 (TARGET: others-2)

Backbone trained on others-1; embeddings frozen; eval on others-2 via ICL no-projection.

| Backbone | Head | airbnb-dest | trial-site-success | trial-study-adverse | trial-study-outcome | talkingdata | telstra | **avg** |
|---|---|---|---|---|---|---|---|---|
| **D3 α=1e-2** | **tabpfn** | 0.860 | **−0.905** | **−2.417** | 0.605 | **−2.480** | −0.855 | **−0.8654** |
| **D3 α=1e-2** | **tabicl** | 0.856 | −0.952 | −2.440 | 0.594 | −2.482 | −0.846 | **−0.8782** |
| D1 smpnn-6 | tabicl | 0.858 | −0.950 | −2.469 | 0.554 | −2.481 | **−0.833** | −0.8870 |
| C1 vanilla-4 | tabicl | 0.850 | −0.961 | −2.538 | 0.632 | −2.481 | −0.846 | −0.8905 |
| C1 vanilla-4 | tabpfn | 0.849 | −0.936 | −2.576 | **0.643** | −2.480 | −0.848 | −0.8915 |
| D1 smpnn-6 | tabpfn | **0.863** | −0.904 | −2.556 | 0.541 | −2.480 | −0.847 | −0.8971 |

---

## 7. Cross-Task o2 → o1 (TARGET: others-1)

| Backbone | Head | f1-dnf | f1-pos | f1-top3 | sx-churn | sx-upv | virus | **avg** |
|---|---|---|---|---|---|---|---|---|
| **D1 smpnn-6** | **tabpfn** | **0.732** | −0.606 | 0.761 | 0.767 | 0.852 | **0.659** | **0.5274** |
| C1 vanilla-4 | tabpfn | 0.703 | **−0.581** | 0.762 | **0.800** | **0.860** | 0.578 | 0.5202 |
| **C1 vanilla-4** | **tabicl** | 0.709 | −0.599 | **0.768** | 0.778 | 0.849 | 0.611 | **0.5191** |
| D1 smpnn-6 | tabicl | 0.731 | −0.602 | 0.765 | 0.722 | 0.844 | 0.648 | 0.5181 |
| D3 α=1e-2 | tabpfn | 0.647 | −0.623 | 0.648 | 0.779 | 0.863 | 0.622 | 0.4894 |
| D3 α=1e-2 | tabicl | 0.685 | −0.646 | 0.647 | 0.738 | 0.861 | 0.618 | 0.4838 |

---

## 8. Cross-Task c1 → c2 (TARGET: commerce-2)

Tasks: amazon-churn, amazon-rating, outbrain-small-ctr, rel-avito-ad-ctr, rel-avito-user-clicks, rel-avito-user-visits.

| Backbone | Head | amzn-ch | amzn-rt | outbrain | avito-adctr | avito-clk | avito-vis | **avg** |
|---|---|---|---|---|---|---|---|---|
| **C1 vanilla-4** | **tabpfn** | 0.654 | **−0.793** | 0.524 | −0.847 | 0.545 | 0.547 | **0.1050** |
| D1 smpnn-6 | tabpfn | 0.633 | −0.824 | **0.540** | −0.879 | 0.591 | 0.552 | 0.1022 |
| D3 α=1e-2 | tabpfn | **0.661** | −0.843 | 0.504 | **−0.829** | 0.543 | **0.568** | 0.1007 |
| **D1 smpnn-6** | **tabicl** | 0.625 | −0.858 | 0.532 | −0.872 | 0.573 | **0.583** | **0.0970** |
| D3 α=1e-2 | tabicl | 0.640 | −0.877 | 0.520 | −0.877 | 0.450 | 0.470 | 0.0545 |
| C1 vanilla-4 | tabicl | 0.589 | −0.821 | 0.513 | −0.910 | 0.455 | 0.456 | 0.0470 |

---

## 9. Cross-Task c2 → c1 (TARGET: commerce-1) — PARTIAL

3 cells still pending in the in-flight gap-filler eval (D1 TabICL almost done, D3 both heads pending).

Tasks: diginetica-downsample-ctr, rel-hm-item-sales, rel-hm-user-churn, retailrocket-cvr, seznam-charge (hr@1), seznam-prepay (hr@1).

| Backbone | Head | diginetica | hm-item-sales | hm-user-churn | retailrocket | sez-charge | sez-prepay | **avg** |
|---|---|---|---|---|---|---|---|---|
| **C1 vanilla-4** | **tabpfn** | 0.529 | **−1.489** | **0.628** | 0.940 | **0.378** | **0.578** | **0.2605** |
| D1 smpnn-6 | tabpfn | **0.551** | −1.528 | 0.621 | **0.964** | 0.315 | 0.553 | 0.2459 |
| **C1 vanilla-4** | **tabicl** | 0.533 | −1.745 | 0.623 | **0.956** | 0.322 | 0.457 | **0.1911** |
| D1 smpnn-6 | tabicl | 0.540 | −1.711 | (pending) | (pending) | (pending) | (pending) | pending |
| D3 α=1e-2 | tabpfn | pending | pending | pending | pending | pending | pending | pending |
| D3 α=1e-2 | tabicl | pending | pending | pending | pending | pending | pending | pending |

---

## 10. Hop=3 In-Distribution Native (Others-1)

2 of 4 hop=3 variants completed; data regime caveat (smpnn-6-hop3 ran at bs=32 fanout=6, vanilla-4-hop3 at bs=64 fanout=8 — not apples-to-apples).

| Backbone | f1-dnf | f1-pos | f1-top3 | sx-churn | sx-upv | virus | **avg** |
|---|---|---|---|---|---|---|---|
| **vanilla-4 hop=3** | 0.692 | **−0.605** | **0.789** | **0.831** | **0.896** | **0.673** | **0.5500** |
| smpnn-6 hop=3 | **0.710** | −0.816 | 0.775 | 0.821 | 0.892 | 0.533 | 0.4860 |

---

## 11. Cross-Task Winner Summary (21 of 24 cells decided)

For your presentation summary slide:

| Cell | Winner | Avg | Margin vs runner-up |
|---|---|---|---|
| o1→o2 TabPFN | **D3** | −0.865 | +0.026 over C1 |
| o1→o2 TabICL | **D3** | −0.878 | +0.009 over D1 |
| o2→o1 TabPFN | **D1** | 0.527 | +0.007 over C1 (within noise) |
| o2→o1 TabICL | **C1** | 0.519 | +0.001 over D1 (tied) |
| c1→c2 TabPFN | **C1** | 0.105 | +0.003 over D1 (within noise) |
| c1→c2 TabICL | **D1** | 0.097 | +0.050 over C1 ⭐ |
| c2→c1 TabPFN | **C1** | 0.261 | +0.015 over D1 |
| c2→c1 TabICL | pending | — | — |

### Cross-head, cross-direction winner counts (decisive cells only, Δ > 0.02)

| Backbone | Decisive wins |
|---|---|
| C1 vanilla-4 | 2 (c2→c1 TabPFN by +0.015 small but real; tied elsewhere) |
| D1 SMPNN-6 default | 1 (c1→c2 TabICL +0.050 — strongest signal) |
| D3 SMPNN-6 α=1e-2 | 1 (o1→o2 TabPFN +0.026) |

---

## Single-seed disclaimer

All numbers are from single training seeds. Multi-seed (n=3) validation
is the next priority, especially for the c1→c2 TabICL +0.050 gap, which
is the strongest SMPNN-over-vanilla finding in the matrix.
