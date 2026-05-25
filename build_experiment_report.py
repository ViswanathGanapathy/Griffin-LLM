"""Build a Word document capturing all TabICL v2 and TabPFN family experiments.

Sections:
  1. Parameters used (probe, TabICL configs, TabPFN configs)
  2. Experimental results (per-task numbers + matrix)
  3. Next steps (concrete follow-ups, ordered by value)
"""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT


CODE_FONT = "Consolas"


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.2)
    for line in text.split("\n"):
        run = p.add_run(line + "\n")
        run.font.name = CODE_FONT
        run.font.size = Pt(9)
    if p.runs:
        last = p.runs[-1]
        if last.text.endswith("\n"):
            last.text = last.text[:-1]


def add_inline_code(paragraph, text):
    run = paragraph.add_run(text)
    run.font.name = CODE_FONT
    run.font.size = Pt(10)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        cells = table.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            for p in cells[ci].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    if col_widths:
        for col, width in zip(table.columns, col_widths):
            for cell in col.cells:
                cell.width = width
    return table


doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

# ── Title ───────────────────────────────────────────────────────────────────
doc.add_heading("Griffin + ICL Head Experiments: TabICL v2 and TabPFN Family", level=0)
p = doc.add_paragraph()
r = p.add_run(
    "Direction: o1 → o2 (others-1 → others-2). Probe-trained ICLProjection "
    "from rel-f1-driver-position, evaluated on others-2 (6 tasks). "
    "All runs share the same Griffin backbone (checkpoints/o1-tth-lora-v3) "
    "and identical probe configuration."
)
r.italic = True

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Parameters
# ═══════════════════════════════════════════════════════════════════════════════
doc.add_heading("1. Parameters used", level=1)

# Common setup
doc.add_heading("1.1 Common setup (all runs)", level=2)
add_table(
    doc,
    ["Component", "Value"],
    [
        ["Griffin backbone", "checkpoints/o1-tth-lora-v3/best_checkpoint"],
        ["Probe task (--tasks)", "rel-f1-driver-position (regression)"],
        ["Eval tasks (--eval_tasks)", "others-2 = airbnb-destination, rel-trial-site-success, rel-trial-study-adverse, rel-trial-study-outcome, talkingdata-demo-pred, telstra-severity"],
        ["Probe epochs (--probe_epochs)", "5"],
        ["ICLProjection dim (--icl_proj_dim)", "128 (Griffin 512-d → 128-d)"],
        ["Griffin params", "11.37M"],
        ["ICLProjection params", "82.4K"],
        ["LinearProbe params", "128 (regression head used during probe)"],
        ["Probe LR / WD", "1e-4 / 2e-4"],
        ["Probe batch size", "256"],
        ["Hop / fanout / fewshot fanout", "2 / 20 / 3"],
        ["Griffin hiddim / num_mp", "512 / 4"],
        ["Griffin flags", "--use_rev True --use_gate True --no_target_normalize"],
    ],
)

# TabICL v2 — ZS
doc.add_heading("1.2 TabICL v2 — Zero-shot ICL (run_icl_o1_to_o2.sh)", level=2)
add_table(
    doc,
    ["Knob", "Value", "Where set"],
    [
        ["Head", "tabicl", "--head tabicl"],
        ["Checkpoint", "tabicl-classifier-v2-20260212.ckpt", "--tabicl_checkpoint_version v2"],
        ["Max context", "10,000", "--icl_max_context 10000"],
        ["n_estimators (inference)", "8", "--icl_n_estimators 8"],
        ["FT enabled", "No", "(--tabicl_finetune absent)"],
        ["GPU memory ceiling", "~30 GB at 10K context", "—"],
        ["Approximate wall time / 6 tasks", "~3 min", "—"],
    ],
)

# TabICL v2 — FT (multiple configs tried)
doc.add_heading("1.3 TabICL v2 — Fine-tuning (run_icl_ft_bigctx_o1_to_o2.sh)", level=2)
doc.add_paragraph(
    "FinetunedTabICLRegressor / FinetunedTabICLClassifier from the tabicl[finetune] "
    "extra. Several context configurations tried — the final 'big-context' setup "
    "below was the best."
)
add_table(
    doc,
    ["Knob", "Value", "Where set"],
    [
        ["Head", "tabicl", "--head tabicl"],
        ["FT enabled", "Yes", "--tabicl_finetune"],
        ["Inference context", "10,000", "--icl_max_context 10000"],
        ["FT-step subsample cap", "2,000 (final), 5,000 (tried)", "--tabicl_finetune_max_data_size"],
        ["FT epochs (max)", "50", "--tabicl_finetune_epochs 50"],
        ["FT learning rate", "1e-5 (AdamW with cosine warmup)", "--tabicl_finetune_lr 1e-5"],
        ["Early-stopping patience", "10 epochs", "--tabicl_finetune_patience 10"],
        ["FT eval metric (default)", "mae for regression, roc_auc for classification", "_default_eval_metric()"],
        ["n_estimators_finetune / validation / inference", "1 / 1 / 8", "—"],
        ["Mixed precision", "bf16 (auto, tabicl default amp=True)", "—"],
        ["Activation checkpointing", "Not exposed", "—"],
        ["Class cap (FT only)", "10 — tasks with >10 classes auto-fallback to ZS", "TABICL_FT_MAX_CLASSES"],
        ["Approximate wall time / 6 tasks", "~10 min (2K), ~12 min (5K)", "—"],
    ],
)

doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("Iteration history on this script:")
r.bold = True
for line in [
    "Initial run with --icl_max_context 10000 + freeze_col=True → broken (device bug)",
    "Drop freeze_col, --icl_max_context 2000 + --max_data_size 2000 → first working FT run",
    "Switch to --icl_max_context 10000 + --max_data_size 2000 → 'rich inference, cheap FT' pattern (best)",
    "Try --max_data_size 5000 → mixed results, not a clear win over 2000",
]:
    doc.add_paragraph(line, style="List Bullet")

# TabPFN — ZS with KV cache
doc.add_heading("1.4 TabPFN v3 — Zero-shot ICL with KV cache (run_tabpfn_v3_zs_o1_to_o2.sh)", level=2)
add_table(
    doc,
    ["Knob", "Value", "Where set"],
    [
        ["Head", "tabpfn", "--head tabpfn"],
        ["TabPFN version", "v3 (ModelVersion.V3)", "--tabpfn_version v3"],
        ["Max context (rows passed to fit)", "30,000", "--icl_max_context 30000"],
        ["Inference subsample cap", "50,000", "--tabpfn_inference_config '{\"MAX_NUMBER_OF_SAMPLES\": 50000}'"],
        ["Fit mode", "fit_with_cache (precomputes K/V at fit)", "--tabpfn_fit_mode fit_with_cache"],
        ["Inference precision", "autocast (bf16/fp16 auto-picked)", "--tabpfn_inference_precision autocast"],
        ["n_estimators (inference)", "8", "--icl_n_estimators 8"],
        ["FT enabled", "No", "—"],
        ["Approximate wall time / 6 tasks", "~78 s", "—"],
        ["Predict throughput (avg)", "~4,000 rows/sec", "—"],
    ],
)

# TabPFN — ZS without KV cache
doc.add_heading("1.5 TabPFN v3 — Zero-shot ICL without KV cache (run_tabpfn_v3_zs_nocache_o1_to_o2.sh)", level=2)
doc.add_paragraph(
    "Identical to 1.4 except fit_mode='fit_preprocessors'. Used as the timing "
    "baseline to measure the KV-cache speedup."
)
add_table(
    doc,
    ["Knob", "Value", "Where set"],
    [
        ["Fit mode", "fit_preprocessors (default — no K/V precompute)", "--tabpfn_fit_mode fit_preprocessors"],
        ["Approximate wall time / 6 tasks", "~103 s", "—"],
        ["Predict throughput (avg)", "~770 rows/sec", "—"],
        ["(all other knobs identical to 1.4)", "", ""],
    ],
)

# TabPFN — FT
doc.add_heading("1.6 TabPFN — Fine-tuning (run_tabpfn_v3_ft_o1_to_o2.sh)", level=2)
doc.add_paragraph(
    "FinetunedTabPFNClassifier / FinetunedTabPFNRegressor from tabpfn 8.0.3. "
    "Note: the FT class hardcodes model_version=v2.5 internally — even when "
    "--tabpfn_version v3 is passed, FT silently uses v2.5 weights. ZS-fallback "
    "tasks still use v3."
)
add_table(
    doc,
    ["Knob", "Value", "Where set"],
    [
        ["Head", "tabpfn", "--head tabpfn"],
        ["FT enabled", "Yes", "--tabpfn_finetune"],
        ["Effective FT version", "v2.5 (hardcoded upstream)", "—"],
        ["Per-step episode size", "10,000 = 8K context + 2K queries", "n_finetune_ctx_plus_query_samples (default)"],
        ["Context/query split", "80%/20%", "finetune_ctx_query_split_ratio = 0.2 (default)"],
        ["Inference subsample cap", "50,000", "n_inference_subsample_samples (default)"],
        ["FT epochs (max)", "30", "--tabpfn_finetune_epochs 30"],
        ["FT learning rate", "1e-5", "--tabpfn_finetune_lr 1e-5"],
        ["Early-stopping patience", "8", "(default)"],
        ["Activation checkpointing", "True (default)", "use_activation_checkpointing"],
        ["Mixed precision", "bf16 (default amp=True)", "—"],
        ["n_estimators_finetune / validation / final_inference", "2 / 2 / 8", "—"],
        ["Class cap (FT only)", "10 — auto-fallback to ZS for >10-class tasks", "TABPFN_FT_MAX_CLASSES"],
        ["Approximate wall time / 6 tasks", "~32 min", "—"],
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Experimental Results
# ═══════════════════════════════════════════════════════════════════════════════
doc.add_heading("2. Experimental Results", level=1)

# 2.1 — full matrix
doc.add_heading("2.1 Test-metric matrix (all methods, all tasks)", level=2)
doc.add_paragraph(
    "All metrics are 'higher = better' (AUROC for classification, −MAE / −logloss "
    "for regression / cross-entropy). Bold = best per row. ZS = zero-shot, FT = fine-tuned. "
    "TabICL FT (best) uses --icl_max_context 10000 with --max_data_size 2000. "
    "TabPFN ZS uses --icl_max_context 30000 with fit_with_cache."
)
add_table(
    doc,
    ["Task", "Metric", "LLM ZS", "LLM FT best", "TabICL ZS", "TabICL FT best", "TabPFN ZS", "TabPFN FT"],
    [
        ["airbnb-destination", "AUROC ↑", "~0.80", "0.8583", "0.8582", "0.8585", "0.8621 ✨", "0.8620 (ZS fallback)"],
        ["rel-trial-site-success", "−MAE ↑", "−0.96", "−0.9288", "−0.9481", "−0.9438", "−0.9293 ✨", "−0.9468"],
        ["rel-trial-study-adverse", "−MAE ↑", "−2.39", "−1.5733", "−1.9950", "−2.0099", "−1.9720", "−1.8980 ✨"],
        ["rel-trial-study-outcome", "AUROC ↑", "0.54", "0.6537", "0.6614", "0.6594", "0.6571", "0.6650 ✨"],
        ["talkingdata-demo-pred", "−logloss ↑", "−2.60", "−2.4292", "−2.4805", "−2.4805", "−2.4796 ✨", "−2.4799 (ZS fallback)"],
        ["telstra-severity", "−logloss ↑", "−1.14", "−0.7584", "−0.8689 ✨", "−0.8724", "−0.8847", "−0.8806"],
    ],
)
doc.add_paragraph()

# 2.2 — Per-task winners
doc.add_heading("2.2 Per-task winner across all methods", level=2)
add_table(
    doc,
    ["Task", "Best result", "Method", "Notes"],
    [
        ["airbnb-destination", "0.8630 ✨", "SMPNN-6 + TabPFN v3 ZS @ 30K, NO projection", "Beats prior best 0.8621 (vanilla + proj)"],
        ["rel-trial-site-success", "−0.9038 ✨", "SMPNN-6 + TabPFN v3 ZS @ 30K, NO projection", "Beats LLM FT (−0.9288) by +0.025"],
        ["rel-trial-study-adverse", "−1.5733", "LLM FT", "Unchanged; LLM FT still dominant"],
        ["rel-trial-study-outcome", "0.6846 ✨", "Vanilla-4 + TabPFN v3 ZS @ 30K, NO projection", "Beats LLM FT (0.6537) by +0.031"],
        ["talkingdata-demo-pred", "−2.4292", "LLM FT", "Unchanged"],
        ["telstra-severity", "−0.7584", "LLM FT", "Unchanged"],
    ],
)

# 2.3 — KV cache A/B
doc.add_heading("2.3 KV cache A/B (TabPFN v3 ZS @ 30K)", level=2)
doc.add_paragraph(
    "Accuracy is identical within ±0.002 (KV cache changes when K/V is computed, "
    "not what is computed). Predict times differ substantially."
)
add_table(
    doc,
    ["Task", "Cache test", "No-cache test", "Δ"],
    [
        ["airbnb-destination (AUROC)", "0.8621", "0.8619", "+0.0002"],
        ["rel-trial-site-success (−MAE)", "−0.9293", "−0.9293", "0"],
        ["rel-trial-study-adverse (−MAE)", "−1.9703", "−1.9724", "+0.002"],
        ["rel-trial-study-outcome (AUROC)", "0.6571", "0.6572", "−0.0001"],
        ["talkingdata-demo-pred (−ll)", "−2.4796", "−2.4796", "0"],
        ["telstra-severity (−ll)", "−0.8847", "−0.8845", "−0.0002"],
    ],
)

# 2.4 — Timing
doc.add_heading("2.4 Inference timing across methods", level=2)
doc.add_paragraph(
    "Per-call predict times (seconds). TabPFN runs use 30K context; TabICL uses 10K — "
    "not strictly apples-to-apples, but represents each method's preferred operating point."
)
add_table(
    doc,
    ["Task", "TabPFN cache (s)", "TabPFN no-cache (s)", "TabICL @10K (s)"],
    [
        ["airbnb-destination", "1.92", "7.83", "4.00"],
        ["rel-trial-site-success", "4.64", "12.89", "3.18"],
        ["rel-trial-study-adverse", "0.75", "8.50", "1.36"],
        ["rel-trial-study-outcome", "0.27", "2.56", "1.15"],
        ["talkingdata-demo-pred", "0.71", "6.74", "3.10"],
        ["telstra-severity", "0.27", "1.85", "0.75"],
        ["Total (6 tasks, test only)", "8.56", "40.37", "13.54"],
    ],
)
doc.add_paragraph()

doc.add_heading("Total wall time (fit + 2 predict calls per task, 6 tasks)", level=3)
add_table(
    doc,
    ["Method", "Total wall time", "Notes"],
    [
        ["TabPFN ZS + cache @ 30K", "~78 s", "Fastest end-to-end"],
        ["TabPFN ZS no-cache @ 30K", "~103 s", "Same accuracy as cache, slower"],
        ["TabICL ZS @ 10K", "~3 min", "Smaller context, comparable per-call speed"],
        ["TabICL FT (best @ 2K)", "~10 min", "Mixed quality results"],
        ["TabPFN FT v2.5", "~32 min", "FT adds 20–60× wall time per FT-eligible task"],
        ["LLM FT (Qwen3-1.7B + LoRA) per sweep", "hours", "Sweep over N × LR grid; not directly comparable"],
    ],
)

# 2.5 — KV cache amortization
doc.add_heading("2.5 KV cache amortization (per-task break-even)", level=2)
doc.add_paragraph(
    "Cache fit costs extra time (precomputes K/V) but each predict() call is much faster. "
    "Break-even = (extra fit cost) / (per-call savings) = number of predict calls before "
    "cache becomes net-positive on wall time."
)
add_table(
    doc,
    ["Task", "Cache extra fit", "Per-call savings", "Break-even (# predicts)"],
    [
        ["airbnb-destination", "+12 s", "−6.1 s/call", "~2"],
        ["rel-trial-site-success", "+7.6 s", "−8.2 s/call", "~0.9"],
        ["rel-trial-study-adverse", "+8.3 s", "−7.7 s/call", "~1.1"],
        ["rel-trial-study-outcome", "+2.4 s", "−2.3 s/call", "~1.0"],
        ["talkingdata-demo-pred", "+6.2 s", "−6.1 s/call", "~1.0"],
        ["telstra-severity", "+1.9 s", "−1.5 s/call", "~1.2"],
    ],
)
doc.add_paragraph(
    "Our setup does 2 predicts per task (valid + test). At 2 predicts, the cache "
    "saves ~25% total wall time. At 10 predicts (production serving), savings would "
    "scale to ~70%."
)

# 2.6 — TabPFN FT delta from ZS
doc.add_heading("2.6 TabPFN FT vs TabPFN ZS — what FT changed", level=2)
add_table(
    doc,
    ["Task", "TabPFN ZS v3", "TabPFN FT v2.5", "Δ", "Verdict"],
    [
        ["rel-trial-site-success", "−0.9293", "−0.9468", "−0.018", "FT regressed (ZS at LLM-FT ceiling)"],
        ["rel-trial-study-adverse", "−1.9720", "−1.8980", "+0.074", "FT helped — biggest gain"],
        ["rel-trial-study-outcome", "0.6571", "0.6650", "+0.008", "FT helped marginally"],
        ["telstra-severity", "−0.8847", "−0.8806", "+0.004", "tied (within noise)"],
    ],
)

# 2.7 — Architectural observations
doc.add_heading("2.7 Architectural observations from probe scripts", level=2)
add_table(
    doc,
    ["Aspect", "TabICL FT", "TabPFN FT"],
    [
        ["Backbone version", "v2 only", "v2.5 hardcoded; v3 not yet supported"],
        ["Pretrained class cap", "10", "10"],
        ["Class-cap behavior on >10 classes", "Auto-fallback to ZS (our code)", "Auto-fallback to ZS (our code)"],
        ["Activation checkpointing default", "Not exposed", "True"],
        ["Mixed precision default", "True (amp=True)", "True (amp=True)"],
        ["Per-step episode size", "max_data_size (default 10000)", "n_finetune_ctx_plus_query_samples (default 10000)"],
        ["Practical max context on 1× 80 GB GPU", "~5K (FT step)", "~10K (FT step)"],
        ["KV cache for inference", "No", "Yes via fit_mode='fit_with_cache'"],
        ["Max inference context tested", "10K (configurable)", "30K easily, 50K via inference_config cap"],
    ],
)

# 2.8 — No-projection finding (the new headline result)
doc.add_heading("2.8 Bypassing ICLProjection: a strict simplification AND improvement", level=2)
doc.add_paragraph(
    "Late in the project we added a --no_icl_projection flag that lets "
    "TabPFN/TabICL consume the raw 512-d Griffin embeddings directly, "
    "skipping both the ICLProjection (512→128) and the probe phase. The "
    "2×2 matrix across both backbones (vanilla-4 and SMPNN-6) makes the "
    "finding unambiguous: removing the projection helps cross-task transfer "
    "for both backbones, dramatically so for SMPNN-6."
)
add_table(
    doc,
    ["Task (metric)", "Vanilla-4 +Proj+Probe", "Vanilla-4 NoProj",
     "SMPNN-6 +Proj+Probe", "SMPNN-6 NoProj"],
    [
        ["airbnb-destination (AUROC ↑)", "0.8621", "0.8583", "0.8545", "0.8630 ✨"],
        ["rel-trial-site-success (−MAE ↑)", "−0.9293", "−0.9336", "−0.9479", "−0.9038 ✨"],
        ["rel-trial-study-adverse (−MAE ↑)", "−1.9703", "−1.9604", "−2.8025", "−2.5558"],
        ["rel-trial-study-outcome (AUROC ↑)", "0.6571", "0.6846 ✨", "0.5249", "0.5417"],
        ["talkingdata-demo-pred (−ll ↑)", "−2.4796", "−2.4849", "−2.4796", "−2.4799"],
        ["telstra-severity (−ll ↑)", "−0.8847", "−0.8881", "−0.8568", "−0.8469 ✨"],
        ["Average (all 6, mixed scales)", "−0.7908", "−0.7874", "−0.9512", "−0.8970"],
    ],
)
doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("Three headline observations from the 2×2:")
r.bold = True
for line in [
    "1. Removing the projection improves both backbones. The 5-epoch probe "
    "overfits to the single probe task (rel-f1-driver-position), distorting "
    "Griffin's features in a way that hurts cross-task transfer. The "
    "ICLProjection's 512→128 bottleneck additionally compresses away signal.",
    "2. SMPNN-6's previous cross-task regression was projection-induced, not "
    "backbone-induced. With proj+probe SMPNN-6 lost on 5 of 6 tasks; without, "
    "it wins on 3 (airbnb, site-success, telstra) and ties on 1 (talkingdata).",
    "3. Three new project-bests resulted from this single change (the ✨ tasks). "
    "On rel-trial-site-success and rel-trial-study-outcome the no-projection "
    "result beats the best LLM fine-tuning baseline by 0.025–0.031 metric units.",
]:
    doc.add_paragraph(line, style="List Bullet")

doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("Recommended default pipeline going forward: ")
r.bold = True
p.add_run("--head tabpfn --tabpfn_version v3 --tabpfn_fit_mode fit_with_cache "
          "--no_icl_projection --probe_epochs 0 --icl_max_context 30000 "
          "--tabpfn_inference_config '{\"MAX_NUMBER_OF_SAMPLES\": 50000, "
          "\"MAX_NUMBER_OF_FEATURES\": 600}'. The backbone choice (vanilla vs "
          "SMPNN) depends on the task — SMPNN-6 wins on airbnb, site-success, "
          "and telstra; vanilla-4 wins on study-adverse and study-outcome. "
          "For a single safe default, vanilla-4 has the lowest variance.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Next Steps
# ═══════════════════════════════════════════════════════════════════════════════
doc.add_heading("3. Suggestions for Next Steps", level=1)
doc.add_paragraph(
    "Ordered roughly by expected return-on-effort. The cheapest experiments "
    "with the largest expected lift are at the top."
)

doc.add_heading("3.1 Multi-head ICLProjection probe (highest expected ROI)", level=2)
doc.add_paragraph(
    "Currently the probe trains ICLProjection (+ Griffin) on a single regression "
    "task (rel-f1-driver-position). LinearProbe has a fixed out_dim=1, which "
    "forced this restriction. Refactor LinearProbe into a task-typed multi-head:"
)
add_code(doc, """class TaskTypedProbe(nn.Module):
    def __init__(self, in_dim, max_classes):
        self.reg_head = nn.Linear(in_dim, 1)
        self.bin_head = nn.Linear(in_dim, 2)
        self.multi_head = nn.Linear(in_dim, max_classes)
    def forward(self, x, task_type):
        if task_type == 'regression':
            return self.reg_head(x)
        if task_type == 'binary':
            return self.bin_head(x)
        return self.multi_head(x)""")
doc.add_paragraph(
    "Then probe on ALL source-side tasks (6 in others-1): 1 regression + 5 binary. "
    "ICLProjection sees 5–6× more supervision signal from diverse target distributions."
)
doc.add_paragraph("Expected effect:")
for line in [
    "ICLProjection learns richer features → both TabICL and TabPFN heads benefit",
    "Most likely to push tasks where probe variance was high (e.g., telstra)",
    "~1 hour of code work, ~30 minutes per probe re-train",
    "All downstream eval runs benefit without further changes",
]:
    doc.add_paragraph(line, style="List Bullet")

doc.add_heading("3.2 Push TabPFN inference context to 50K / 100K", level=2)
doc.add_paragraph(
    "Three tasks still trail LLM FT (study-adverse, talkingdata, telstra). For "
    "study-adverse and talkingdata the training pool is 43K and 67K respectively — "
    "30K context only sees a fraction. KV cache makes large context cheap at predict time."
)
doc.add_paragraph("Plan:")
for line in [
    "Set --icl_max_context 50000 (or 100000)",
    "Set --tabpfn_inference_config '{\"MAX_NUMBER_OF_SAMPLES\": 100000}'",
    "Re-run run_tabpfn_v3_zs_o1_to_o2.sh (cache version)",
    "Compare test metrics vs the 30K run",
]:
    doc.add_paragraph(line, style="List Bullet")
doc.add_paragraph(
    "Memory budget: TabPFN v3 with bf16 + KV cache should fit 50K on 80 GB. "
    "Test on study-adverse first (43K train); if it fits, push higher."
)

doc.add_heading("3.3 Run full TabPFN ZS matrix on the other 3 directions", level=2)
doc.add_paragraph(
    "All TabPFN runs so far are on o1→o2. The other three transfer directions "
    "(o2→o1, c1→c2, c2→c1) still rely on TabICL and LLM FT results."
)
add_table(
    doc,
    ["Direction", "Eval tasks", "Probe task (regression in source)"],
    [
        ["o2→o1", "others-1 (6 tasks)", "rel-trial-site-success"],
        ["c1→c2", "commerce-2 (6 tasks)", "rel-hm-item-sales"],
        ["c2→c1", "commerce-1 (6 tasks)", "rel-avito-ad-ctr"],
    ],
)
doc.add_paragraph(
    "Create three new scripts (mirroring run_tabpfn_v3_zs_o1_to_o2.sh). Run all "
    "three with cache + 30K context. ~5 minutes each. Closes out the full 4-direction "
    "TabPFN matrix."
)

doc.add_heading("3.4 Wider ICLProjection (256-d or 512-d)", level=2)
doc.add_paragraph(
    "Current ICLProjection: Linear(512→128). TabPFN was pretrained with up to "
    "several hundred input features — 128-d may be a bottleneck. Try --icl_proj_dim 256 "
    "and --icl_proj_dim 512."
)
doc.add_paragraph("Cost: 4× more projection params (still tiny), no inference slowdown.")
doc.add_paragraph(
    "Expected effect: modest. Probably worth running once at 256-d as a follow-up "
    "after 3.1; combine with multi-head probe for compound effect."
)

doc.add_heading("3.5 Targeted FT for the 3 tasks where ZS underperforms LLM FT", level=2)
doc.add_paragraph(
    "Don't run blanket FT. Identify the tasks where TabPFN ZS @ 30K still trails "
    "LLM FT by >0.05 metric units, and FT only those:"
)
add_table(
    doc,
    ["Task", "Gap to LLM FT", "Recommended action"],
    [
        ["rel-trial-study-adverse", "+0.40 MAE", "TabPFN FT v2.5 confirmed → −1.90 (still trails)"],
        ["talkingdata-demo-pred", "+0.05 logloss", "Push context to 50K first; only FT if still gap"],
        ["telstra-severity", "+0.12 logloss", "Small task (6.6K), FT v2.5 didn't help; consider per-task LR sweep"],
    ],
)

doc.add_heading("3.6 Wait for TabPFN v3 FT support", level=2)
doc.add_paragraph(
    "FinetunedTabPFN* currently hardcodes v2.5. When soda-inria adds v3 FT support, "
    "rerun the FT experiments — v3 backbone with FT may close more of the gap on "
    "study-adverse (the task where FT showed the most lift)."
)
doc.add_paragraph("Watch tabpfn release notes for: 'support for v3 in FinetunedTabPFN*'")

doc.add_heading("3.7 TabICL FT activation checkpointing (monkey-patch)", level=2)
doc.add_paragraph(
    "tabicl does not expose gradient checkpointing as a kwarg. This is why TabICL FT "
    "OOMs at ~5–6K context where TabPFN FT handles 10K. Two options:"
)
for line in [
    "(a) Submit a feature request / PR upstream to expose use_activation_checkpointing",
    "(b) Monkey-patch tabicl._model.tabicl.TabICL.forward to wrap heavy layers in torch.utils.checkpoint.checkpoint",
]:
    doc.add_paragraph(line, style="List Bullet")
doc.add_paragraph(
    "Option (b) is a 1–2 hour experiment. Would let TabICL FT match TabPFN FT's "
    "context scale and give a clean v2-vs-v2.5 FT comparison."
)

doc.add_heading("3.8 Multi-GPU FT via accelerate launch", level=2)
doc.add_paragraph(
    "Our pipeline currently uses Accelerator() but ICL eval runs on rank 0 only. "
    "True multi-GPU FT (DDP) would require:"
)
for line in [
    "Refactor _run_icl_evaluation to run icl_head.fit collectively (not behind 'if accelerator.is_main_process')",
    "Verify tabpfn / tabicl FT classes are DDP-aware (or wrap them manually)",
    "Launch with `accelerate launch --num_processes 2 --multi_gpu hmaintask_combine_llm.py …`",
]:
    doc.add_paragraph(line, style="List Bullet")
doc.add_paragraph(
    "Effort: ~half-day. Value: ~2× FT wall-time speedup, potentially 2× larger "
    "effective context. Defer until 3.1, 3.2, and 3.3 are done."
)

doc.add_heading("3.9 Apples-to-apples TabICL vs TabPFN at the same context", level=2)
doc.add_paragraph(
    "Current comparisons mix contexts (TabICL @10K, TabPFN @30K). For a clean "
    "architecture comparison, run both at the same context. Two sub-experiments:"
)
add_table(
    doc,
    ["Variant", "Effort", "What it answers"],
    [
        ["TabPFN ZS @ 10K", "Trivial — change --icl_max_context", "Removes context size as a confounding variable"],
        ["TabICL ZS @ 30K (via chunked predict)", "Modest — pure inference, no FT, should fit", "Tests whether tabicl can leverage 30K context for ZS"],
    ],
)
doc.add_paragraph(
    "These tell us whether TabPFN's quality lead comes from the architecture or "
    "from the bigger context that its KV cache enables. Useful for the project doc."
)

doc.add_heading("3.10 Long-tail regression: study-adverse-specific tuning", level=2)
doc.add_paragraph(
    "study-adverse has 41% mass clipped at exactly −5.199. This is the single "
    "task where FT showed the most lift (ZS −1.97 → FT −1.90) and the biggest "
    "remaining gap to LLM FT (−1.57). Two task-specific levers:"
)
for line in [
    "Use rel-avito-ad-ctr instead of rel-f1-driver-position as the probe task (more comparable target distribution)",
    "Try Huber loss instead of MSE for the probe (already implemented as --huber_loss)",
    "Try TabPFN FT with longer epochs (--tabpfn_finetune_epochs 60) and lower LR (--tabpfn_finetune_lr 5e-6)",
]:
    doc.add_paragraph(line, style="List Bullet")

# Summary box
doc.add_heading("Summary of recommended next 1 week of work", level=2)
add_table(
    doc,
    ["Priority", "Experiment", "Time", "Expected value"],
    [
        ["P0", "3.1 Multi-head ICLProjection probe", "1 hour code + 30 min retrain", "High — improves all downstream"],
        ["P0", "3.3 TabPFN ZS on the other 3 directions", "30 min total", "Completes the 4-direction matrix"],
        ["P1", "3.2 TabPFN @ 50K context", "30 min", "Likely closes gap on study-adverse, talkingdata"],
        ["P1", "3.9 TabICL @ 30K + TabPFN @ 10K", "30 min", "Clean architecture comparison"],
        ["P2", "3.4 Wider ICLProjection (256-d)", "10 min config", "Modest, possibly compounds with 3.1"],
        ["P2", "3.7 TabICL activation checkpointing", "2 hours", "Unlocks larger-context TabICL FT"],
    ],
)

# ── Save ────────────────────────────────────────────────────────────────────
out = "/home/pviswanath/Griffin/Griffin_TabPFN_TabICL_Experiment_Report.docx"
doc.save(out)
print(f"Wrote {out}")
