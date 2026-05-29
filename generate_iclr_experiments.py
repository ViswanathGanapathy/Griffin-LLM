"""Generate the exhaustive ICLR-manuscript experiment plan as a CSV.

One row per experiment group (where a "group" = a set of seeds sharing
the same backbone, source, target, head, etc. — i.e. the unit you'd run
together). Seed count, wall-time, priority, status, and paper-table
mapping are columns.

Output: iclr_experiments_plan.csv

Usage:
    python3 generate_iclr_experiments.py [--out PATH]
"""
import argparse
import csv
from itertools import product


# ============================================================
# Knobs
# ============================================================

SEEDS_HEADLINE = 3        # P0 rows that go in the headline tables
SEEDS_AUXILIARY = 1       # Component-ablation rows (in-dist only) -- 1 seed is fine
SEEDS_MULTISEED_AUDIT = 3 # The few variants we promote to "robust" claims

TRAIN_HOURS_PER_BACKBONE = 3.5   # 20 epochs on A100 (in-dist regime)
ICL_HOURS_PER_EVAL = 0.1         # ~5-6 min per eval

# 11 in-distribution backbone variants from the ablation suite
VARIANTS = [
    # (tag, label, train_flags, native_avg_others1_1seed, params_M, family)
    ("V0",  "Vanilla-4",                  "--num_mp 4",                                                          0.5476, 11.4, "baseline"),
    ("V1",  "SMPNN-4 (parity)",           "--num_mp 4 --use_smpnn",                                              0.5426, 13.0, "smpnn"),
    ("V2",  "SMPNN-6 (D1, default)",      "--num_mp 6 --use_smpnn",                                              0.5572, 17.2, "smpnn"),
    ("V3",  "SMPNN-8",                    "--num_mp 8 --use_smpnn",                                              0.5405, 23.0, "smpnn"),
    ("V4",  "Vanilla-6 (C3)",             "--num_mp 6",                                                          0.5490, 16.4, "depth-ctrl"),
    ("V5",  "SMPNN-6 alpha=1e-2 (D3)",    "--num_mp 6 --use_smpnn --alpha_init 1e-2",                            0.5650, 17.2, "smpnn-alpha"),
    ("V6",  "SMPNN-6 alpha=1e-4 (D2)",    "--num_mp 6 --use_smpnn --alpha_init 1e-4",                            0.5393, 17.2, "smpnn-alpha"),
    ("V7",  "SMPNN-6 + attention (B1)",   "--num_mp 6 --use_smpnn --use_attention True --num_heads 1",           0.5241, 21.9, "smpnn-attn"),
    ("V8a", "SMPNN-6 no-alpha (A2)",      "--num_mp 6 --use_smpnn --use_alpha False",                            0.5466, 17.2, "smpnn-ablate"),
    ("V8b", "SMPNN-6 no-FF (A3)",         "--num_mp 6 --use_smpnn --use_ff False",                               0.5280, 14.0, "smpnn-ablate"),
    ("V8c", "SMPNN-6 no-GNN-LN (A4)",     "--num_mp 6 --use_smpnn --use_gnn_ln False",                           0.5422, 17.2, "smpnn-ablate"),
]

# Anchor variants we promote to multi-seed + cross-task
ANCHOR_TAGS = ["V0", "V2", "V5"]

# Four RelBench source families
SOURCE_FAMILIES = ["others-1", "others-2", "commerce-1", "commerce-2"]

# Cross-task transfer directions: (source, target, short_name)
DIRECTIONS = [
    ("others-1",   "others-2",   "o1->o2"),
    ("others-2",   "others-1",   "o2->o1"),
    ("commerce-1", "commerce-2", "c1->c2"),
    ("commerce-2", "commerce-1", "c2->c1"),
]

# Downstream heads to evaluate ICL embedding quality through
HEADS = ["tabpfn-zs", "tabicl-zs"]


# ============================================================
# Status of already-completed work
# ============================================================
# Backbone trainings that are already done (source, variant, seed_count_done)
DONE_BACKBONES = {
    ("others-1", "V0"): 1,
    ("others-1", "V1"): 1,
    ("others-1", "V2"): 1,
    ("others-1", "V3"): 1,
    ("others-1", "V4"): 1,
    ("others-1", "V5"): 1,
    ("others-1", "V6"): 1,
    ("others-1", "V7"): 1,
    ("others-1", "V8a"): 1,
    ("others-1", "V8b"): 1,
    ("others-1", "V8c"): 1,
}

# ICL evals that are already done (source, target, variant, head, seed_count_done)
DONE_EVALS = {
    # In-dist others-1 ICL sweep landed already
    **{("others-1", "others-1", tag, head): 1 for tag in [v[0] for v in VARIANTS] for head in HEADS},
}


# ============================================================
# Row generation
# ============================================================

def status(done, target):
    if done >= target:
        return "DONE"
    if done > 0:
        return f"PARTIAL ({done}/{target})"
    return "TODO"


def rows_backbones():
    """One row per (source family, variant, seed-group) needing backbone training."""
    rows = []
    for src, (tag, label, flags, native_avg, params_m, family) in product(SOURCE_FAMILIES, VARIANTS):
        # Multi-seed only the anchors; auxiliary variants stay single-seed on others-1
        if tag in ANCHOR_TAGS:
            target_seeds = SEEDS_HEADLINE
        elif src == "others-1":
            target_seeds = SEEDS_AUXILIARY
        else:
            # Non-anchor variants on non-others-1 families: skip (the spreadsheet
            # marks these as "skipped"; we still emit a row to make the gap explicit)
            target_seeds = 0

        done = DONE_BACKBONES.get((src, tag), 0)

        if target_seeds == 0:
            stat = "SKIP (anchors only on this family)"
            est_hours = 0.0
            priority = "—"
        else:
            stat = status(done, target_seeds)
            remaining = max(0, target_seeds - done)
            est_hours = remaining * TRAIN_HOURS_PER_BACKBONE
            # Priority: anchors on non-others-1 are P0; others-1 multi-seed is P1; others are P2
            if tag in ANCHOR_TAGS and src != "others-1":
                priority = "P0"
            elif tag in ANCHOR_TAGS and src == "others-1":
                priority = "P1" if done < target_seeds else "DONE"
            else:
                priority = "P3"

        rows.append({
            "study": "Backbone Training",
            "id": f"B-{src}-{tag}",
            "source_family": src,
            "target_family": "—",
            "variant": tag,
            "variant_label": label,
            "head": "native (training-only)",
            "seeds_target": target_seeds,
            "seeds_done": done,
            "status": stat,
            "wall_hours_remaining": round(est_hours, 1),
            "priority": priority,
            "paper_table": "T1 backbone ablation (in-dist) / T2 cross-task transfer",
            "dependencies": "—",
            "notes": f"{label} on {src}; params={params_m}M",
        })
    return rows


def rows_indist_icl():
    """In-dist ICL evals: backbone trained AND evaluated on the same source family."""
    rows = []
    for src, (tag, label, flags, _, _, _), head in product(
        SOURCE_FAMILIES, VARIANTS, HEADS,
    ):
        if tag in ANCHOR_TAGS:
            target_seeds = SEEDS_HEADLINE
        elif src == "others-1":
            target_seeds = SEEDS_AUXILIARY
        else:
            target_seeds = 0

        done = DONE_EVALS.get((src, src, tag, head), 0)

        if target_seeds == 0:
            stat = "SKIP"
            est_hours = 0.0
            priority = "—"
        else:
            stat = status(done, target_seeds)
            remaining = max(0, target_seeds - done)
            est_hours = remaining * ICL_HOURS_PER_EVAL
            if tag in ANCHOR_TAGS and src != "others-1":
                priority = "P0"
            elif src == "others-1" and tag in ANCHOR_TAGS:
                priority = "P1"
            else:
                priority = "P3"

        rows.append({
            "study": "In-dist ICL eval",
            "id": f"I-{src}-{tag}-{head}",
            "source_family": src,
            "target_family": src,
            "variant": tag,
            "variant_label": label,
            "head": head,
            "seeds_target": target_seeds,
            "seeds_done": done,
            "status": stat,
            "wall_hours_remaining": round(est_hours, 2),
            "priority": priority,
            "paper_table": "T1 in-dist ablation",
            "dependencies": f"requires backbone B-{src}-{tag}",
            "notes": f"{label} embedding -> {head} no-proj on {src}",
        })
    return rows


def rows_xtask_icl():
    """Cross-task ICL evals: backbone trained on source, eval on target."""
    rows = []
    for (src, tgt, dir_name), (tag, label, flags, _, _, _), head in product(
        DIRECTIONS, VARIANTS, HEADS,
    ):
        # Cross-task: only the 3 anchors. Component ablations stay in-dist.
        if tag not in ANCHOR_TAGS:
            continue
        target_seeds = SEEDS_HEADLINE
        done = DONE_EVALS.get((src, tgt, tag, head), 0)
        stat = status(done, target_seeds)
        remaining = max(0, target_seeds - done)
        est_hours = remaining * ICL_HOURS_PER_EVAL
        priority = "P0"

        rows.append({
            "study": "Cross-task ICL eval",
            "id": f"X-{dir_name}-{tag}-{head}",
            "source_family": src,
            "target_family": tgt,
            "variant": tag,
            "variant_label": label,
            "head": head,
            "seeds_target": target_seeds,
            "seeds_done": done,
            "status": stat,
            "wall_hours_remaining": round(est_hours, 2),
            "priority": priority,
            "paper_table": "T2 cross-task transfer matrix",
            "dependencies": f"requires backbone B-{src}-{tag}",
            "notes": f"{dir_name}: {label} -> {head} no-proj",
        })
    return rows


def rows_hop3():
    """Hop=3 backbone trainings + their ICL evals."""
    rows = []
    hop3_variants = [
        ("V0",  "Vanilla-4 @ hop=3",   "--num_mp 4 --hop 3 --fanout 6 --batchsize 32",   "exists (bs=64, fanout=8 -- regime confound)"),
        ("V2",  "SMPNN-6 @ hop=3",     "--num_mp 6 --use_smpnn --hop 3 --fanout 6 --batchsize 32", "1 seed, 0.486 (likely undertrained)"),
        ("V7",  "SMPNN-6+attn @ hop=3","--num_mp 6 --use_smpnn --use_attention True --hop 3 --fanout 6 --batchsize 32", "not yet trained"),
    ]
    for tag, label, flags, note in hop3_variants:
        rows.append({
            "study": "Hop=3 backbone (apples-to-apples)",
            "id": f"H-others-1-{tag}-hop3",
            "source_family": "others-1",
            "target_family": "—",
            "variant": f"{tag}-hop3",
            "variant_label": label,
            "head": "native",
            "seeds_target": SEEDS_HEADLINE,
            "seeds_done": 1 if tag in ("V0", "V2") else 0,
            "status": "PARTIAL or TODO",
            "wall_hours_remaining": SEEDS_HEADLINE * TRAIN_HOURS_PER_BACKBONE * 1.5,  # hop=3 slower
            "priority": "P2",
            "paper_table": "T3 hop=3 with/without attention",
            "dependencies": "—",
            "notes": note,
        })
        for head in HEADS:
            rows.append({
                "study": "Hop=3 ICL eval",
                "id": f"H-others-1-{tag}-hop3-{head}",
                "source_family": "others-1",
                "target_family": "others-1",
                "variant": f"{tag}-hop3",
                "variant_label": label,
                "head": head,
                "seeds_target": SEEDS_HEADLINE,
                "seeds_done": 0,
                "status": "TODO",
                "wall_hours_remaining": SEEDS_HEADLINE * ICL_HOURS_PER_EVAL,
                "priority": "P2",
                "paper_table": "T3 hop=3 with/without attention",
                "dependencies": f"requires backbone H-others-1-{tag}-hop3",
                "notes": f"{label} -> {head} no-proj",
            })
    return rows


def rows_diagnostics():
    """Diagnostic / specialised experiments that don't fit the main matrix."""
    return [
        {
            "study": "Alpha-trajectory logging",
            "id": "D-alpha-traj-V2-V5",
            "source_family": "others-1",
            "target_family": "—",
            "variant": "V2, V5",
            "variant_label": "SMPNN-6 default vs alpha=1e-2",
            "head": "native (logged during training)",
            "seeds_target": 1,
            "seeds_done": 1,
            "status": "DONE (already logging via --log_alpha_every 2)",
            "wall_hours_remaining": 0.0,
            "priority": "DONE",
            "paper_table": "F2 alpha self-pruning figure",
            "dependencies": "comes free with B-others-1-V2 / V5",
            "notes": "Final per-layer alpha_gnn[i] / alpha_ff[i] for the figure",
        },
        {
            "study": "Constant-feature diagnostic on c1->avito",
            "id": "D-c1-avito-collapse",
            "source_family": "commerce-1",
            "target_family": "commerce-2 (specifically avito-user-clicks, avito-user-visits)",
            "variant": "V0, V2, V5",
            "variant_label": "all 3 anchors",
            "head": "tabpfn-zs",
            "seeds_target": 1,
            "seeds_done": 0,
            "status": "TODO",
            "wall_hours_remaining": 0.5,
            "priority": "P1",
            "paper_table": "S1 failure-mode case study",
            "dependencies": "requires B-commerce-1-V0/V2/V5",
            "notes": "Does any SMPNN variant avoid the constant-feature TabPFN error?",
        },
        {
            "study": "Parameter count + peak GPU memory",
            "id": "D-compute-cost",
            "source_family": "—",
            "target_family": "—",
            "variant": "all 11",
            "variant_label": "params + peak memory per variant",
            "head": "—",
            "seeds_target": 1,
            "seeds_done": 1,
            "status": "DONE (collected via [Params] logs + nvidia-smi snapshots)",
            "wall_hours_remaining": 0.0,
            "priority": "DONE",
            "paper_table": "T1 also shows params, F3 memory scaling",
            "dependencies": "—",
            "notes": "Already printed by hmaintask_combine.py construction-time log line",
        },
        {
            "study": "LLM-FT baseline comparison",
            "id": "D-llm-ft-baseline",
            "source_family": "all 4",
            "target_family": "all 4 (in-dist + cross-task)",
            "variant": "Qwen3-1.7B LoRA r=16",
            "variant_label": "Griffin + LLM FT recipe",
            "head": "llm",
            "seeds_target": 1,
            "seeds_done": 1,
            "status": "DONE (from prior project work)",
            "wall_hours_remaining": 0.0,
            "priority": "DONE",
            "paper_table": "T1, T2 (LLM-FT column)",
            "dependencies": "results in prior Griffin_Head_Comparison_Report.docx",
            "notes": "Used as the third baseline alongside vanilla-4 and TabPFN ZS",
        },
        {
            "study": "Inference latency / KV cache A/B",
            "id": "D-kv-cache-ab",
            "source_family": "—",
            "target_family": "all RelBench (others-2 used in prior work)",
            "variant": "V0, V5",
            "variant_label": "TabPFN with vs without fit_with_cache",
            "head": "tabpfn-zs",
            "seeds_target": 1,
            "seeds_done": 1,
            "status": "DONE (others-2)",
            "wall_hours_remaining": 0.0,
            "priority": "DONE",
            "paper_table": "T4 / F4 KV-cache speedup",
            "dependencies": "results in prior Griffin_Head_Comparison_Report.docx",
            "notes": "25% wall-time saving at 2 predicts; 70% at 10 predicts; 0 accuracy delta",
        },
        {
            "study": "Multi-dataset validation (out of scope for v1)",
            "id": "D-multi-dataset",
            "source_family": "OGB-LSC or custom RDB",
            "target_family": "—",
            "variant": "V0, V5",
            "variant_label": "Headline anchors on a 2nd RDB benchmark",
            "head": "all",
            "seeds_target": 3,
            "seeds_done": 0,
            "status": "DEFER",
            "wall_hours_remaining": 60.0,
            "priority": "P3",
            "paper_table": "T5 (optional) cross-benchmark validation",
            "dependencies": "—",
            "notes": "Recommended for ICLR-main strength; skippable for workshop",
        },
    ]


# ============================================================
# Write CSV
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="iclr_experiments_plan.csv")
    args = ap.parse_args()

    all_rows = (
        rows_backbones()
        + rows_indist_icl()
        + rows_xtask_icl()
        + rows_hop3()
        + rows_diagnostics()
    )

    # Sort by priority, then study, then id
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "DONE": 4, "—": 5}
    all_rows.sort(key=lambda r: (
        priority_order.get(r["priority"], 9),
        r["study"],
        r["id"],
    ))

    fieldnames = [
        "id", "study", "source_family", "target_family",
        "variant", "variant_label", "head",
        "seeds_target", "seeds_done", "status",
        "wall_hours_remaining", "priority",
        "paper_table", "dependencies", "notes",
    ]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # Summary by priority
    print(f"wrote {args.out} ({len(all_rows)} rows)")
    print()
    print("Summary by priority (remaining wall-hours):")
    by_prio = {}
    for r in all_rows:
        p = r["priority"]
        by_prio.setdefault(p, {"count": 0, "hours": 0.0})
        by_prio[p]["count"] += 1
        by_prio[p]["hours"] += float(r["wall_hours_remaining"])
    for p in ["P0", "P1", "P2", "P3", "DONE", "—"]:
        if p in by_prio:
            d = by_prio[p]
            print(f"  {p:6s}  {d['count']:4d} rows   {d['hours']:6.1f} GPU-hours")
    print()
    print(f"Total GPU-hours remaining: {sum(d['hours'] for d in by_prio.values()):.1f}")
    print(f"Of which P0 (must-do for ICLR submission): "
          f"{by_prio.get('P0', {}).get('hours', 0):.1f}")


if __name__ == "__main__":
    main()
