"""Turn the LoG-rebuttal eval logs into the paper's missing numbers.

Parses logs/log-rebuttal-eval/*.log (from run_log_rebuttal_ablations.sh),
computes the per-seed direction averages exactly as the paper does
(classification AUROC only), and reports mean +/- sigma with Welch and
paired tests against the paper's published baselines:

  ABLATION 1  V6 on o2->o1 (TabPFN + TabICL): direction average over the
              5 classification tasks of others-1; compared to V4 and the
              two SMPNN arms from Appendix B.4.
  ABLATION 2  SMPNN-6 alpha=1e-6 on o1->o2 (TabPFN): airbnb-destination
              only (the paper's anchor); compared per-seed (paired t)
              against V4, V6, SMPNN-1e-2 from Appendix B.1.

Usage:
    python analyze_rebuttal_ablations.py [--logs 'logs/log-rebuttal-eval/*.log']
"""
import argparse
import glob
import math
import re
from collections import defaultdict
from pathlib import Path

# Direction average = classification tasks of the TARGET family (paper A).
DIRECTION_TASKS = {
    "o2-to-o1": ["rel-f1-driver-dnf", "rel-f1-driver-top3",
                 "stackexchange-churn", "stackexchange-upvote",
                 "virus-wnv-pred"],
    "o1-to-o2": ["airbnb-destination"],   # the paper's single-task anchor
}

# Published baselines (paper Appendix B.1 / B.4), n=5, seeds 42-46.
BASELINES = {
    ("o2-to-o1", "tabpfn"): {
        "V4": (0.7540, 0.0054), "SMPNN-6 a1e-6": (0.7520, 0.0091),
        "SMPNN-6 a1e-2": (0.7442, 0.0172),
    },
    ("o2-to-o1", "tabicl"): {
        "V4": (0.7443, 0.0081), "SMPNN-6 a1e-6": (0.7420, 0.0119),
        "SMPNN-6 a1e-2": (0.7352, 0.0133),
    },
    ("o1-to-o2", "tabpfn"): {
        "V4": (0.8151, 0.0656), "V6": (0.8574, 0.0102),
        "SMPNN-6 a1e-2": (0.8602, 0.0015),
    },
}
# Per-seed values where the paper prints them (B.1) -> paired t possible.
BASELINE_SEEDS = {
    ("o1-to-o2", "tabpfn", "V4"): [0.709, 0.867, 0.870, 0.827, 0.803],
    ("o1-to-o2", "tabpfn", "V6"): [0.862, 0.841, 0.859, 0.868, 0.856],
    ("o1-to-o2", "tabpfn", "SMPNN-6 a1e-2"): [0.862, 0.860, 0.860, 0.861, 0.858],
}

TAG_RE = re.compile(r"^s(\d+)-((?:[oc]\d)-to-(?:[oc]\d))-(.+)-(tabpfn|tabicl)-noproj$")
METRIC_RE = re.compile(r"test_metric/([\w\-]+?)(?:/[\w\-]+)?:\s*([-0-9.eE]+)")


def mean(xs):
    return sum(xs) / len(xs)


def sstd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t, two-sided p, and Welch-Satterthwaite df — same test the
    paper reports (Appendix B.5)."""
    from scipy import stats
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    if v1 + v2 == 0:
        return float("nan"), float("nan"), float("nan")
    t = (m1 - m2) / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return t, 2 * stats.t.sf(abs(t), df), df


def _t_sf(t, df):
    from scipy import stats
    return stats.t.sf(t, df)


def paired_t(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    sd = sstd(d)
    if sd == 0:
        return float("nan"), float("nan")
    t = mean(d) / (sd / math.sqrt(n))
    return t, 2 * _t_sf(abs(t), n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/log-rebuttal-eval/*.log")
    args = ap.parse_args()

    # cell[(direction, head, backbone)][seed] = direction-average AUROC
    cells = defaultdict(dict)
    for path in sorted(glob.glob(args.logs)):
        m = TAG_RE.match(Path(path).stem)
        if not m:
            continue
        seed, direction, backbone, head = m.groups()
        wanted = DIRECTION_TASKS.get(direction)
        if wanted is None:
            continue
        vals = {}
        for line in open(path, errors="replace"):
            mm = METRIC_RE.search(line)
            if mm and mm.group(1) in wanted:
                vals[mm.group(1)] = float(mm.group(2))  # last occurrence wins
        if len(vals) < len(wanted):
            print(f"[warn] {path}: {len(vals)}/{len(wanted)} tasks found "
                  f"(missing: {sorted(set(wanted) - set(vals))}) — skipped")
            continue
        cells[(direction, head, backbone)][int(seed)] = mean(
            [vals[t] for t in wanted])

    if not cells:
        raise SystemExit(f"no complete eval logs matched {args.logs}")

    for (direction, head, backbone), per_seed in sorted(cells.items()):
        seeds = sorted(per_seed)
        xs = [per_seed[s] for s in seeds]
        m, s = mean(xs), sstd(xs)
        print(f"\n=== {direction} ({head})  {backbone}  n={len(xs)} ===")
        print("  per-seed: " + ", ".join(f"s{sd}={v:.4f}"
                                         for sd, v in zip(seeds, xs)))
        print(f"  mean ± σ: {m:.4f} ± {s:.4f}")
        for name, (bm, bs) in BASELINES.get((direction, head), {}).items():
            t, p, df = welch(m, s, len(xs), bm, bs, 5)
            print(f"  vs {name:14s} ({bm:.4f}±{bs:.4f}): Δ={m - bm:+.4f}  "
                  f"Welch t={t:+.2f} p={p:.3f} (df={df:.1f})")
            bl_seeds = BASELINE_SEEDS.get((direction, head, name))
            if bl_seeds and seeds == [42, 43, 44, 45, 46]:
                pt, pp = paired_t(xs, bl_seeds)
                print(f"     paired (matched seeds): t={pt:+.2f} p={pp:.3f}")
        if len(xs) < 5:
            print(f"  [note] paper cells use n=5; only {len(xs)} seeds found")

    print("\nInterpretation crib (Section 4.3):")
    print("  A1: V6 ~= V4 on o2->o1  -> depth-tracking account WRONG "
          "(directional story instead)")
    print("      V6 <  V4 on o2->o1  -> depth-tracking account CONFIRMED "
          "(4th direction)")
    print("  A2: fills Table 1 (o1->o2, alpha=1e-6) and completes Appendix D "
          "where SMPNN wins")


if __name__ == "__main__":
    main()
