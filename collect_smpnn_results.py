"""Collect per-task test metrics across all SMPNN ablation + depth-scan runs.

Parses the tee log files produced by run_smpnn_ablations.sh and
run_smpnn_depth_native.sh, and emits a CSV + a markdown table suitable for
pasting into the paper or report doc.

Usage:
    python collect_smpnn_results.py \
        --logs logs/smpnn-ablations.log logs/smpnn-ablations-resume.log \
               logs/smpnn-depth.log logs/smpnn-hop3.log \
        --out smpnn_results.csv
"""
import argparse
import re
import csv
from collections import defaultdict
from pathlib import Path


# Markers in the tee log:
#   ">>> Training <tag>"           starts a new run block
#   "test_metric/<task>/<m>: <v>"  per-task test metric line (final ones win)
#   "Average test metric: <v>"     summary line at end of each run
TRAIN_RE = re.compile(r">>>\s+(?:Training\s+)?(\S+)")
TEST_RE = re.compile(r"test_metric/([\w\-]+)/(\w+):\s*([\-0-9.eE]+)")
AVG_RE = re.compile(r"Average test metric:\s*([\-0-9.eE]+)")


def parse_logs(paths):
    runs = defaultdict(lambda: {"per_task": {}, "avg": None})
    current = None
    for p in paths:
        if not Path(p).exists():
            print(f"[skip] not found: {p}")
            continue
        with open(p, "r", errors="replace") as f:
            for line in f:
                m = TRAIN_RE.search(line)
                if m:
                    current = m.group(1)
                    continue
                if current is None:
                    continue
                m = TEST_RE.search(line)
                if m:
                    task, metric, val = m.group(1), m.group(2), float(m.group(3))
                    # Final eval wins (overwrite earlier evals in same run)
                    runs[current]["per_task"][task] = (metric, val)
                    continue
                m = AVG_RE.search(line)
                if m:
                    runs[current]["avg"] = float(m.group(1))
    return runs


def to_csv(runs, out_path):
    all_tasks = sorted({t for r in runs.values() for t in r["per_task"]})
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run"] + all_tasks + ["avg"])
        for tag, r in sorted(runs.items()):
            row = [tag]
            for t in all_tasks:
                v = r["per_task"].get(t)
                row.append(f"{v[1]:.4f}" if v else "")
            row.append(f"{r['avg']:.4f}" if r["avg"] is not None else "")
            w.writerow(row)
    print(f"wrote {out_path} ({len(runs)} runs, {len(all_tasks)} tasks)")


def to_markdown(runs):
    all_tasks = sorted({t for r in runs.values() for t in r["per_task"]})
    # Shorten task names for the table
    short = {t: t.replace("rel-", "").replace("-pred", "")[:18] for t in all_tasks}
    headers = ["run"] + [short[t] for t in all_tasks] + ["avg"]
    rows = []
    for tag, r in sorted(runs.items()):
        row = [tag]
        for t in all_tasks:
            v = r["per_task"].get(t)
            row.append(f"{v[1]:.3f}" if v else "—")
        row.append(f"{r['avg']:.4f}" if r["avg"] is not None else "—")
        rows.append(row)
    # Print
    def fmt(cells):
        return "| " + " | ".join(c.ljust(8) for c in cells) + " |"
    print(fmt(headers))
    print(fmt(["---"] * len(headers)))
    for row in rows:
        print(fmt(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="One or more tee log files to parse.")
    ap.add_argument("--out", default="smpnn_results.csv",
                    help="Output CSV path.")
    args = ap.parse_args()
    runs = parse_logs(args.logs)
    if not runs:
        print("No runs found.")
        return
    to_csv(runs, args.out)
    print()
    to_markdown(runs)


if __name__ == "__main__":
    main()
