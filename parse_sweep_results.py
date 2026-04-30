#!/usr/bin/env python
"""Parse a fine-tune sweep log and print per-task metrics vs N as a table.

Usage:
    python parse_sweep_results.py logs/ft-unified-sweep.log
    python parse_sweep_results.py logs/ft-c1c2-sweep.log logs/ft-c2c1-sweep.log
"""

import re
import sys
from collections import defaultdict


# Matches "=== ... ===" markers. Captures everything between the
# triple-equals as the column key — supports both N-only sweeps
# (=== N=1024 ===) and grid sweeps (=== N=1024 LR=3e-4 epochs=5 ===).
N_PATTERN = re.compile(r"=== (.+?) ===")
# Matches both formats:
#   test_metric/<task>/<metric>: <val>   (full eval path)
#   test_metric/<task>: <val>            (per-task fine-tune path)
METRIC_PATTERN = re.compile(
    r"test_metric/([\w\-]+)(?:/([\w_]+))?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
AVG_PATTERN = re.compile(r"Average test metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(path):
    """Return {marker: {task: (metric_name, value)}, marker: {'__avg__': value}}.

    `marker` is the full text between the triple-equals separators in the log
    (e.g. 'N=1024' for simple sweeps, 'N=1024 LR=3e-4 epochs=5' for grid sweeps).
    """
    results = defaultdict(dict)
    current = None
    with open(path) as f:
        for line in f:
            m = N_PATTERN.search(line)
            if m:
                current = m.group(1).strip()
                # Strip trailing parens like "(zero-shot, no fine-tuning)"
                current = re.sub(r"\s*\(.*?\)\s*$", "", current).strip()
                continue
            if current is None:
                continue
            m = METRIC_PATTERN.search(line)
            if m:
                task = m.group(1)
                metric = m.group(2) or ""
                val = float(m.group(3))
                results[current][task] = (metric, val)
                continue
            m = AVG_PATTERN.search(line)
            if m:
                results[current]["__avg__"] = ("avg", float(m.group(1)))
    return results


def _sort_marker_key(m):
    """Sort markers numerically by extracted N (or 0 if not present),
    then alphabetically. Lets 'N=1024' come before 'N=2048' before 'N=4096'."""
    n_match = re.search(r"N=(\d+)", m)
    n_val = int(n_match.group(1)) if n_match else 0
    return (n_val, m)


def format_table(results, label=""):
    Ns = sorted(results.keys(), key=_sort_marker_key)
    if not Ns:
        print(f"[{label}] no results parsed")
        return
    tasks = sorted({t for n in Ns for t in results[n] if t != "__avg__"})

    # Header
    print(f"\n=== {label} ===")
    header = ["task \\ N"] + [str(n) for n in Ns] + ["best", "@N"]
    widths = [max(len(h), 10) for h in header]
    widths[0] = max(widths[0], max(len(t) for t in tasks + ["__avg__"]))

    def row(cells):
        return " | ".join(c.rjust(w) for c, w in zip(cells, widths))

    print(row(header))
    print("-+-".join("-" * w for w in widths))

    # Track best N per task for summary
    best_per_task = {}

    for t in tasks:
        metric_name = next(
            (results[n][t][0] for n in Ns if t in results[n]), ""
        )
        label_cell = f"{t} ({metric_name})"
        cells = [label_cell]
        task_vals = []  # (N, value)
        for n in Ns:
            if t in results[n]:
                v = results[n][t][1]
                cells.append(f"{v:.4f}")
                task_vals.append((n, v))
            else:
                cells.append("—")
        # All metrics are higher-is-better (regression metrics are pre-negated)
        if task_vals:
            best_n, best_v = max(task_vals, key=lambda x: x[1])
            best_per_task[t] = (best_n, best_v)
            cells.append(f"{best_v:.4f}")
            cells.append(str(best_n))
        else:
            cells.extend(["—", "—"])
        # Adjust first-column width on the fly if task label is wider
        if len(label_cell) > widths[0]:
            widths[0] = len(label_cell)
        print(row(cells))

    # Average row (over the per-N averages reported in the log)
    cells = ["AVERAGE"]
    avg_vals = []
    for n in Ns:
        if "__avg__" in results[n]:
            v = results[n]["__avg__"][1]
            cells.append(f"{v:.4f}")
            avg_vals.append((n, v))
        else:
            cells.append("—")
    if avg_vals:
        best_n, best_v = max(avg_vals, key=lambda x: x[1])
        cells.append(f"{best_v:.4f}")
        cells.append(str(best_n))
    else:
        cells.extend(["—", "—"])
    print(row(cells))

    # Oracle row: best-per-task picked independently, averaged
    if best_per_task:
        oracle_avg = sum(v for _, v in best_per_task.values()) / len(best_per_task)
        print()
        print(f"Oracle avg (best-N picked per-task): {oracle_avg:.4f}")
        print("Best N per task:")
        for t, (n, v) in best_per_task.items():
            print(f"  {t:45s} {str(n):<28s} {v:.4f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        results = parse_log(path)
        format_table(results, label=path)


if __name__ == "__main__":
    main()
