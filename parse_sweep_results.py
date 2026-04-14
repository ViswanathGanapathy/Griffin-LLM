#!/usr/bin/env python
"""Parse a fine-tune sweep log and print per-task metrics vs N as a table.

Usage:
    python parse_sweep_results.py logs/ft-unified-sweep.log
    python parse_sweep_results.py logs/ft-c1c2-sweep.log logs/ft-c2c1-sweep.log
"""

import re
import sys
from collections import defaultdict


N_PATTERN = re.compile(r"=== N=(\d+)")
METRIC_PATTERN = re.compile(
    r"test_metric/([\w\-]+)/([\w_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
AVG_PATTERN = re.compile(r"Average test metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(path):
    """Return {N: {task: (metric_name, value)}, N: {'__avg__': value}}."""
    results = defaultdict(dict)
    current_n = None
    with open(path) as f:
        for line in f:
            m = N_PATTERN.search(line)
            if m:
                current_n = int(m.group(1))
                continue
            if current_n is None:
                continue
            m = METRIC_PATTERN.search(line)
            if m:
                task, metric, val = m.group(1), m.group(2), float(m.group(3))
                results[current_n][task] = (metric, val)
                continue
            m = AVG_PATTERN.search(line)
            if m:
                results[current_n]["__avg__"] = ("avg", float(m.group(1)))
    return results


def format_table(results, label=""):
    Ns = sorted(results.keys())
    if not Ns:
        print(f"[{label}] no results parsed")
        return
    tasks = sorted({t for n in Ns for t in results[n] if t != "__avg__"})

    # Header
    print(f"\n=== {label} ===")
    header = ["task \\ N"] + [str(n) for n in Ns]
    widths = [max(len(h), 10) for h in header]
    widths[0] = max(widths[0], max(len(t) for t in tasks + ["__avg__"]))

    def row(cells):
        return " | ".join(c.rjust(w) for c, w in zip(cells, widths))

    print(row(header))
    print("-+-".join("-" * w for w in widths))

    for t in tasks:
        metric_name = next(
            (results[n][t][0] for n in Ns if t in results[n]), ""
        )
        label_cell = f"{t} ({metric_name})"
        cells = [label_cell]
        for n in Ns:
            if t in results[n]:
                cells.append(f"{results[n][t][1]:.4f}")
            else:
                cells.append("—")
        # Adjust first-column width on the fly if task label is wider
        if len(label_cell) > widths[0]:
            widths[0] = len(label_cell)
        print(row(cells))

    # Average row
    cells = ["AVERAGE"]
    for n in Ns:
        if "__avg__" in results[n]:
            cells.append(f"{results[n]['__avg__'][1]:.4f}")
        else:
            cells.append("—")
    print(row(cells))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for path in sys.argv[1:]:
        results = parse_log(path)
        format_table(results, label=path)


if __name__ == "__main__":
    main()
