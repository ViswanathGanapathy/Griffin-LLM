#!/usr/bin/env python3
"""Parse α trajectories from run_smpnn_alpha_evolution.sh training logs.

The training script prints per-epoch α values in the form:
    Epoch 0 starts
      [alpha] L0(gnn=1.00e-06,ff=1.00e-06) L1(gnn=1.00e-06,ff=1.00e-06) ...
    Epoch 1 starts
      [alpha] L0(gnn=1.32e-05,ff=8.71e-06) L1(...) ...

This script scans one or more logs and emits a CSV with one row per
(seed, alpha_init_tag, epoch, layer, alpha_gnn, alpha_ff).

Usage:
    python parse_alpha_evolution.py \\
        --logs "logs/smpnn-alpha-evolution/*.log" \\
        --out smpnn_alpha_evolution.csv
"""
import argparse
import csv
import glob
import os
import re
from typing import Optional


# Filename convention from run_smpnn_alpha_evolution.sh:
#   s{SEED}-{ALPHA_TAG}-alpha-evo.log
_TAG_RE = re.compile(r"s(?P<seed>\d+)-(?P<tag>[a-z0-9]+)-alpha-evo\.log$")

# Epoch marker (immediately precedes the [alpha] line for that epoch)
_EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s+starts")

# [alpha] block; captures each L<i>(gnn=<g>,ff=<f>) or L<i>(gnn=<g>)
_ALPHA_LINE_RE = re.compile(r"\[alpha\]\s+(.*)$")
_LAYER_RE = re.compile(
    r"L(?P<i>\d+)\("
    r"gnn=(?P<gnn>-?[\d.]+e[+-]?\d+)"
    r"(?:,ff=(?P<ff>-?[\d.]+e[+-]?\d+))?"
    r"\)"
)


def parse_log(path: str) -> list[dict]:
    """Return a list of {seed, alpha_init_tag, epoch, layer, alpha_gnn,
    alpha_ff} rows for one log file."""
    m = _TAG_RE.search(os.path.basename(path))
    if not m:
        print(f"[warn] skipping {path} — filename does not match "
              "s<SEED>-<TAG>-alpha-evo.log")
        return []
    seed = int(m.group("seed"))
    alpha_tag = m.group("tag")

    rows: list[dict] = []
    current_epoch: Optional[int] = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            em = _EPOCH_RE.search(line)
            if em:
                current_epoch = int(em.group(1))
                continue
            am = _ALPHA_LINE_RE.search(line)
            if am and current_epoch is not None:
                for lm in _LAYER_RE.finditer(am.group(1)):
                    rows.append({
                        "seed": seed,
                        "alpha_init_tag": alpha_tag,
                        "epoch": current_epoch,
                        "layer": int(lm.group("i")),
                        "alpha_gnn": float(lm.group("gnn")),
                        "alpha_ff": (
                            float(lm.group("ff"))
                            if lm.group("ff") is not None else None
                        ),
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="Log files or globs, e.g. "
                         "'logs/smpnn-alpha-evolution/*.log'")
    ap.add_argument("--out", required=True,
                    help="Output CSV path")
    args = ap.parse_args()

    # Expand any glob patterns in --logs
    log_files: list[str] = []
    for pattern in args.logs:
        matches = sorted(glob.glob(pattern))
        if not matches and os.path.isfile(pattern):
            matches = [pattern]
        log_files.extend(matches)
    if not log_files:
        raise SystemExit(f"No files matched --logs={args.logs}")

    all_rows: list[dict] = []
    for lf in log_files:
        rows = parse_log(lf)
        print(f"  {lf}: {len(rows)} α rows")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(
            "No α rows parsed. Was the training run with "
            "--log_alpha_every > 0?"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "alpha_init_tag", "epoch", "layer",
                        "alpha_gnn", "alpha_ff"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
