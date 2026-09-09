#!/usr/bin/env python3
"""Plot α trajectories from smpnn_alpha_evolution.csv.

Produces two subplots (α_gnn on left, α_ff on right). Each subplot has
one curve per (alpha_init_tag, layer) pair, with light shading showing
±1 std across seeds. X-axis is epoch, Y-axis is α value on log scale.

Usage:
    python plot_alpha_evolution.py \\
        --csv smpnn_alpha_evolution.csv \\
        --out figs/alpha_evolution.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_one(ax, df: pd.DataFrame, value_col: str, title: str) -> None:
    """Plot one α family (gnn or ff) across all (tag, layer) pairs."""
    for tag, tag_df in df.groupby("alpha_init_tag"):
        for layer, layer_df in tag_df.groupby("layer"):
            # Aggregate across seeds
            agg = layer_df.groupby("epoch")[value_col].agg(["mean", "std"])
            epochs = agg.index.values
            means = agg["mean"].values
            stds = agg["std"].values

            label = f"{tag} L{layer}"
            line = ax.plot(epochs, means, marker="o", markersize=3,
                           linewidth=1.2, label=label)[0]
            ax.fill_between(
                epochs,
                np.maximum(means - stds, 1e-10),
                means + stds,
                alpha=0.15,
                color=line.get_color(),
            )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{value_col} (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=7, ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to smpnn_alpha_evolution.csv")
    ap.add_argument("--out", required=True,
                    help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"{args.csv} is empty")

    has_ff = df["alpha_ff"].notna().any()
    n_subplots = 2 if has_ff else 1

    fig, axes = plt.subplots(
        1, n_subplots, figsize=(6 * n_subplots, 5), sharey=False,
    )
    if n_subplots == 1:
        axes = [axes]

    plot_one(axes[0], df, "alpha_gnn",
             "α_gnn evolution (sub-block 1 residual scale)")
    if has_ff:
        # Filter out rows with NaN ff (some ablations don't have alpha_ff)
        ff_df = df.dropna(subset=["alpha_ff"])
        plot_one(axes[1], ff_df, "alpha_ff",
                 "α_ff evolution (sub-block 2 residual scale)")

    fig.suptitle(
        "SMPNN α evolution — 20 epochs on others-1, "
        f"n={df['seed'].nunique()} seeds (mean ± std)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
