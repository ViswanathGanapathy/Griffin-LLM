"""Small-multiple histograms of hop-1/2/3 neighborhood sizes, 4 seed types."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os as _os
SCRATCH = _os.path.dirname(_os.path.abspath(__file__))
hm = np.load(f"{SCRATCH}/hops_hm.npz")
rb = np.load(f"{SCRATCH}/hops_relbench.npz")

ROWS = [
    ("rel-hm", "customer seed", hm["customer"]),
    ("rel-hm", "article seed", hm["article"]),
    ("rel-salt", "salesdocument seed", rb["salt"]),
    ("rel-ratebeer", "user seed", rb["ratebeer"]),
]
HOP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (light)
SURFACE, GRID, BASELINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

fig, axes = plt.subplots(4, 3, figsize=(11.5, 11), dpi=150,
                         facecolor=SURFACE, sharey=False)
fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.075,
                    hspace=0.52, wspace=0.22)

# column-shared x ranges so datasets compare vertically
col_max = [max(r[2][:, h].max() for r in ROWS) for h in range(3)]

def fmt_count(x, _pos=None):
    if x >= 1e6: return f"{x/1e6:.3g}M"
    if x >= 1e3: return f"{x/1e3:.3g}K"
    return f"{x:.3g}"

for i, (ds, seed, data) in enumerate(ROWS):
    for h in range(3):
        ax = axes[i][h]
        vals = data[:, h]
        nz = vals[vals > 0]
        pct_zero = 100.0 * (len(vals) - len(nz)) / len(vals)
        # integer-snapped log bins: no comb artifact below ~40
        edges = np.unique(np.round(np.logspace(0, np.log10(col_max[h] * 1.3), 44)))
        bins = np.concatenate([[0.7], edges + 0.5])
        w = np.full(len(nz), 100.0 / len(vals))
        ax.hist(nz, bins=bins, weights=w, color=HOP_COLORS[h],
                rwidth=0.88, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.7, col_max[h] * 1.3)

        med, p90 = np.median(vals), np.percentile(vals, 90)
        top = ax.get_ylim()[1]
        # med label sits left of its line, p90 right of its line — no collision
        for v, name, ha, dx in [(med, "med", "right", 1 / 1.15),
                                (p90, "p90", "left", 1.15)]:
            ax.axvline(v, color=INK2, lw=1, ls=(0, (3, 2)), zorder=4)
            ax.text(v * dx, top * 1.02, f"{name} {fmt_count(v)}", color=INK2,
                    fontsize=6.6, ha=ha, va="bottom", clip_on=False)
        if pct_zero > 0.5:
            ax.text(0.02, 0.93, f"{pct_zero:.1f}% zero", transform=ax.transAxes,
                    fontsize=6.6, color=MUTED, va="top")

        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=7, length=3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_count))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}%"))
        if i == 0:
            ax.set_title(f"hop {h+1}", fontsize=11, color=INK, pad=22,
                         fontweight="bold")
        if h == 0:
            ax.text(-0.22, 0.5, f"{ds}\n({seed})", transform=ax.transAxes,
                    fontsize=9, color=INK, ha="center", va="center",
                    rotation=90, fontweight="bold")

fig.suptitle("How many neighbors does one row reach?", x=0.10, y=0.975,
             ha="left", fontsize=14, color=INK, fontweight="bold")
fig.text(0.10, 0.945,
         "Share of sampled seed rows (y) by k-hop neighborhood size (x, log scale). "
         "Dashed lines: median and 90th percentile.",
         fontsize=9, color=INK2)
fig.text(0.10, 0.022,
         "Samples: 20,000 seeds each for rel-hm customer/article (joint-v65); 4,000 rel-salt documents; "
         "1,500 rel-ratebeer users (raw RelBench v2 parquet). Unique-node counts; x-axes shared per column.",
         fontsize=7.5, color=MUTED)

out = "/home/pviswanath/Griffin/sprints/v2/figs/neighborhood_hop_distributions.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
