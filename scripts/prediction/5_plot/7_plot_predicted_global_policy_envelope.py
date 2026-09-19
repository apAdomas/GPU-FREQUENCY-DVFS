#!/usr/bin/env python3
"""Global-policy scatter per slowdown bound (0/1/2/5%).

Circle = no runtime features, square = with runtime features.
Grey = feasible, red = bound violation, large green = best feasible.
Black triangle = measured global oracle. Writes ~/thesis/figures/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"
OUT_DIR = Path.home() / "thesis/figures"
SUMMARY_CSV = BASE / "predicted_global_policy_summary.csv"

BOUNDS = ["0pct", "1pct", "2pct", "5pct"]
BOUND_LABELS = {"0pct": "0%", "1pct": "1%", "2pct": "2%", "5pct": "5%"}

GREY = "#9AA0A6"
RED = "#E65A55"
GREEN = "#76B900"
ORACLE = "#1f1f1f"
BEST_EDGE = "#2E7D32"

SMALL = 85
LARGE = 130


def has_runtime(feature_set: str) -> bool:
    return "runtime" in str(feature_set)


def marker_for(feature_set: str) -> str:
    return "s" if has_runtime(feature_set) else "o"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["constraint_name"].isin(BOUNDS)].copy()

    x_of = {b: i for i, b in enumerate(BOUNDS)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    ax.set_facecolor("white")

    for bound in BOUNDS:
        g = df[df["constraint_name"] == bound]
        if g.empty:
            continue

        x0 = x_of[bound]

        valid = g[~g["global_violates_constraint"].astype(bool)]
        best_idx = (
            valid["actual_global_energy_saving_pct"].idxmax()
            if not valid.empty else None
        )

        for idx, row in g.iterrows():
            violates = bool(row["global_violates_constraint"])
            is_best = idx == best_idx
            y = float(row["actual_global_energy_saving_pct"])
            mk = marker_for(row["feature_set"])

            if violates:
                color, size, alpha, edge, lw = RED, SMALL, 0.6, "none", 0.0
                x = x0 + rng.uniform(-0.16, 0.16)
            elif is_best:
                color, size, alpha, edge, lw = GREEN, LARGE, 1.0, BEST_EDGE, 1.0
                x = x0
            else:
                color, size, alpha, edge, lw = GREY, SMALL, 0.65, "none", 0.0
                x = x0 + rng.uniform(-0.16, 0.16)

            ax.scatter(
                x, y,
                s=size, marker=mk, c=color, alpha=alpha,
                edgecolors=edge, linewidths=lw, zorder=5 if is_best else 2,
            )

            if is_best:
                ax.annotate(
                    f"{y:.1f}%",
                    (x, y),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=8, color="black",
                )

        oracle_val = g["measured_oracle_global_energy_saving_pct"].dropna()
        if not oracle_val.empty:
            oracle_y_val = float(oracle_val.iloc[0])
            ax.scatter(
                x0, oracle_y_val,
                s=LARGE, marker="^", c=ORACLE, edgecolors=ORACLE,
                linewidths=1.0, zorder=6,
            )
            ax.annotate(
                f"{oracle_y_val:.1f}%",
                (x0, oracle_y_val),
                textcoords="offset points", xytext=(10, 5),
                fontsize=8, color="black",
            )

    # Zoom y-axis to data (skip empty 0–6% band).
    y_vals = df["actual_global_energy_saving_pct"].tolist()
    y_vals += df["measured_oracle_global_energy_saving_pct"].dropna().tolist()
    y_lo = min(y_vals)
    y_hi = max(y_vals)
    pad = 0.8
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_xticks(list(x_of.values()))
    ax.set_xticklabels([BOUND_LABELS[b] for b in BOUNDS])
    ax.set_xlim(-0.4, len(BOUNDS) - 0.6)
    ax.set_xlabel("Allowed slowdown bound")
    ax.set_ylabel("Energy saving vs automatic clocks (%)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    shape_edge = "#666666"
    shape_lw = 1.0
    legend_handles = [
        Line2D([0], [0], marker="^", color="none",
               markerfacecolor=ORACLE, markeredgecolor=ORACLE,
               markersize=11, linestyle="None", label="Measured global oracle"),
        Patch(facecolor=GREEN, edgecolor=BEST_EDGE,
              label="Best within-bound prediction"),
        Patch(facecolor=RED, edgecolor="none",
              label="Bound violation"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor="none", markeredgecolor=shape_edge,
               markeredgewidth=shape_lw, markersize=8, fillstyle="none",
               linestyle="None", label="Runtime"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="none", markeredgecolor=shape_edge,
               markeredgewidth=shape_lw, markersize=8, fillstyle="none",
               linestyle="None", label="Non-runtime"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.45,
    )

    out_png = OUT_DIR / "fig_predicted_global_all_policies.png"
    out_pdf = OUT_DIR / "fig_predicted_global_all_policies.pdf"

    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
