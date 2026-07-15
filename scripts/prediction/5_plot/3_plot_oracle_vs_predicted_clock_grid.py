#!/usr/bin/env python3
"""
Oracle vs predicted global policy: clock-pair selection heatmaps.

2 rows (Oracle / Predicted) × 2 columns (0% and 5% bounds only).
Cell color = number of kernels assigned to that (core, mem) pair.
Zero cells = white; low counts = light green; smooth ramp to dark NVIDIA green.

Outputs to ~/thesis/figures/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# Smooth NVIDIA green ramp: light for low counts, gradual transition to dark.
# (White is only for empty/masked cells, not mapped data values.)
NVIDIA_CMAP = LinearSegmentedColormap.from_list(
    "nvidia_green",
    ["#EEF7EA", "#DCEFD4", "#C5E1A5", "#9CCC65", "#76B900", "#558B2F", "#33691E"],
    N=256,
)


BASE = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"
OUT_DIR = Path.home() / "thesis/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORACLE_CSV = BASE / "measured_global_oracle_selections.csv"
PRED_CSV = BASE / "predicted_global_policy_selections.csv"
SUMMARY_CSV = BASE / "predicted_global_policy_summary.csv"

LIMITS = [0, 5]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Missing expected column. Tried {candidates}\nColumns:\n{list(df.columns)}")


def load_selected(path: Path) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)

    core_col = pick_col(df, [
        "effective_core_clock",
        "applied_core_clock",
        "selected_core_clock",
        "core_clock",
    ])
    mem_col = pick_col(df, [
        "effective_mem_clock",
        "applied_mem_clock",
        "selected_mem_clock",
        "mem_clock",
        "memory_clock",
    ])

    if "constraint_name" not in df.columns:
        raise SystemExit(f"No constraint_name in {path}")

    df[core_col] = pd.to_numeric(df[core_col], errors="coerce")
    df[mem_col] = pd.to_numeric(df[mem_col], errors="coerce")
    df = df.dropna(subset=[core_col, mem_col]).copy()
    df[core_col] = df[core_col].round().astype(int)
    df[mem_col] = df[mem_col].round().astype(int)
    return df, core_col, mem_col


def pick_best_predicted(limit: int) -> tuple[str, str, str]:
    """Best predicted policy that respects the global runtime bound."""
    summary = pd.read_csv(SUMMARY_CSV)
    cname = f"{limit}pct"
    g = summary[summary["constraint_name"] == cname].copy()
    if g.empty:
        raise SystemExit(f"No summary rows for {cname}")

    ok = g[g["global_violates_constraint"] == False]
    if ok.empty:
        print(f"WARN: no violation-free predicted policy at {cname}; using best energy anyway")
        ok = g

    best = ok.sort_values(
        ["actual_global_energy_saving_pct", "global_violation_over_bound_pct"],
        ascending=[False, True],
    ).iloc[0]

    return str(best["constraint_name"]), str(best["feature_set"]), str(best["model"])


def matrix(df, core_col, mem_col, all_cores, all_mems):
    mat = np.zeros((len(all_mems), len(all_cores)), dtype=int)
    for _, r in df.iterrows():
        i = all_mems.index(int(r[mem_col]))
        j = all_cores.index(int(r[core_col]))
        mat[i, j] += 1
    return mat


def text_color(value: int, vmax: int) -> str:
    if vmax <= 0:
        return "#1B5E20"
    return "white" if value >= max(2, 0.45 * vmax) else "#1B5E20"


def main():
    oracle, o_core, o_mem = load_selected(ORACLE_CSV)
    pred, p_core, p_mem = load_selected(PRED_CSV)

    panels = []
    all_cores: set[int] = set()
    all_mems: set[int] = set()
    pred_choices = {}

    for limit in LIMITS:
        cname = f"{limit}pct"
        od = oracle[oracle["constraint_name"] == cname].copy()
        if od.empty:
            raise SystemExit(f"No oracle selections for {cname}")

        pred_constraint, feature_set, model = pick_best_predicted(limit)
        pred_choices[limit] = (pred_constraint, feature_set, model)

        pd_ = pred[
            (pred["constraint_name"] == pred_constraint)
            & (pred["feature_set"] == feature_set)
            & (pred["model"] == model)
        ].copy()
        if pd_.empty:
            raise SystemExit(
                f"No predicted selections for {limit}%: "
                f"{pred_constraint} / {feature_set} / {model}"
            )

        panels.append(("Oracle", limit, od, o_core, o_mem))
        panels.append(("Predicted", limit, pd_, p_core, p_mem))

        all_cores.update(od[o_core].tolist())
        all_cores.update(pd_[p_core].tolist())
        all_mems.update(od[o_mem].tolist())
        all_mems.update(pd_[p_mem].tolist())

    all_cores = sorted(all_cores)
    all_mems = sorted(all_mems)

    mats = []
    vmax = 0
    for label, limit, df, core_col, mem_col in panels:
        mat = matrix(df, core_col, mem_col, all_cores, all_mems)
        mats.append((label, limit, mat))
        vmax = max(vmax, int(mat.max()))

    vmax_int = max(vmax, 1)
    # gamma < 1 keeps 1–3 light, stretches mid/high for visible transition.
    count_norm = PowerNorm(gamma=0.65, vmin=1, vmax=vmax_int)

    fig = plt.figure(figsize=(4.3, 3.4), facecolor="white")

    # One gap value (figure fraction) for all panel spacing (H, V, and to colorbar).
    left, right, top, bottom = 0.11, 0.84, 0.93, 0.17
    panel_gap = 0.018
    cbar_width = 0.028
    plot_w = right - left
    plot_h = top - bottom
    # GridSpec wspace/hspace are relative to average axes size; solve for matching gap.
    wspace = (2 * panel_gap / plot_w) / (1 - panel_gap / plot_w)
    hspace = (2 * panel_gap / plot_h) / (1 - panel_gap / plot_h)

    gs = GridSpec(
        2, 2, figure=fig,
        wspace=wspace,
        hspace=hspace,
        left=left, right=right, top=top, bottom=bottom,
    )
    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(gs[r, c])

    im = None
    for label, limit, mat in mats:
        row = 0 if label == "Oracle" else 1
        col = LIMITS.index(limit)
        ax = axes[row, col]
        ax.set_facecolor("white")

        # Mask zeros so they render as white background.
        mat_show = np.ma.masked_where(mat == 0, mat.astype(float))
        im = ax.imshow(
            mat_show,
            aspect="auto",
            origin="lower",
            cmap=NVIDIA_CMAP,
            norm=count_norm,
            interpolation="nearest",
        )

        if row == 0:
            ax.set_title(f"{limit}% limit", fontsize=8, pad=3)

        if col == 0:
            ax.set_ylabel(f"{label}\nMemory MHz", fontsize=8)

        if row == 1:
            ax.set_xlabel("Core MHz", fontsize=8)

        for i in range(len(all_mems)):
            for j in range(len(all_cores)):
                v = int(mat[i, j])
                if v > 0:
                    ax.text(
                        j, i, str(v),
                        ha="center", va="center",
                        fontsize=6,
                        color=text_color(v, vmax_int),
                    )

        ax.set_xticks(range(len(all_cores)))
        ax.set_yticks(range(len(all_mems)))

        if row == 1:
            ax.set_xticklabels(all_cores, rotation=55, ha="right", fontsize=6)
        else:
            ax.tick_params(labelbottom=False)

        if col == 0:
            ax.set_yticklabels(all_mems, fontsize=6)
        else:
            ax.tick_params(labelleft=False)

    # Colorbar: full panel height; gap matches inter-panel gap above.
    pos_top = axes[0, 1].get_position()
    pos_bot = axes[1, 1].get_position()
    cax = fig.add_axes([
        pos_top.x1 + panel_gap,
        pos_bot.y0,
        cbar_width,
        pos_top.y1 - pos_bot.y0,
    ])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Selected kernels", fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.5)
    # Sparse integer ticks only (e.g. 5, 10, 15, 20 — not every count).
    cbar.locator = MaxNLocator(integer=True, nbins=6)
    cbar.update_ticks()

    out_pdf = OUT_DIR / "fig_oracle_vs_predicted_clock_grid.pdf"
    out_png = OUT_DIR / "fig_oracle_vs_predicted_clock_grid.png"

    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, facecolor="white", transparent=True, bbox_inches="tight")

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print()
    print("Predicted policies used:")
    for limit, (c, fs, m) in pred_choices.items():
        print(f"  {limit}%: constraint={c}, feature_set={fs}, model={m}")


if __name__ == "__main__":
    main()
