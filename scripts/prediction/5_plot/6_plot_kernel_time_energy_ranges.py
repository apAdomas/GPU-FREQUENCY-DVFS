#!/usr/bin/env python3
"""
Per-kernel time and energy across all measured clock configurations.

Dot   = auto configuration.
Bar   = [min, max] achieved by ANY clock config (auto included so the dot is
        always inside the bar).

PolyBench and Rodinia only (Parboil is excluded on purpose).

Suite is inferred from the full executable path, which is the only reliable
signal: workload names are lower-cased (2dconv, jacobi2d, srad_v1) and
source_folder is a bare folder name (2DCONV, euler3d), so neither matches a
"polybench"/"rodinia" substring on its own.
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", PerformanceWarning)


DEFAULT_IN = Path.home() / "thesis/results/master/ml_dataset_ptx_launch_runtime_normalized.csv"
DEFAULT_OUT_DIR = Path.home() / "thesis/figures"

PATH_COLS = ["executable", "source_cu_ncu", "source_folder", "source_ptx"]


def infer_suite(row: pd.Series) -> str:
    """Classify a kernel by the full executable/source path."""
    for col in PATH_COLS:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).lower()
            if "polybench" in s:
                return "PolyBench"
            if "rodinia" in s:
                return "Rodinia"
            if "parboil" in s:
                return "Parboil"
    return "Other"


def build_kernel_table(df: pd.DataFrame) -> pd.DataFrame:
    if "kernel_id" not in df.columns:
        df = df.assign(
            kernel_id=df["join_workload"].astype(str) + "::" + df["join_kernel"].astype(str)
        )

    required = [
        "kernel_id",
        "workload",
        "kernel",
        "time_per_launch_mean_ms",
        "energy_per_launch_mean_mJ",
        "auto_time_per_launch_mean_ms",
        "auto_energy_per_launch_mean_mJ",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    rows = []
    for kid, g in df.groupby("kernel_id", sort=False):
        first = g.iloc[0]

        auto_time_ms = float(first["auto_time_per_launch_mean_ms"])
        auto_energy_mj = float(first["auto_energy_per_launch_mean_mJ"])

        fixed_time = pd.to_numeric(g["time_per_launch_mean_ms"], errors="coerce").dropna()
        fixed_energy = pd.to_numeric(g["energy_per_launch_mean_mJ"], errors="coerce").dropna()

        if fixed_time.empty or fixed_energy.empty:
            print(f"WARN: {kid} has no usable time/energy rows, skipping")
            continue

        # Include auto so the error bar always contains the dot.
        min_time_ms = min(auto_time_ms, float(fixed_time.min()))
        max_time_ms = max(auto_time_ms, float(fixed_time.max()))
        min_energy_mj = min(auto_energy_mj, float(fixed_energy.min()))
        max_energy_mj = max(auto_energy_mj, float(fixed_energy.max()))

        rows.append({
            "kernel_id": kid,
            "workload": first["workload"],
            "kernel": first["kernel"],
            "suite": infer_suite(first),
            "n_configs": int(len(g)),

            "auto_time_s": auto_time_ms / 1000.0,
            "min_time_s": min_time_ms / 1000.0,
            "max_time_s": max_time_ms / 1000.0,

            "auto_energy_j": auto_energy_mj / 1000.0,
            "min_energy_j": min_energy_mj / 1000.0,
            "max_energy_j": max_energy_mj / 1000.0,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sort",
        choices=["auto_time", "auto_energy", "suite", "workload"],
        default="auto_time",
        help="Kernel ordering on the x-axis.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    k = build_kernel_table(df)

    # PolyBench + Rodinia only.
    before = len(k)
    dropped = k[~k["suite"].isin(["PolyBench", "Rodinia"])]
    if len(dropped):
        print("Dropped non-PolyBench/Rodinia kernels:")
        print(dropped[["kernel_id", "suite"]].to_string(index=False))
    k = k[k["suite"].isin(["PolyBench", "Rodinia"])].copy()
    print(f"Kept {len(k)} of {before} kernels "
          f"(PolyBench: {(k['suite'] == 'PolyBench').sum()}, "
          f"Rodinia: {(k['suite'] == 'Rodinia').sum()})")

    if k.empty:
        raise SystemExit("No PolyBench/Rodinia kernels found; check input/classification.")

    if args.sort == "auto_time":
        k = k.sort_values(["auto_time_s", "suite", "workload", "kernel"],
                          ascending=[False, True, True, True])
    elif args.sort == "auto_energy":
        k = k.sort_values(["auto_energy_j", "suite", "workload", "kernel"],
                          ascending=[False, True, True, True])
    elif args.sort == "suite":
        k = k.sort_values(["suite", "workload", "kernel"])
    else:
        k = k.sort_values(["workload", "kernel"])

    k = k.reset_index(drop=True)
    k["x"] = np.arange(len(k))

    # Asymmetric error bars around the auto dot (clip tiny negatives from FP).
    time_yerr = np.vstack([
        (k["auto_time_s"] - k["min_time_s"]).clip(lower=0.0),
        (k["max_time_s"] - k["auto_time_s"]).clip(lower=0.0),
    ])
    energy_yerr = np.vstack([
        (k["auto_energy_j"] - k["min_energy_j"]).clip(lower=0.0),
        (k["max_energy_j"] - k["auto_energy_j"]).clip(lower=0.0),
    ])

    suite_style = {
        # NVIDIA-inspired mono greens: darker PolyBench, brighter Rodinia.
        "PolyBench": {"color": "#2E7D32"},
        "Rodinia": {"color": "#76B900"},
    }

    fig, axes = plt.subplots(
        2, 1,
        figsize=(6.2, 5.0),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    for suite, style in suite_style.items():
        s = k[k["suite"] == suite]
        if s.empty:
            continue
        idx = s.index.to_numpy()

        axes[0].errorbar(
            s["x"], s["auto_time_s"], yerr=time_yerr[:, idx],
            fmt="s", markersize=4.0, capsize=1.8,
            elinewidth=0.85, markeredgewidth=0.55, linestyle="none",
            color=style["color"], alpha=1.0, label=suite,
        )
        axes[1].errorbar(
            s["x"], s["auto_energy_j"], yerr=energy_yerr[:, idx],
            fmt="s", markersize=4.0, capsize=1.8,
            elinewidth=0.85, markeredgewidth=0.55, linestyle="none",
            color=style["color"], alpha=1.0, label=suite,
        )

    for ax in axes:
        ax.set_xlim(-0.8, len(k) - 0.2)
        ax.margins(x=0)
        ax.minorticks_off()
        ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
        ax.grid(False, which="minor")
        ax.tick_params(axis="y", which="minor", length=0)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Time per launch (s)")
    axes[1].set_ylabel("Energy per launch (J)")
    axes[1].set_xlabel("Kernel rank, sorted by automatic-clock runtime")

    # Suite labels: right-aligned, just above the top panel.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        bbox_transform=axes[0].transAxes,
        ncol=2, fontsize=8, frameon=False,
        borderaxespad=0, handletextpad=0.4, columnspacing=0.8,
    )

    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.10, hspace=0.08)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "fig_kernel_time_energy_ranges.png"
    out_pdf = args.out_dir / "fig_kernel_time_energy_ranges.pdf"
    summary_csv = args.out_dir / "fig_kernel_time_energy_ranges_data.csv"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    k.drop(columns=["x"]).to_csv(summary_csv, index=False)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {summary_csv}  (per-kernel auto/min/max values used in the figure)")

    print()
    print("Largest kernels by auto time:")
    print(k[["kernel_id", "suite", "auto_time_s", "auto_energy_j", "n_configs"]]
          .head(15).to_string(index=False))


if __name__ == "__main__":
    main()
