#!/usr/bin/env python3
"""Per-launch duration from the 5s oracle measure window.

    duration_per_launch_ms = total_time_mean_ms / measured_launches_mean

In:  ~/thesis/results/master/aggregated_all_kernels_effective.csv
Out: printed table + ~/thesis/results/master/kernel_duration_auto.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IN_CSV = Path.home() / "thesis/results/master/aggregated_all_kernels_effective.csv"
OUT_CSV = Path.home() / "thesis/results/master/kernel_duration_auto.csv"


def kernel_durations(df: pd.DataFrame, clock: str = "auto") -> pd.DataFrame:
    core = df["requested_core_clock"].astype(str).str.lower()
    mem = df["requested_mem_clock"].astype(str).str.lower()

    if clock == "auto":
        sub = df[(core == "auto") & (mem == "auto")].copy()
    else:
        sub = df.copy()

    if sub.empty:
        raise SystemExit("No matching rows found.")

    launches = sub["measured_launches_mean"].astype(float)
    total_ms = sub["total_time_mean_ms"].astype(float)

    if "time_per_launch_mean_ms" in sub.columns:
        dur_ms = sub["time_per_launch_mean_ms"].astype(float)
    else:
        dur_ms = total_ms / launches

    out = pd.DataFrame({
        "workload": sub["workload"],
        "kernel": sub["kernel"],
        "measure_window_ms": total_ms.round(3),
        "launches_in_window": launches.round(0).astype(int),
        "duration_per_launch_ms": dur_ms,
        "duration_per_launch_us": (dur_ms * 1000.0).round(3),
        "launches_per_second": (launches / (total_ms / 1000.0)).round(1),
    })

    if "energy_per_launch_mean_mJ" in sub.columns:
        out["energy_per_launch_mJ"] = sub["energy_per_launch_mean_mJ"].astype(float).round(4)

    return out.sort_values("duration_per_launch_ms", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-kernel duration from 5s oracle measurement windows.",
    )
    parser.add_argument("--input", type=Path, default=IN_CSV, help="Aggregated oracle CSV")
    parser.add_argument("--workload", type=str, default="", help="Filter one workload")
    parser.add_argument("--kernel", type=str, default="", help="Filter one kernel")
    parser.add_argument(
        "--all-clocks",
        action="store_true",
        help="Include all clock configs, not just auto/auto",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_CSV,
        help=f"Write CSV here (default: {OUT_CSV})",
    )
    parser.add_argument("--no-out", action="store_true", help="Skip writing CSV")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing: {args.input}")

    df = pd.read_csv(args.input)

    if args.workload:
        df = df[df["workload"].astype(str).str.lower() == args.workload.lower()]
    if args.kernel:
        df = df[df["kernel"].astype(str).str.lower() == args.kernel.lower()]

    if args.all_clocks:
        rows = []
        for (core, mem), g in df.groupby(
            [df["requested_core_clock"].astype(str), df["requested_mem_clock"].astype(str)],
            sort=False,
        ):
            part = kernel_durations(g, clock="all")
            part["requested_core_clock"] = core
            part["requested_mem_clock"] = mem
            rows.append(part)
        result = pd.concat(rows, ignore_index=True)
        result = result.sort_values(
            ["duration_per_launch_ms", "workload", "kernel"],
            ascending=[False, True, True],
        )
    else:
        result = kernel_durations(df)

    print(f"Kernels: {result['workload'].nunique()} workloads, {len(result)} kernel rows")
    print(f"Formula: duration_per_launch_ms = measure_window_ms / launches_in_window")
    print()

    show_cols = [
        "workload",
        "kernel",
        "launches_in_window",
        "measure_window_ms",
        "duration_per_launch_ms",
        "duration_per_launch_us",
        "launches_per_second",
    ]
    if args.all_clocks:
        show_cols = ["requested_core_clock", "requested_mem_clock"] + show_cols
    if "energy_per_launch_mJ" in result.columns:
        show_cols.append("energy_per_launch_mJ")

    print(result[show_cols].to_string(index=False))

    if not args.no_out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
