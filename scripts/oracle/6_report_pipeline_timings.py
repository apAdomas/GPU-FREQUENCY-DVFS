#!/usr/bin/env python3
"""
Summarize recorded pipeline timings for thesis reporting.

Sources:
  - NCU profiling wall time: results/ncu_features_all.csv  (column ncu_wall_s)
  - Oracle energy measurements: */summary.log under results/*_kernel_energy/
    (configured warmup/measure windows; wall clock NOT logged for main pipeline)
  - Model training: NOT recorded by default (see note in output)

For skewed per-run durations, report the median as the typical cost and the mean
for total-budget estimates. Both are printed; a LaTeX-ready line uses median.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

THESIS = Path.home() / "thesis"
RESULTS = THESIS / "results"
DEFAULT_NCU = RESULTS / "ncu_features_all.csv"
DEFAULT_NCU_PARBOIL = RESULTS / "ncu_features_parboil.csv"
OUT_CSV = RESULTS / "master" / "pipeline_timing_summary.csv"


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.2f}min"
    return f"{s / 3600:.2f}h"


def summarize_series(name: str, values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if v.empty:
        return {"stage": name, "n": 0}

    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    return {
        "stage": name,
        "n": int(len(v)),
        "mean_s": float(v.mean()),
        "median_s": float(v.median()),
        "std_s": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "p25_s": float(q1),
        "p75_s": float(q3),
        "p90_s": float(v.quantile(0.90)),
        "min_s": float(v.min()),
        "max_s": float(v.max()),
        "total_s": float(v.sum()),
    }


def print_row(row: dict) -> None:
    if row.get("n", 0) == 0:
        print(f"  {row['stage']}: no data")
        return
    print(f"  {row['stage']} (n={row['n']})")
    print(f"    median: {row['median_s']:.2f}s   mean: {row['mean_s']:.2f}s   "
          f"IQR: [{row['p25_s']:.2f}, {row['p75_s']:.2f}]s")
    print(f"    p90: {row['p90_s']:.2f}s   min: {row['min_s']:.2f}s   max: {row['max_s']:.2f}s")
    print(f"    total (sum): {fmt_seconds(row['total_s'])}")


def load_ncu(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [{label}] missing: {path}")
        return None
    df = pd.read_csv(path, usecols=lambda c: c in {
        "ncu_wall_s", "ncu_returncode", "workload", "kernel", "join_workload", "join_kernel",
    })
    df["ncu_wall_s"] = pd.to_numeric(df["ncu_wall_s"], errors="coerce")
    df = df.dropna(subset=["ncu_wall_s"])
    df["source_file"] = label
    return df


def scan_oracle_measurements(results: Path) -> pd.DataFrame:
    rows = []
    for path in results.rglob("summary.log"):
        if "_kernel_energy/" not in str(path).replace("\\", "/"):
            continue
        text = path.read_text(errors="ignore")
        if "measure_seconds=" not in text:
            continue
        row = {"path": str(path)}
        for key in ("warmup_seconds", "measure_seconds", "elapsed_s"):
            m = re.search(rf"^{key}=(.+)$", text, re.MULTILINE)
            if m:
                try:
                    row[key] = float(m.group(1).strip())
                except ValueError:
                    pass
        if "warmup_seconds" in row and "measure_seconds" in row:
            row["configured_cuda_window_s"] = row["warmup_seconds"] + row["measure_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Report pipeline timing statistics.")
    parser.add_argument("--ncu", type=Path, default=DEFAULT_NCU)
    parser.add_argument("--ncu-parboil", type=Path, default=DEFAULT_NCU_PARBOIL)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    print("Pipeline timing summary")
    print("=" * 60)

    summary_rows = []

    # --- NCU profiling (wall clock recorded) ---
    print("\n1) NCU profiling (Nsight Compute)")
    print(f"   Recorded in: {args.ncu}  [column: ncu_wall_s]")
    print(f"   Produced by: scripts/prediction/1_feature_extraction/3_profile_ncu_features.py")

    ncu_parts = []
    for path, label in [(args.ncu, "polybench_rodinia"), (args.ncu_parboil, "parboil")]:
        part = load_ncu(path, label)
        if part is not None and not part.empty:
            ncu_parts.append(part)

    if ncu_parts:
        ncu = pd.concat(ncu_parts, ignore_index=True)
        ok = ncu[ncu["ncu_returncode"].fillna(0).astype(int) == 0]
        for name, subset in [("ncu_all_runs", ncu["ncu_wall_s"]), ("ncu_success", ok["ncu_wall_s"])]:
            row = summarize_series(name, subset)
            summary_rows.append(row)
            print_row(row)

        med = summary_rows[-1]["median_s"]
        print()
        print("   Thesis line (typical NCU profile, use median):")
        print(f"   \"Each NCU kernel profile took a median of {med:.1f}\\,s wall-clock time "
              f"({len(ok)} kernels).\"")
    else:
        print("   No NCU timing data found.")

    # --- Oracle energy measurements ---
    print("\n2) Oracle energy measurements (clock sweeps)")
    print("   Per-run logs: results/*_kernel_energy/**/summary.log")
    print("   NOTE: main pipeline logs configured CUDA windows, NOT shell wall-clock time.")

    oracle = scan_oracle_measurements(RESULTS)
    if oracle.empty:
        print("   No oracle measurement logs found.")
    else:
        cfg = oracle["configured_cuda_window_s"].dropna()
        row = summarize_series("oracle_configured_cuda_window_per_run", cfg)
        summary_rows.append(row)
        print_row(row)
        print(f"    number of measurement runs: {len(oracle)}")
        total_cuda = float(cfg.sum())
        print(f"    sum of configured CUDA windows: {fmt_seconds(total_cuda)} "
              f"(25s warmup + 5s measure per run)")
        if "elapsed_s" in oracle.columns and oracle["elapsed_s"].notna().any():
            wall = oracle["elapsed_s"].dropna()
            row2 = summarize_series("oracle_wall_clock_recorded_subset", wall)
            summary_rows.append(row2)
            print_row(row2)
        else:
            print("    wall-clock elapsed_s: NOT recorded in current oracle logs")

    # --- Training ---
    print("\n3) LOKO model training")
    print("   Script: scripts/prediction/3_train/1_train_ptx_launch_runtime_loko.py")
    print("   STATUS: training wall time is NOT recorded to CSV/log.")
    print("   The train log only stores prediction metrics (MAPE, etc.).")
    print("   To report training cost, re-run with timing added or use log timestamps manually.")

    train_logs = sorted(RESULTS.glob("train_*.log"))
    if train_logs:
        latest = train_logs[-1]
        print(f"   Latest train log: {latest.name}  (mtime only, not duration)")

    print("\nMean vs median (for your write-up)")
    print("-" * 60)
    print("  Use MEDIAN for \"typical per-run profiling cost\" (robust to slow outliers).")
    print("  Use MEAN (or total sum) for \"expected total pipeline overhead / budget\".")
    print("  Reporting both + n is standard in systems papers.")

    if summary_rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
