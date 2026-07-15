#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

IN_CSV = Path.home() / "thesis/results/master/aggregated_all_kernels.csv"
OUT_CSV = Path.home() / "thesis/results/master/aggregated_all_kernels_effective.csv"


GROUP_FIXED = [
    "workload",
    "kernel",
    "applied_core_clock",
    "applied_mem_clock",
]


MEAN_COLS = [
    "measured_launches_mean",
    "total_time_mean_ms",
    "total_energy_mean_mJ",
    "avg_power_mean_W",
    "time_per_launch_mean_ms",
    "energy_per_launch_mean_mJ",
]

STD_COLS = [
    "measured_launches_std",
    "total_time_std_ms",
    "total_energy_std_mJ",
    "avg_power_std_W",
    "time_per_launch_std_ms",
    "energy_per_launch_std_mJ",
]


def main():
    df = pd.read_csv(IN_CSV)

    # Normalize auto/fixed masks.
    core_s = df["requested_core_clock"].astype(str).str.lower()
    mem_s = df["requested_mem_clock"].astype(str).str.lower()

    auto = df[(core_s == "auto") & (mem_s == "auto")].copy()
    fixed = df[(core_s != "auto") & (mem_s != "auto")].copy()

    # Numeric applied clocks for fixed rows.
    fixed["applied_core_clock"] = pd.to_numeric(fixed["applied_core_clock"], errors="coerce")
    fixed["applied_mem_clock"] = pd.to_numeric(fixed["applied_mem_clock"], errors="coerce")

    fixed["requested_pair"] = (
        fixed["requested_core_clock"].astype(str)
        + "/"
        + fixed["requested_mem_clock"].astype(str)
    )

    # Aggregate fixed rows by effective/applied clock pair.
    agg_spec = {}

    for c in fixed.columns:
        if c in GROUP_FIXED:
            continue

        if c in MEAN_COLS:
            agg_spec[c] = "mean"
        elif c in STD_COLS:
            # Existing std columns are not meaningful with n=1, recompute later.
            agg_spec[c] = "mean"
        elif c == "n":
            agg_spec[c] = "sum"
        elif c == "requested_pair":
            agg_spec[c] = lambda x: ";".join(sorted(set(map(str, x))))
        elif c == "requested_core_clock":
            agg_spec[c] = lambda x: ";".join(sorted(set(map(str, x))))
        elif c == "requested_mem_clock":
            agg_spec[c] = lambda x: ";".join(sorted(set(map(str, x))))
        elif pd.api.types.is_numeric_dtype(fixed[c]):
            agg_spec[c] = "first"
        else:
            agg_spec[c] = "first"

    collapsed = fixed.groupby(GROUP_FIXED, as_index=False).agg(agg_spec)

    # Count how many requested grid points mapped to each effective pair.
    counts = (
        fixed.groupby(GROUP_FIXED, as_index=False)
        .agg(
            num_requested_pairs_merged=("requested_pair", "nunique"),
            requested_pairs_merged=("requested_pair", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )

    collapsed = collapsed.merge(counts, on=GROUP_FIXED, how="left")

    # Use effective clocks as requested clocks for downstream scripts if needed.
    # Keep original merged requested info separately.
    collapsed["effective_core_clock"] = collapsed["applied_core_clock"]
    collapsed["effective_mem_clock"] = collapsed["applied_mem_clock"]

    # Important: downstream scripts identify fixed rows by requested != auto.
    # Set requested to effective numeric clock for the cleaned dataset.
    collapsed["requested_core_clock_original"] = collapsed["requested_core_clock"]
    collapsed["requested_mem_clock_original"] = collapsed["requested_mem_clock"]
    collapsed["requested_core_clock"] = collapsed["effective_core_clock"]
    collapsed["requested_mem_clock"] = collapsed["effective_mem_clock"]

    # Auto rows: keep as is; but add effective columns.
    auto = auto.copy()
    auto["effective_core_clock"] = np.nan
    auto["effective_mem_clock"] = np.nan
    auto["num_requested_pairs_merged"] = 1
    auto["requested_pairs_merged"] = "auto/auto"
    auto["requested_core_clock_original"] = auto["requested_core_clock"]
    auto["requested_mem_clock_original"] = auto["requested_mem_clock"]

    # Align columns.
    all_cols = []
    for d in [auto, collapsed]:
        for c in d.columns:
            if c not in all_cols:
                all_cols.append(c)

    out = pd.concat(
        [auto.reindex(columns=all_cols), collapsed.reindex(columns=all_cols)],
        ignore_index=True,
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Input rows:  {len(df)}")
    print(f"Auto rows:   {len(auto)}")
    print(f"Fixed rows before collapse: {len(fixed)}")
    print(f"Fixed rows after collapse:  {len(collapsed)}")
    print(f"Output rows: {len(out)}")
    print(f"Wrote {OUT_CSV}")

    print()
    print("Effective fixed pairs per kernel:")
    per_kernel = collapsed.groupby(["workload", "kernel"]).size().reset_index(name="effective_pairs")
    print(per_kernel["effective_pairs"].value_counts().sort_index().to_string())

    print()
    print("Merged requested-pair counts:")
    print(collapsed["num_requested_pairs_merged"].value_counts().sort_index().to_string())

    print()
    print("Examples where requested pairs were merged:")
    ex = collapsed[collapsed["num_requested_pairs_merged"] > 1][[
        "workload",
        "kernel",
        "effective_core_clock",
        "effective_mem_clock",
        "num_requested_pairs_merged",
        "requested_pairs_merged",
    ]]
    print(ex.head(80).to_string(index=False))


if __name__ == "__main__":
    main()