#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except ImportError as e:
    raise SystemExit(
        "scipy.optimize.milp is required. Install/update scipy in the venv:\n"
        "  pip install -U scipy"
    ) from e


IN_CSV = Path.home() / "thesis/results/master/ml_dataset_ptx_launch_runtime_normalized.csv"
OUT_DIR = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"

OUT_SELECTIONS = OUT_DIR / "measured_global_oracle_selections.csv"
OUT_SUMMARY = OUT_DIR / "measured_global_oracle_summary.csv"

CONSTRAINTS = {
    "0pct": 1.00,
    "1pct": 1.01,
    "2pct": 1.02,
    "5pct": 1.05,
}


def prepare_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["kernel_id"] = df["join_workload"].astype(str) + "::" + df["join_kernel"].astype(str)

    needed = [
        "workload",
        "kernel",
        "kernel_id",
        "requested_core_clock",
        "requested_mem_clock",
        "effective_core_clock",
        "effective_mem_clock",
        "time_per_launch_mean_ms",
        "energy_per_launch_mean_mJ",
        "auto_time_per_launch_mean_ms",
        "auto_energy_per_launch_mean_mJ",
        "time_ratio_to_auto",
        "energy_ratio_to_auto",
    ]

    c = df[needed].copy()

    c = c.rename(columns={
        "time_ratio_to_auto": "actual_time_ratio",
        "energy_ratio_to_auto": "actual_energy_ratio",
    })

    c["requested_core_clock"] = c["requested_core_clock"].astype(str)
    c["requested_mem_clock"] = c["requested_mem_clock"].astype(str)
    c["effective_core_clock"] = c["effective_core_clock"].astype(str)
    c["effective_mem_clock"] = c["effective_mem_clock"].astype(str)

    c["is_virtual_auto"] = False

    return c


def add_virtual_auto_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    autos = []

    for _, g in candidates.groupby(["workload", "kernel", "kernel_id"], sort=False):
        first = g.iloc[0]

        autos.append({
            "workload": first["workload"],
            "kernel": first["kernel"],
            "kernel_id": first["kernel_id"],

            "requested_core_clock": "auto",
            "requested_mem_clock": "auto",
            "effective_core_clock": "auto",
            "effective_mem_clock": "auto",

            "time_per_launch_mean_ms": first["auto_time_per_launch_mean_ms"],
            "energy_per_launch_mean_mJ": first["auto_energy_per_launch_mean_mJ"],
            "auto_time_per_launch_mean_ms": first["auto_time_per_launch_mean_ms"],
            "auto_energy_per_launch_mean_mJ": first["auto_energy_per_launch_mean_mJ"],

            "actual_time_ratio": 1.0,
            "actual_energy_ratio": 1.0,
            "is_virtual_auto": True,
        })

    return pd.concat([candidates, pd.DataFrame(autos)], ignore_index=True)


def solve_global_waste_oracle(candidates: pd.DataFrame, limit: float) -> pd.DataFrame:
    """
    Jeffrey/Spaan-style global compute-waste oracle.

    Choose exactly one configuration per kernel.
    Minimize total measured energy.
    Constraint: total measured runtime <= total auto runtime * limit.

    Individual kernels may exceed the local slowdown bound. That is allowed here
    if the full sequence remains within the global runtime budget.
    """
    c = candidates.copy().reset_index(drop=True)

    kernels = sorted(c["kernel_id"].unique())
    n_vars = len(c)

    # Objective: minimize measured energy.
    objective = c["energy_per_launch_mean_mJ"].to_numpy(dtype=float)

    constraints = []

    # One selected candidate per kernel.
    for kid in kernels:
        row = np.zeros(n_vars)
        row[c.index[c["kernel_id"] == kid].to_numpy()] = 1.0
        constraints.append(LinearConstraint(row, lb=1.0, ub=1.0))

    # Global sequence-time constraint.
    auto_total_time = (
        c.drop_duplicates("kernel_id")["auto_time_per_launch_mean_ms"]
        .sum()
    )
    allowed_total_time = auto_total_time * limit

    time_row = c["time_per_launch_mean_ms"].to_numpy(dtype=float)
    constraints.append(LinearConstraint(time_row, lb=-np.inf, ub=allowed_total_time))

    result = milp(
        c=objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars)),
        constraints=constraints,
        options={
            "time_limit": 300,
            "mip_rel_gap": 0.0,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    x = np.asarray(result.x)
    selected_idx = np.where(x > 0.5)[0]

    selected = c.iloc[selected_idx].copy()

    if selected["kernel_id"].nunique() != len(kernels):
        raise RuntimeError("Solver did not select exactly one candidate per kernel.")

    selected["global_limit"] = limit
    selected["allowed_total_time_ms"] = allowed_total_time
    selected["auto_total_time_ms"] = auto_total_time
    selected["solver_objective_energy_mJ"] = float(result.fun)

    return selected


def summarize_selection(selected: pd.DataFrame, constraint_name: str, limit: float) -> dict:
    auto_energy_sum = selected["auto_energy_per_launch_mean_mJ"].sum()
    selected_energy_sum = selected["energy_per_launch_mean_mJ"].sum()

    auto_time_sum = selected["auto_time_per_launch_mean_ms"].sum()
    selected_time_sum = selected["time_per_launch_mean_ms"].sum()

    global_energy_saving_pct = (1.0 - selected_energy_sum / auto_energy_sum) * 100.0
    global_slowdown_pct = (selected_time_sum / auto_time_sum - 1.0) * 100.0
    global_time_ratio = selected_time_sum / auto_time_sum
    global_energy_ratio = selected_energy_sum / auto_energy_sum

    return {
        "constraint_name": constraint_name,
        "constraint_limit": limit,
        "n_kernels": int(selected["kernel_id"].nunique()),

        "global_energy_saving_pct": global_energy_saving_pct,
        "global_slowdown_pct": global_slowdown_pct,
        "global_time_ratio": global_time_ratio,
        "global_energy_ratio": global_energy_ratio,

        "auto_total_time_ms": auto_time_sum,
        "selected_total_time_ms": selected_time_sum,
        "allowed_total_time_ms": auto_time_sum * limit,

        "auto_total_energy_mJ": auto_energy_sum,
        "selected_total_energy_mJ": selected_energy_sum,

        "auto_selection_rate_pct": selected["is_virtual_auto"].mean() * 100.0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    candidates = prepare_candidates(df)
    candidates = add_virtual_auto_rows(candidates)

    print(f"Candidates including auto: {len(candidates)}")
    print(f"Kernels: {candidates['kernel_id'].nunique()}")
    print(f"Effective pairs: {candidates[['effective_core_clock', 'effective_mem_clock']].drop_duplicates().shape[0]}")

    all_selected = []
    summary_rows = []

    for constraint_name, limit in CONSTRAINTS.items():
        selected = solve_global_waste_oracle(candidates, limit)
        selected["constraint_name"] = constraint_name
        selected["constraint_limit"] = limit

        selected["selected_actual_energy_saving_pct"] = (1.0 - selected["actual_energy_ratio"]) * 100.0

        all_selected.append(selected)
        summary_rows.append(summarize_selection(selected, constraint_name, limit))

    selections = pd.concat(all_selected, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    selections.to_csv(OUT_SELECTIONS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print()
    print(f"Wrote {OUT_SELECTIONS}")
    print(f"Wrote {OUT_SUMMARY}")

    print()
    print("Global compute-waste oracle:")
    print(summary[[
        "constraint_name",
        "global_energy_saving_pct",
        "global_slowdown_pct",
        "global_time_ratio",
        "auto_selection_rate_pct",
    ]].to_string(index=False))

    print()
    print("Selected clock counts:")
    counts = (
        selections
        .groupby(["constraint_name", "effective_core_clock", "effective_mem_clock"])
        .size()
        .reset_index(name="count")
        .sort_values(["constraint_name", "count"], ascending=[True, False])
    )
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
