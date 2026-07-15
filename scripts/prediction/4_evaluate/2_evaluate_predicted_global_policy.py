#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except ImportError as e:
    raise SystemExit(
        "scipy.optimize.milp is required. Install/update scipy:\n"
        "  pip install -U scipy"
    ) from e


PRED_CSV = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models/normalized_loko_predictions.csv"
OUT_DIR = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"

OUT_SELECTIONS = OUT_DIR / "predicted_global_policy_selections.csv"
OUT_SUMMARY = OUT_DIR / "predicted_global_policy_summary.csv"

SUMMARY_DISPLAY_COLS = [
    "constraint_name",
    "feature_set",
    "model",
    "actual_global_energy_saving_pct",
    "measured_oracle_global_energy_saving_pct",
    "global_oracle_gap_pp",
    "actual_global_slowdown_pct",
    "pred_global_slowdown_pct",
    "global_violates_constraint",
    "global_violation_over_bound_pct",
    "auto_selection_rate_pct",
]

CONSTRAINTS = {
    "0pct": 1.00,
    "1pct": 1.01,
    "2pct": 1.02,
    "5pct": 1.05,
}


def build_candidate_table(preds: pd.DataFrame) -> pd.DataFrame:
    key = [
        "workload",
        "kernel",
        "kernel_id",
        "effective_core_clock",
        "effective_mem_clock",
        "feature_set",
        "model",
    ]

    keep_actual = [
        "requested_core_clock",
        "requested_mem_clock",
        "time_per_launch_mean_ms",
        "energy_per_launch_mean_mJ",
        "auto_time_per_launch_mean_ms",
        "auto_energy_per_launch_mean_mJ",
        "time_ratio_to_auto",
        "energy_ratio_to_auto",
    ]

    time = preds[preds["target_name"] == "time_ratio"][key + keep_actual + ["y_pred"]].copy()
    energy = preds[preds["target_name"] == "energy_ratio"][key + ["y_pred"]].copy()

    time = time.rename(columns={"y_pred": "pred_time_ratio"})
    energy = energy.rename(columns={"y_pred": "pred_energy_ratio"})

    merged = time.merge(
        energy,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    merged = merged.rename(columns={
        "time_ratio_to_auto": "actual_time_ratio",
        "energy_ratio_to_auto": "actual_energy_ratio",
    })

    merged["requested_core_clock"] = merged["requested_core_clock"].astype(str)
    merged["requested_mem_clock"] = merged["requested_mem_clock"].astype(str)
    merged["effective_core_clock"] = merged["effective_core_clock"].astype(str)
    merged["effective_mem_clock"] = merged["effective_mem_clock"].astype(str)

    merged["is_virtual_auto"] = False

    return merged


def add_virtual_auto_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["workload", "kernel", "kernel_id", "feature_set", "model"]

    autos = []
    for _, g in candidates.groupby(group_cols, sort=False):
        first = g.iloc[0]

        autos.append({
            "workload": first["workload"],
            "kernel": first["kernel"],
            "kernel_id": first["kernel_id"],
            "feature_set": first["feature_set"],
            "model": first["model"],

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

            "pred_time_ratio": 1.0,
            "pred_energy_ratio": 1.0,

            "is_virtual_auto": True,
        })

    return pd.concat([candidates, pd.DataFrame(autos)], ignore_index=True)


def solve_predicted_global_policy(candidates: pd.DataFrame, limit: float) -> pd.DataFrame:
    """
    Solve global compute-waste selection using predicted time and energy.

    Objective:
        minimize predicted total energy

    Constraint:
        predicted total time <= auto_total_time * limit

    Then the selected policy is later evaluated using actual measured time/energy.
    """
    c = candidates.copy().reset_index(drop=True)

    kernels = sorted(c["kernel_id"].unique())
    n_vars = len(c)

    auto_total_time = (
        c.drop_duplicates("kernel_id")["auto_time_per_launch_mean_ms"]
        .sum()
    )
    allowed_total_time = auto_total_time * limit

    # Convert predicted ratios back to predicted absolute per-launch time/energy.
    c["pred_time_per_launch_ms"] = (
        c["pred_time_ratio"] * c["auto_time_per_launch_mean_ms"]
    )
    c["pred_energy_per_launch_mJ"] = (
        c["pred_energy_ratio"] * c["auto_energy_per_launch_mean_mJ"]
    )

    objective = c["pred_energy_per_launch_mJ"].to_numpy(dtype=float)

    constraints = []

    # Exactly one candidate per kernel.
    for kid in kernels:
        row = np.zeros(n_vars)
        row[c.index[c["kernel_id"] == kid].to_numpy()] = 1.0
        constraints.append(LinearConstraint(row, lb=1.0, ub=1.0))

    # Predicted global time constraint.
    pred_time_row = c["pred_time_per_launch_ms"].to_numpy(dtype=float)

    constraints.append(LinearConstraint(pred_time_row, lb=-np.inf, ub=allowed_total_time))

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
    selected["solver_pred_objective_energy_mJ"] = float(result.fun)

    return selected


def summarize_selection(selected: pd.DataFrame, constraint_name: str, limit: float) -> dict:
    auto_energy_sum = selected["auto_energy_per_launch_mean_mJ"].sum()
    actual_energy_sum = selected["energy_per_launch_mean_mJ"].sum()
    pred_energy_sum = selected["pred_energy_per_launch_mJ"].sum()

    auto_time_sum = selected["auto_time_per_launch_mean_ms"].sum()
    actual_time_sum = selected["time_per_launch_mean_ms"].sum()
    pred_time_sum = selected["pred_time_per_launch_ms"].sum()

    actual_global_slowdown_pct = (actual_time_sum / auto_time_sum - 1.0) * 100.0
    pred_global_slowdown_pct = (pred_time_sum / auto_time_sum - 1.0) * 100.0

    return {
        "constraint_name": constraint_name,
        "constraint_limit": limit,
        "feature_set": selected["feature_set"].iloc[0],
        "model": selected["model"].iloc[0],
        "n_kernels": int(selected["kernel_id"].nunique()),

        "actual_global_energy_saving_pct": (1.0 - actual_energy_sum / auto_energy_sum) * 100.0,
        "actual_global_slowdown_pct": actual_global_slowdown_pct,
        "actual_global_time_ratio": actual_time_sum / auto_time_sum,
        "actual_global_energy_ratio": actual_energy_sum / auto_energy_sum,

        "pred_global_energy_saving_pct": (1.0 - pred_energy_sum / auto_energy_sum) * 100.0,
        "pred_global_slowdown_pct": pred_global_slowdown_pct,

        "global_violates_constraint": bool(actual_time_sum > auto_time_sum * limit + 1e-9),
        "global_violation_over_bound_pct": max(
            0.0,
            (actual_time_sum / auto_time_sum - limit) * 100.0,
        ),

        "auto_total_time_ms": auto_time_sum,
        "actual_selected_total_time_ms": actual_time_sum,
        "pred_selected_total_time_ms": pred_time_sum,
        "allowed_total_time_ms": auto_time_sum * limit,

        "auto_total_energy_mJ": auto_energy_sum,
        "actual_selected_total_energy_mJ": actual_energy_sum,
        "pred_selected_total_energy_mJ": pred_energy_sum,

        "auto_selection_rate_pct": selected["is_virtual_auto"].mean() * 100.0,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(PRED_CSV)

    print(f"Loaded predictions: {preds.shape}")
    print("Targets:")
    print(preds["target_name"].value_counts().to_string())

    candidates = build_candidate_table(preds)
    candidates = add_virtual_auto_rows(candidates)

    print(f"Candidate rows including auto: {len(candidates)}")
    print(f"Kernels: {candidates['kernel_id'].nunique()}")
    print(f"Feature/model combinations: {candidates[['feature_set', 'model']].drop_duplicates().shape[0]}")

    all_selected = []
    summary_rows = []

    group_cols = ["feature_set", "model"]

    for constraint_name, limit in CONSTRAINTS.items():
        for (feature_set, model), g in candidates.groupby(group_cols, sort=False):
            selected = solve_predicted_global_policy(g, limit)

            selected["constraint_name"] = constraint_name
            selected["constraint_limit"] = limit

            selected["selected_actual_energy_saving_pct"] = (
                1.0 - selected["actual_energy_ratio"]
            ) * 100.0

            all_selected.append(selected)
            summary_rows.append(
                summarize_selection(
                    selected,
                    constraint_name,
                    limit,
                )
            )

    selections = pd.concat(all_selected, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    oracle_path = OUT_DIR / "measured_global_oracle_summary.csv"

    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)

        oracle_keep = oracle[[
            "constraint_name",
            "global_energy_saving_pct",
            "global_slowdown_pct",
            "global_time_ratio",
            "global_energy_ratio",
        ]].rename(columns={
            "global_energy_saving_pct": "measured_oracle_global_energy_saving_pct",
            "global_slowdown_pct": "measured_oracle_global_slowdown_pct",
            "global_time_ratio": "measured_oracle_global_time_ratio",
            "global_energy_ratio": "measured_oracle_global_energy_ratio",
        })

        summary = summary.merge(
            oracle_keep,
            on="constraint_name",
            how="left",
        )

        summary["global_oracle_gap_pp"] = (
            summary["measured_oracle_global_energy_saving_pct"]
            - summary["actual_global_energy_saving_pct"]
        )

    selections.to_csv(OUT_SELECTIONS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print()
    print(f"Wrote {OUT_SELECTIONS}")
    print(f"Wrote {OUT_SUMMARY}")

    print()
    print("Best predicted global policies with no global violation:")
    display_cols = [c for c in SUMMARY_DISPLAY_COLS if c in summary.columns]
    best = (
        summary[summary["global_violates_constraint"] == False]
        .sort_values(
            [
                "constraint_name",
                "actual_global_energy_saving_pct",
                "global_violation_over_bound_pct",
            ],
            ascending=[True, False, True],
        )
        .groupby("constraint_name")
        .head(10)
    )

    print(best[display_cols].to_string(index=False))

    print()
    print("Best predicted global policies allowing <= 0.5 pp global excess:")
    near = (
        summary[summary["global_violation_over_bound_pct"] <= 0.5]
        .sort_values(
            [
                "constraint_name",
                "actual_global_energy_saving_pct",
                "global_violation_over_bound_pct",
            ],
            ascending=[True, False, True],
        )
        .groupby("constraint_name")
        .head(10)
    )

    print(near[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
