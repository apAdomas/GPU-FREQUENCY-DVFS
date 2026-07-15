#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

PRED_CSV = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models/normalized_loko_predictions.csv"
OUT_DIR = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"

OUT_SELECTIONS = OUT_DIR / "predicted_local_policy_selections.csv"
OUT_SUMMARY = OUT_DIR / "predicted_local_policy_summary.csv"

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

    return merged


def add_virtual_auto_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Auto is always a valid fallback:
      predicted ratio = 1
      actual ratio = 1
      energy saving = 0

    The prediction file only has fixed clock candidates, so we add auto manually.
    """
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

    candidates = candidates.copy()
    candidates["is_virtual_auto"] = False

    return pd.concat([candidates, pd.DataFrame(autos)], ignore_index=True)


def select_predicted_policy(g: pd.DataFrame, limit: float):
    g = g.copy()

    feasible = g[g["pred_time_ratio"] <= limit].copy()

    no_pred_feasible = False
    if feasible.empty:
        feasible = g[
            (g["effective_core_clock"].astype(str) == "auto")
            & (g["effective_mem_clock"].astype(str) == "auto")
        ].copy()
        no_pred_feasible = True

    feasible = feasible.sort_values(
        ["pred_energy_ratio", "pred_time_ratio", "effective_core_clock", "effective_mem_clock"],
        ascending=[True, True, True, True],
    )

    return feasible.iloc[0], no_pred_feasible


def select_actual_oracle(g: pd.DataFrame, limit: float) -> pd.Series:
    """
    True oracle: lowest actual energy among actually-feasible candidates.
    Auto is included, so this always has at least one feasible candidate.
    """
    feasible = g[g["actual_time_ratio"] <= limit + 1e-9].copy()

    if feasible.empty:
        feasible = g[
            (g["effective_core_clock"].astype(str) == "auto")
            & (g["effective_mem_clock"].astype(str) == "auto")
        ].copy()

    feasible = feasible.sort_values(
        ["actual_energy_ratio", "actual_time_ratio", "effective_core_clock", "effective_mem_clock"],
        ascending=[True, True, True, True],
    )

    return feasible.iloc[0]


def evaluate(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["feature_set", "model", "kernel_id"]

    for constraint_name, limit in CONSTRAINTS.items():
        for (feature_set, model, kernel_id), g in candidates.groupby(group_cols, sort=False):
            chosen, no_pred_feasible = select_predicted_policy(g, limit)
            oracle = select_actual_oracle(g, limit)

            auto_time = float(chosen["auto_time_per_launch_mean_ms"])
            auto_energy = float(chosen["auto_energy_per_launch_mean_mJ"])

            selected_time = float(chosen["actual_time_ratio"]) * auto_time
            selected_energy = float(chosen["actual_energy_ratio"]) * auto_energy

            oracle_time = float(oracle["actual_time_ratio"]) * auto_time
            oracle_energy = float(oracle["actual_energy_ratio"]) * auto_energy

            selected_saving_pct = (1.0 - float(chosen["actual_energy_ratio"])) * 100.0
            oracle_saving_pct = (1.0 - float(oracle["actual_energy_ratio"])) * 100.0

            rows.append({
                "constraint_name": constraint_name,
                "constraint_limit": limit,

                "feature_set": feature_set,
                "model": model,

                "workload": chosen["workload"],
                "kernel": chosen["kernel"],
                "kernel_id": kernel_id,

                "selected_requested_core_clock": chosen["requested_core_clock"],
                "selected_requested_mem_clock": chosen["requested_mem_clock"],
                "selected_effective_core_clock": chosen["effective_core_clock"],
                "selected_effective_mem_clock": chosen["effective_mem_clock"],
                "selected_is_auto": bool(str(chosen["effective_core_clock"]) == "auto"),
                "no_pred_feasible": bool(no_pred_feasible),

                "selected_pred_time_ratio": float(chosen["pred_time_ratio"]),
                "selected_pred_energy_ratio": float(chosen["pred_energy_ratio"]),
                "selected_actual_time_ratio": float(chosen["actual_time_ratio"]),
                "selected_actual_energy_ratio": float(chosen["actual_energy_ratio"]),
                "selected_actual_slowdown_pct": (float(chosen["actual_time_ratio"]) - 1.0) * 100.0,
                "selected_actual_energy_saving_pct": selected_saving_pct,
                "selected_violates_constraint": bool(float(chosen["actual_time_ratio"]) > limit + 1e-9),

                "oracle_requested_core_clock": oracle["requested_core_clock"],
                "oracle_requested_mem_clock": oracle["requested_mem_clock"],
                "oracle_effective_core_clock": oracle["effective_core_clock"],
                "oracle_effective_mem_clock": oracle["effective_mem_clock"],
                "oracle_is_auto": bool(str(oracle["effective_core_clock"]) == "auto"),
                "oracle_actual_time_ratio": float(oracle["actual_time_ratio"]),
                "oracle_actual_energy_ratio": float(oracle["actual_energy_ratio"]),
                "oracle_actual_slowdown_pct": (float(oracle["actual_time_ratio"]) - 1.0) * 100.0,
                "oracle_actual_energy_saving_pct": oracle_saving_pct,

                "oracle_gap_energy_saving_pp": oracle_saving_pct - selected_saving_pct,
                "oracle_gap_energy_ratio": float(chosen["actual_energy_ratio"]) - float(oracle["actual_energy_ratio"]),

                "auto_time_per_launch_mean_ms": auto_time,
                "auto_energy_per_launch_mean_mJ": auto_energy,
                "selected_time_per_launch_ms": selected_time,
                "selected_energy_per_launch_mJ": selected_energy,
                "oracle_time_per_launch_ms": oracle_time,
                "oracle_energy_per_launch_mJ": oracle_energy,
            })

    return pd.DataFrame(rows)


def summarize(selections: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "constraint_name",
        "constraint_limit",
        "feature_set",
        "model",
    ]

    for keys, g in selections.groupby(group_cols, sort=False):
        (
            constraint_name,
            constraint_limit,
            feature_set,
            model,
        ) = keys

        auto_energy_sum = g["auto_energy_per_launch_mean_mJ"].sum()
        selected_energy_sum = g["selected_energy_per_launch_mJ"].sum()
        oracle_energy_sum = g["oracle_energy_per_launch_mJ"].sum()

        auto_time_sum = g["auto_time_per_launch_mean_ms"].sum()
        selected_time_sum = g["selected_time_per_launch_ms"].sum()
        oracle_time_sum = g["oracle_time_per_launch_ms"].sum()

        selected_weighted_saving_pct = (1.0 - selected_energy_sum / auto_energy_sum) * 100.0
        oracle_weighted_saving_pct = (1.0 - oracle_energy_sum / auto_energy_sum) * 100.0
        weighted_oracle_gap_pp = oracle_weighted_saving_pct - selected_weighted_saving_pct

        selected_weighted_slowdown_pct = (selected_time_sum / auto_time_sum - 1.0) * 100.0
        oracle_weighted_slowdown_pct = (oracle_time_sum / auto_time_sum - 1.0) * 100.0

        violation_rate_pct = g["selected_violates_constraint"].mean() * 100.0
        auto_selection_rate_pct = g["selected_is_auto"].mean() * 100.0
        no_pred_feasible_rate_pct = g["no_pred_feasible"].mean() * 100.0

        viol = g[g["selected_violates_constraint"]].copy()

        if len(viol):
            max_violation_over_bound_pct = (
                (viol["selected_actual_time_ratio"] - constraint_limit).max() * 100.0
            )
            mean_violation_over_bound_pct = (
                (viol["selected_actual_time_ratio"] - constraint_limit).mean() * 100.0
            )
            max_violating_slowdown_pct = viol["selected_actual_slowdown_pct"].max()
        else:
            max_violation_over_bound_pct = 0.0
            mean_violation_over_bound_pct = 0.0
            max_violating_slowdown_pct = 0.0

        rows.append({
            "constraint_name": constraint_name,
            "constraint_limit": constraint_limit,
            "feature_set": feature_set,
            "model": model,

            "n_kernels": int(len(g)),

            "mean_selected_energy_saving_pct": g["selected_actual_energy_saving_pct"].mean(),
            "median_selected_energy_saving_pct": g["selected_actual_energy_saving_pct"].median(),
            "weighted_selected_energy_saving_pct": selected_weighted_saving_pct,

            "mean_oracle_energy_saving_pct": g["oracle_actual_energy_saving_pct"].mean(),
            "median_oracle_energy_saving_pct": g["oracle_actual_energy_saving_pct"].median(),
            "weighted_oracle_energy_saving_pct": oracle_weighted_saving_pct,
            "weighted_oracle_gap_pp": weighted_oracle_gap_pp,

            "mean_oracle_gap_pp": g["oracle_gap_energy_saving_pp"].mean(),
            "median_oracle_gap_pp": g["oracle_gap_energy_saving_pp"].median(),

            "mean_selected_slowdown_pct": g["selected_actual_slowdown_pct"].mean(),
            "median_selected_slowdown_pct": g["selected_actual_slowdown_pct"].median(),
            "p90_selected_slowdown_pct": g["selected_actual_slowdown_pct"].quantile(0.90),

            "weighted_selected_slowdown_pct": selected_weighted_slowdown_pct,
            "weighted_oracle_slowdown_pct": oracle_weighted_slowdown_pct,
            "weighted_selected_time_ratio": selected_time_sum / auto_time_sum,
            "weighted_oracle_time_ratio": oracle_time_sum / auto_time_sum,

            "violation_count": int(g["selected_violates_constraint"].sum()),
            "violation_rate_pct": violation_rate_pct,
            "max_violation_over_bound_pct": max_violation_over_bound_pct,
            "mean_violation_over_bound_pct": mean_violation_over_bound_pct,
            "max_violating_slowdown_pct": max_violating_slowdown_pct,
            "auto_selection_rate_pct": auto_selection_rate_pct,
            "no_pred_feasible_rate_pct": no_pred_feasible_rate_pct,
        })

    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(PRED_CSV)

    print(f"Loaded predictions: {preds.shape}")
    print("Targets:")
    print(preds["target_name"].value_counts().to_string())

    candidates = build_candidate_table(preds)
    candidates = add_virtual_auto_rows(candidates)

    print(f"Candidate rows including virtual auto: {len(candidates)}")
    print(f"Kernels: {candidates['kernel_id'].nunique()}")
    print(f"Feature/model combinations: {candidates[['feature_set', 'model']].drop_duplicates().shape[0]}")

    selections = evaluate(candidates)
    summary = summarize(selections)

    selections.to_csv(OUT_SELECTIONS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print()
    print(f"Wrote {OUT_SELECTIONS}")
    print(f"Wrote {OUT_SUMMARY}")

    print()
    print("Best predicted local policies by energy saving:")
    best = (
        summary
        .sort_values([
            "constraint_name",
            "weighted_selected_energy_saving_pct",
        ], ascending=[True, False])
        .groupby("constraint_name")
        .head(10)
    )

    print(best[[
        "constraint_name",
        "feature_set",
        "model",
        "weighted_selected_energy_saving_pct",
        "weighted_oracle_energy_saving_pct",
        "mean_selected_slowdown_pct",
        "median_selected_slowdown_pct",
        "p90_selected_slowdown_pct",
        "violation_count",
        "violation_rate_pct",
        "max_violation_over_bound_pct",
        "auto_selection_rate_pct",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()