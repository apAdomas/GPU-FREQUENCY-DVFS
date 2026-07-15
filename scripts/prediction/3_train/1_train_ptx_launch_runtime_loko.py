#!/usr/bin/env python3
from pathlib import Path
import math
import time
import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
from xgboost import XGBRegressor


IN_CSV = Path.home() / "thesis/results/master/ml_dataset_ptx_launch_runtime_normalized.csv"
OUT_DIR = Path.home() / "thesis/results/master/ptx_launch_runtime_normalized_models"

TARGETS = {
    "time_ratio": "time_ratio_to_auto",
    "energy_ratio": "energy_ratio_to_auto",
}

CLOCK_COLS = [
    "effective_core_clock",
    "effective_mem_clock",
    "core_clock_norm",
    "mem_clock_norm",
    "inv_core_clock",
    "inv_mem_clock",
    "core_mem_clock_product_norm",
]

GROUPED_PTX_COLS = [
    "opcode_total",
    "instr_form_total",
    "dtype_total",
    "memspace_total",

    "arith_total",
    "int_op_total",
    "control_total",
    "predicate_total",
    "load_total",
    "store_total",
    "memory_total",

    "global_mem_total",
    "shared_mem_total",
    "local_mem_total",
    "param_mem_total",

    "estimated_global_mem_bytes",
    "estimated_shared_mem_bytes",
    "estimated_local_mem_bytes",
    "estimated_param_mem_bytes",
    "estimated_const_mem_bytes",
    "estimated_total_mem_bytes",

    "float_dtype_total",
    "int_dtype_total",
    "compute_instr_total",

    "registers_per_thread_approx",
    "reg_pred_count",
    "reg_f16_count",
    "reg_f32_count",
    "reg_f64_count",
    "reg_b16_count",
    "reg_b32_count",
    "reg_b64_count",
    "reg_s32_count",
    "reg_s64_count",
    "reg_u32_count",
    "reg_u64_count",

    "fma_total",
    "mul_total",
    "add_total",
    "div_total",
    "bra_total",
    "setp_total",

    "arith_ratio",
    "memory_ratio",
    "load_ratio",
    "store_ratio",
    "control_ratio",
    "predicate_ratio",
    "global_mem_ratio",
    "shared_mem_ratio",
    "local_mem_ratio",
    "param_mem_ratio",
    "float_dtype_ratio",
    "int_dtype_ratio",
    "mem_to_arith_ratio",

    "arithmetic_intensity_global",
    "arithmetic_intensity_total",
]

LAUNCH_COLS = [
    "grid_dim_x",
    "grid_dim_y",
    "grid_dim_z",
    "block_dim_x",
    "block_dim_y",
    "block_dim_z",
    "ctas",
    "threads_per_cta",
    "warps_per_cta",
    "dynamic_smem_bytes",

    "max_blocks_by_threads",
    "max_blocks_by_warps",
    "max_blocks_by_regs",
    "max_blocks_by_smem",
    "resident_blocks_per_sm_theoretical",
    "theoretical_occupancy",
]

BAD_RUNTIME_PREFIXES = (
    "ncu_device_attribute_",
    "ncu_numa_",
    "ncu_nvlink_",
    "ncu_profiler_",
    "ncu_launch_",
    "ncu_sass_",
    "ncu_c2clink_",
    "ncu_derived_tempMetric",
)

BAD_RUNTIME_COLS = {
    "ncu_returncode",
    "ncu_error",
    "ncu_wall_s",
    "ncu_raw_stdout",
    "ncu_raw_stderr",

    # identifiers / launch strings
    "ncu_Context",
    "ncu_Stream",
    "ncu_Device",
    "ncu_CC",
    "ncu_Block_Size",
    "ncu_Grid_Size",
}

RUNTIME_KEYWORDS = [
    # basic measured runtime / work amount
    "gpu_time_duration",
    "inst_executed",

    # occupancy / scheduling / warp activity
    "occupancy",
    "waves_per_multiprocessor",
    "sm_cycles_active",
    "sm_throughput",
    "smsp_issue_active",
    "smsp_warps_active",
    "smsp_warps_eligible",
    "warps_active",

    # memory pressure / throughput
    "dram_bytes",
    "dram_cycles_active",
    "dram_throughput",
    "compute_memory_throughput",
    "compute_memory_request_throughput",
    "memory_throughput",
    "lts_throughput",
    "l1tex_throughput",

    # cache / sector behavior
    "sector_hit_rate",
    "sectors_lookup_miss",
    "t_sectors",
    "d_sectors",

    # pipeline activity
    "pipe_alu",
    "pipe_fma",
    "pipe_fmaheavy",
    "pipe_fp64",
    "pipe_tensor",
    "pipe_lsu",
    "pipe_tex",
    "pipe_xu",
    "mio",

    # spilling
    "spilling",
]


def runtime_cols(df):
    cols = []

    for c in df.columns:
        if not c.startswith("ncu_"):
            continue

        if c in BAD_RUNTIME_COLS:
            continue

        if any(c.startswith(p) for p in BAD_RUNTIME_PREFIXES):
            continue

        cl = c.lower()
        if any(k in cl for k in RUNTIME_KEYWORDS):
            cols.append(c)

    return unique_keep_order(cols)


DROP_ALWAYS = {
    # identifiers / strings
    "workload",
    "kernel",
    "requested_pair",
    "requested_core_clock",
    "requested_mem_clock",
    "is_auto_baseline",
    "source_folder",
    "source_ptx",
    "kernel_entry",
    "workload_ptx",
    "source_folder_launch",
    "source_cu",
    "workload_launch",
    "kernel_func_launched",
    "join_workload",
    "join_kernel",
    "dim3_exprs",
    "constants_keys",

    "requested_core_clock_original",
    "requested_mem_clock_original",
    "requested_pairs_merged",
    "num_requested_pairs_merged",

    # measured target-ish columns
    "applied_core_clock",
    "applied_mem_clock",
    "n",
    "measured_launches_mean",
    "measured_launches_std",
    "total_time_mean_ms",
    "total_time_std_ms",
    "total_energy_mean_mJ",
    "total_energy_std_mJ",
    "avg_power_mean_W",
    "avg_power_std_W",
    "time_per_launch_mean_ms",
    "time_per_launch_std_ms",
    "energy_per_launch_mean_mJ",
    "energy_per_launch_std_mJ",

    # auto baseline columns
    "auto_time_per_launch_mean_ms",
    "auto_energy_per_launch_mean_mJ",
    "auto_avg_power_mean_W",

    # normalized targets
    "time_ratio_to_auto",
    "energy_ratio_to_auto",
    "power_ratio_to_auto",
    "speedup_vs_auto",
    "slowdown_vs_auto_pct",
    "energy_saving_vs_auto_pct",
}


# remove dupelicate colums
def unique_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# create table for skloearn
def clean_numeric_features(df, cols, drop_constant=False):
    cols = unique_keep_order([c for c in cols if c in df.columns])
    out = pd.DataFrame(index=df.index)
    kept = []

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            out[c] = s
            kept.append(c)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if drop_constant:
        nonconstant = []
        for c in kept:
            if out[c].nunique(dropna=False) > 1:
                nonconstant.append(c)
        out = out[nonconstant]
        kept = nonconstant

    return out, kept


# define which feature combination to test
def feature_sets(df):
    ptx_cols = unique_keep_order(GROUPED_PTX_COLS)
    launch_cols = unique_keep_order(LAUNCH_COLS)
    runtime = runtime_cols(df)

    print(f"PTX columns selected: {len([c for c in ptx_cols if c in df.columns])}")
    print(f"Launch columns selected: {len([c for c in launch_cols if c in df.columns])}")
    print(f"Runtime columns selected: {len([c for c in runtime if c in df.columns])}")
    print("First runtime columns:")
    for c in [c for c in runtime if c in df.columns][:40]:
        print(f"  {c}")

    return {
        "ptx": unique_keep_order(CLOCK_COLS + ptx_cols),
        "launch": unique_keep_order(CLOCK_COLS + launch_cols),
        "runtime": unique_keep_order(CLOCK_COLS + runtime),

        "ptx_launch": unique_keep_order(CLOCK_COLS + ptx_cols + launch_cols),
        "ptx_runtime": unique_keep_order(CLOCK_COLS + ptx_cols + runtime),
        "launch_runtime": unique_keep_order(CLOCK_COLS + launch_cols + runtime),

        "ptx_launch_runtime": unique_keep_order(CLOCK_COLS + ptx_cols + launch_cols + runtime),
    }


# define models to test
def models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=512,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=512,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=512,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ),
    }


def timing_stats(train_times_s: list[float], predict_times_s: list[float],
                 test_rows: list[int]) -> dict:
    """Summarize LOKO fold timings; per-row predict cost is the runtime-relevant metric."""
    train = np.asarray(train_times_s, dtype=float)
    pred = np.asarray(predict_times_s, dtype=float)
    rows = np.asarray(test_rows, dtype=int)
    per_row_ms = (pred / np.clip(rows, 1, None)) * 1000.0

    return {
        "n_folds": int(len(train)),
        "n_predict_rows": int(rows.sum()),
        "train_total_s": float(train.sum()),
        "train_mean_fold_s": float(train.mean()),
        "train_median_fold_s": float(np.median(train)),
        "train_p90_fold_s": float(np.percentile(train, 90)),
        "predict_total_s": float(pred.sum()),
        "predict_mean_fold_s": float(pred.mean()),
        "predict_median_fold_s": float(np.median(pred)),
        "predict_p90_fold_s": float(np.percentile(pred, 90)),
        "predict_mean_per_row_ms": float((pred.sum() / max(int(rows.sum()), 1)) * 1000.0),
        "predict_median_per_row_ms": float(np.median(per_row_ms)),
        "predict_p90_per_row_ms": float(np.percentile(per_row_ms, 90)),
    }


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-12
    ape = np.abs((y_pred - y_true) / np.clip(np.abs(y_true), eps, None)) * 100.0

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)), # mean absolute error
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))), # root mean squared error
        "mape_pct": float(np.mean(ape)), # mean absolute percentage error
        "median_ape_pct": float(np.median(ape)), # median absolute percentage error
        "p90_ape_pct": float(np.percentile(ape, 90)), # 90th percentile of absolute percentage error
        "max_ape_pct": float(np.max(ape)), # maximum absolute percentage error
        "mean_true": float(np.mean(y_true)), # mean of true values
        "mean_pred": float(np.mean(y_pred)), # mean of predicted values
    }


# main function
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_csv(IN_CSV)
    df["kernel_id"] = df["join_workload"].astype(str) + "::" + df["join_kernel"].astype(str)

    print(f"Rows: {len(df)}")
    print(f"Kernels: {df['kernel_id'].nunique()}")
    print(f"Columns: {len(df.columns)}")

    fsets = feature_sets(df)
    model_defs = models()

    all_metric_rows = []
    all_pred_rows = []
    all_fold_timing_rows = []

    # Get all kernels for LOKO
    kernels = sorted(df["kernel_id"].unique())

    # iterate over targets and feature sets
    for target_name, target_col in TARGETS.items():
        print()
        print(f"TARGET: {target_name} ({target_col})")

        for fset_name, cols in fsets.items():
            print(f"  Feature set: {fset_name}")

            for model_name, model in model_defs.items():
                y_true_all = []
                y_pred_all = []
                pred_parts = []
                fold_train_s = []
                fold_predict_s = []
                fold_test_rows = []

                # LOKO loop: leave kernel out, train on remaining, predict left out kernel
                for kid in kernels:
                    train_idx = df["kernel_id"] != kid
                    test_idx = df["kernel_id"] == kid

                    # split train and test
                    train = df.loc[train_idx].copy()
                    test = df.loc[test_idx].copy()

                    X_train, kept_cols = clean_numeric_features(train, cols, drop_constant=True)
                    X_test, _ = clean_numeric_features(test, kept_cols, drop_constant=False)

                    # Align columns exactly.
                    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

                    y_train_raw = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
                    y_test_raw = pd.to_numeric(test[target_col], errors="coerce").to_numpy(dtype=float)

                    eps = 1e-12
                    y_train_raw = np.clip(y_train_raw, eps, None)
                    y_test_raw = np.clip(y_test_raw, eps, None)

                    # Train on log ratios instead of raw ratios.
                    y_train_log = np.log(y_train_raw)

                    # fit model
                    fold_model = clone(model)
                    t0 = time.perf_counter()
                    fold_model.fit(X_train, y_train_log)
                    train_s = time.perf_counter() - t0

                    # predict held-out kernel
                    t0 = time.perf_counter()
                    y_pred_log = fold_model.predict(X_test)
                    predict_s = time.perf_counter() - t0
                    y_pred_raw = np.exp(y_pred_log)

                    fold_train_s.append(train_s)
                    fold_predict_s.append(predict_s)
                    fold_test_rows.append(len(test))

                    all_fold_timing_rows.append({
                        "target_name": target_name,
                        "feature_set": fset_name,
                        "model": model_name,
                        "held_out_kernel_id": kid,
                        "n_train_rows": int(len(train)),
                        "n_test_rows": int(len(test)),
                        "train_s": train_s,
                        "predict_s": predict_s,
                        "predict_per_row_ms": (predict_s / max(len(test), 1)) * 1000.0,
                        "n_features": len(kept_cols),
                    })

                    # store true/pred values
                    y_true_all.extend(y_test_raw.tolist())
                    y_pred_all.extend(y_pred_raw.tolist())

                    part = test[[
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
                    ]].copy()

                    part["target_name"] = target_name
                    part["target_col"] = target_col
                    part["feature_set"] = fset_name
                    part["model"] = model_name
                    part["y_true"] = y_test_raw
                    part["y_pred"] = y_pred_raw
                    part["abs_pct_error"] = (
                        np.abs((y_pred_raw - y_test_raw) / np.clip(np.abs(y_test_raw), eps, None)) * 100.0
                    )
                    part["n_features"] = len(kept_cols)

                    pred_parts.append(part)

                # calculate metrics
                m = metrics(y_true_all, y_pred_all)
                # update metrics with added labesl
                m.update({
                    "target_name": target_name,
                    "target_col": target_col,
                    "feature_set": fset_name,
                    "model": model_name,
                    "n_features": int(len(kept_cols)),
                })
                m.update(timing_stats(fold_train_s, fold_predict_s, fold_test_rows))

                all_metric_rows.append(m)
                all_pred_rows.append(pd.concat(pred_parts, ignore_index=True))

                print(
                    f"    {model_name:13s} "
                    f"MAPE={m['mape_pct']:.2f}% "
                    f"MedAPE={m['median_ape_pct']:.2f}% "
                    f"P90={m['p90_ape_pct']:.2f}%  "
                    f"train={m['train_total_s']:.1f}s "
                    f"predict={m['predict_total_s']:.2f}s "
                    f"({m['predict_median_per_row_ms']:.3f}ms/row)"
                )

    metrics_df = pd.DataFrame(all_metric_rows)
    preds_df = pd.concat(all_pred_rows, ignore_index=True)
    fold_timings_df = pd.DataFrame(all_fold_timing_rows)

    metrics_path = OUT_DIR / "normalized_loko_metrics.csv"
    preds_path = OUT_DIR / "normalized_loko_predictions.csv"
    timings_path = OUT_DIR / "normalized_loko_fold_timings.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    fold_timings_df.to_csv(timings_path, index=False)

    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {preds_path}")
    print(f"Wrote {timings_path}")

    print()
    print("Timing summary (median predict cost per row = runtime-relevant overhead):")
    timing_cols = [
        "target_name", "feature_set", "model",
        "train_total_s", "train_median_fold_s",
        "predict_total_s", "predict_median_per_row_ms", "predict_p90_per_row_ms",
    ]
    print(
        metrics_df.sort_values(["target_name", "predict_median_per_row_ms"])
        .groupby("target_name")
        .head(6)[timing_cols]
        .to_string(index=False)
    )

    print()
    print("Best by target:")
    best = (
        metrics_df
        .sort_values(["target_name", "mae", "mape_pct", "median_ape_pct"])
        .groupby("target_name")
        .head(8)
    )
    print(best[[
        "target_name",
        "feature_set",
        "model",
        "mape_pct",
        "median_ape_pct",
        "p90_ape_pct",
        "mae",
        
        "rmse",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()