#!/usr/bin/env python3
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd

PTX_CSV = Path.home() / "thesis/results/ptx_features_all.csv"
LAUNCH_CSV = Path.home() / "thesis/results/launch_features_all.csv"
AGG_CSV = Path.home() / "thesis/results/master/aggregated_all_kernels_effective.csv"
NCU_CSV = Path.home() / "thesis/results/ncu_features_all.csv"
OUT_CSV = Path.home() / "thesis/results/master/ml_dataset_ptx_launch_runtime_normalized.csv"

# RTX 3080 Ti / Ampere-ish constants.
# These are approximate theoretical occupancy features, not measured occupancy.
MAX_THREADS_PER_SM = 1536
MAX_WARPS_PER_SM = 48
MAX_BLOCKS_PER_SM = 16
REGS_PER_SM = 65536
SHARED_MEM_PER_SM = 100 * 1024


def norm(s) -> str:
    return str(s).lower().replace("-", "").replace("_", "").replace(" ", "")


def workload_from_folder(folder: str) -> str:
    f = str(folder).upper()
    if f == "CORR":
        return "correlation"
    if f == "COVAR":
        return "covariance"
    return str(folder)


def kernel_from_ptx_filename(filename: str, source_folder: str) -> str:
    stem = Path(str(filename)).stem
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.replace("_repeat", "")

    folder = str(source_folder).upper()

    # PolyBench mappings.
    special = {
        "2DCONV": {
            "2dconv_kernel": "convolution2D_kernel",
        },
        "2MM": {
            "2mm_kernel1": "mm2_kernel1",
            "2mm_kernel2": "mm2_kernel2",
        },
        "3MM": {
            "3mm_kernel1": "mm3_kernel1",
            "3mm_kernel2": "mm3_kernel2",
            "3mm_kernel3": "mm3_kernel3",
        },
        "CORR": {
            "correlation_mean_kernel": "mean_kernel",
            "correlation_std_kernel": "std_kernel",
            "correlation_reduce_kernel": "reduce_kernel",
            "correlation_corr_kernel": "corr_kernel",
        },
        "COVAR": {
            "covariance_mean_kernel": "mean_kernel",
            "covariance_reduce_kernel": "reduce_kernel",
            "covariance_covar_kernel": "covar_kernel",
        },
        "JACOBI1D": {
            "jacobi1d_step1": "kernel1",
            "jacobi1d_step2": "kernel2",
        },
        "JACOBI2D": {
            "jacobi2d_kernel1": "kernel1",
            "jacobi2d_kernel2": "kernel2",
        },
        "FDTD-2D": {
            "fdtd2d_step1": "step1",
            "fdtd2d_step2": "step2",
            "fdtd2d_step3": "step3",
        },
        "DOITGEN": {
            "doitgen_kernel1": "kernel1",
            "doitgen_kernel2": "kernel2",
        },
        "GRAMSCHMIDT": {
            "gramschmidt_kernel1": "kernel1",
            "gramschmidt_kernel2": "kernel2",
            "gramschmidt_kernel3": "kernel3",
        },
        "MVT": {
            "mvt_kernel1": "kernel1",
            "mvt_kernel2": "kernel2",
        },
    }

    if folder in special and stem in special[folder]:
        return special[folder][stem]

    # Rodinia mappings: remove workload prefix from isolated filenames.
    prefixes = [
        "pre_euler3d_",
        "euler3d_",
        "hotspot3d_",
        "hotspot_",
        "kmeans_",
        "lavamd_",
        "myocyte_",
        "nn_",
        "nw_",
        "particlefilter_",
        "pathfinder_",
        "streamcluster_",
        "srad_",

        # Parboil, for later full-grid merge
        "histo_",
        "lbm_",
        "mriq_",
        "sgemm_",
        "spmv_",
        "stencil_",
        "tpacf_",
    ]

    for pref in prefixes:
        if stem.startswith(pref):
            return stem[len(pref):]

    return stem


def occupancy_features(row: pd.Series) -> pd.Series:
    threads = pd.to_numeric(row.get("threads_per_cta"), errors="coerce")
    warps = pd.to_numeric(row.get("warps_per_cta"), errors="coerce")
    regs = pd.to_numeric(row.get("registers_per_thread_approx"), errors="coerce")
    smem = pd.to_numeric(row.get("dynamic_smem_bytes"), errors="coerce")

    if pd.isna(threads) or threads <= 0:
        return pd.Series({
            "max_blocks_by_threads": np.nan,
            "max_blocks_by_warps": np.nan,
            "max_blocks_by_regs": np.nan,
            "max_blocks_by_smem": np.nan,
            "resident_blocks_per_sm_theoretical": np.nan,
            "theoretical_occupancy": np.nan,
        })

    if pd.isna(warps) or warps <= 0:
        warps = math.ceil(threads / 32)

    if pd.isna(regs) or regs <= 0:
        max_blocks_by_regs = MAX_BLOCKS_PER_SM
    else:
        max_blocks_by_regs = math.floor(REGS_PER_SM / (regs * threads))
        max_blocks_by_regs = max(max_blocks_by_regs, 0)

    if pd.isna(smem) or smem <= 0:
        max_blocks_by_smem = MAX_BLOCKS_PER_SM
    else:
        max_blocks_by_smem = math.floor(SHARED_MEM_PER_SM / smem)
        max_blocks_by_smem = max(max_blocks_by_smem, 0)

    max_blocks_by_threads = math.floor(MAX_THREADS_PER_SM / threads)
    max_blocks_by_warps = math.floor(MAX_WARPS_PER_SM / warps)

    resident = min(
        MAX_BLOCKS_PER_SM,
        max_blocks_by_threads,
        max_blocks_by_warps,
        max_blocks_by_regs,
        max_blocks_by_smem,
    )

    occupancy = (resident * warps) / MAX_WARPS_PER_SM

    return pd.Series({
        "max_blocks_by_threads": max_blocks_by_threads,
        "max_blocks_by_warps": max_blocks_by_warps,
        "max_blocks_by_regs": max_blocks_by_regs,
        "max_blocks_by_smem": max_blocks_by_smem,
        "resident_blocks_per_sm_theoretical": resident,
        "theoretical_occupancy": occupancy,
    })


def main():
    ptx = pd.read_csv(PTX_CSV)
    launch = pd.read_csv(LAUNCH_CSV)
    ncu = pd.read_csv(NCU_CSV)
    agg = pd.read_csv(AGG_CSV)

    # PTX join keys.
    ptx["join_workload"] = ptx["source_folder"].map(lambda x: norm(workload_from_folder(x)))
    ptx["join_kernel"] = ptx.apply(
        lambda r: norm(kernel_from_ptx_filename(r["source_ptx"], r["source_folder"])),
        axis=1,
    )

    # Launch join keys.
    launch["join_workload"] = launch["source_folder"].map(lambda x: norm(workload_from_folder(x)))
    launch["join_kernel"] = launch.apply(
        lambda r: norm(kernel_from_ptx_filename(str(r["source_cu"]).replace(".cu", ".ptx"), r["source_folder"])),
        axis=1,
    )

    # NCU join keys.
    ncu["join_workload"] = ncu["join_workload"].map(norm)
    ncu["join_kernel"] = ncu["join_kernel"].map(norm)

    # Keep only successful NCU profiles.
    ncu["ncu_returncode"] = pd.to_numeric(ncu["ncu_returncode"], errors="coerce")
    ncu = ncu[ncu["ncu_returncode"] == 0].copy()

    # There should be one NCU row per kernel.
    dup = ncu.duplicated(["join_workload", "join_kernel"], keep=False)
    if dup.any():
        print("Duplicate NCU rows:")
        print(ncu.loc[dup, ["join_workload", "join_kernel", "source_cu"]].to_string(index=False))
        raise SystemExit("Duplicate NCU rows")

    # Measurement join keys.
    agg["join_workload"] = agg["workload"].map(norm)
    agg["join_kernel"] = agg["kernel"].map(norm)

    core_s = agg["requested_core_clock"].astype(str).str.lower()
    mem_s = agg["requested_mem_clock"].astype(str).str.lower()

    auto = agg[(core_s == "auto") & (mem_s == "auto")].copy()
    fixed = agg[(core_s != "auto") & (mem_s != "auto")].copy()

    fixed["effective_core_clock"] = pd.to_numeric(fixed["effective_core_clock"], errors="coerce")
    fixed["effective_mem_clock"] = pd.to_numeric(fixed["effective_mem_clock"], errors="coerce")

    auto_base = (
        auto.groupby(["join_workload", "join_kernel"], as_index=False)
        .agg({
            "time_per_launch_mean_ms": "mean",
            "energy_per_launch_mean_mJ": "mean",
            "avg_power_mean_W": "mean",
        })
        .rename(columns={
            "time_per_launch_mean_ms": "auto_time_per_launch_mean_ms",
            "energy_per_launch_mean_mJ": "auto_energy_per_launch_mean_mJ",
            "avg_power_mean_W": "auto_avg_power_mean_W",
        })
    )

    fixed = fixed.merge(
        auto_base,
        on=["join_workload", "join_kernel"],
        how="left",
        indicator="auto_merge",
    )

    missing_auto = fixed[fixed["auto_merge"] != "both"]
    if len(missing_auto):
        print("Missing auto baseline rows:")
        print(missing_auto[["workload", "kernel"]].drop_duplicates().to_string(index=False))
        raise SystemExit("Cannot build normalized dataset without auto rows")

    eps = 1e-12

    fixed["time_ratio_to_auto"] = (
        fixed["time_per_launch_mean_ms"] / fixed["auto_time_per_launch_mean_ms"].clip(lower=eps)
    )
    fixed["energy_ratio_to_auto"] = (
        fixed["energy_per_launch_mean_mJ"] / fixed["auto_energy_per_launch_mean_mJ"].clip(lower=eps)
    )
    fixed["power_ratio_to_auto"] = (
        fixed["avg_power_mean_W"] / fixed["auto_avg_power_mean_W"].clip(lower=eps)
    )

    fixed["speedup_vs_auto"] = 1.0 / fixed["time_ratio_to_auto"].clip(lower=eps)
    fixed["slowdown_vs_auto_pct"] = (fixed["time_ratio_to_auto"] - 1.0) * 100.0
    fixed["energy_saving_vs_auto_pct"] = (1.0 - fixed["energy_ratio_to_auto"]) * 100.0

    fixed["core_clock_norm"] = fixed["effective_core_clock"] / fixed["effective_core_clock"].max()
    fixed["mem_clock_norm"] = fixed["effective_mem_clock"] / fixed["effective_mem_clock"].max()
    fixed["inv_core_clock"] = 1.0 / fixed["effective_core_clock"].clip(lower=eps)
    fixed["inv_mem_clock"] = 1.0 / fixed["effective_mem_clock"].clip(lower=eps)
    fixed["core_mem_clock_product_norm"] = fixed["core_clock_norm"] * fixed["mem_clock_norm"]

    merged = fixed.merge(
        ptx,
        on=["join_workload", "join_kernel"],
        how="left",
        suffixes=("", "_ptx"),
        indicator="ptx_merge",
    )

    missing_ptx = merged[merged["ptx_merge"] != "both"]
    if len(missing_ptx):
        print("Missing PTX matches:")
        print(missing_ptx[["workload", "kernel", "join_workload", "join_kernel"]].drop_duplicates().to_string(index=False))
        raise SystemExit("PTX merge failed")

    merged = merged.merge(
        launch,
        on=["join_workload", "join_kernel"],
        how="left",
        suffixes=("", "_launch"),
        indicator="launch_merge",
    )

    missing_launch = merged[merged["launch_merge"] != "both"]
    if len(missing_launch):
        print("Missing launch matches:")
        print(missing_launch[["workload", "kernel", "join_workload", "join_kernel"]].drop_duplicates().to_string(index=False))
        raise SystemExit("Launch merge failed")

    occ = merged.apply(occupancy_features, axis=1)
    merged = pd.concat([merged, occ], axis=1)

    merged = merged.merge(
        ncu,
        on=["join_workload", "join_kernel"],
        how="left",
        suffixes=("", "_ncu"),
        indicator="ncu_merge",
    )

    missing_ncu = merged[merged["ncu_merge"] != "both"]
    if len(missing_ncu):
        print("Missing NCU matches:")
        print(
            missing_ncu[
                ["workload", "kernel", "join_workload", "join_kernel"]
            ].drop_duplicates().to_string(index=False)
        )
        raise SystemExit("NCU merge failed")

    # Remove merge indicators, keep join keys for debugging.
    merged = merged.drop(
        columns=["auto_merge", "ptx_merge", "launch_merge", "ncu_merge"],
        errors="ignore",
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV}")
    print(f"Rows: {len(merged)}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Unique kernels: {merged[['join_workload', 'join_kernel']].drop_duplicates().shape[0]}")
    print()
    print("Target summary:")
    print(merged[[
        "time_ratio_to_auto",
        "energy_ratio_to_auto",
        "speedup_vs_auto",
        "energy_saving_vs_auto_pct",
        "theoretical_occupancy",
    ]].describe().to_string())


if __name__ == "__main__":
    main()