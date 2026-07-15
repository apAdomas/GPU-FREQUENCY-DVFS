#!/usr/bin/env python3
from pathlib import Path
import csv

RESULT_DIRS = [
    # PolyBench-GPU
    Path("~/thesis/results/2dconv_kernel_energy").expanduser(),
    Path("~/thesis/results/2mm_kernel_energy").expanduser(),
    Path("~/thesis/results/3mm_kernel_energy").expanduser(),
    Path("~/thesis/results/atax_kernel_energy").expanduser(),
    Path("~/thesis/results/bicg_kernel_energy").expanduser(),
    Path("~/thesis/results/correlation_kernel_energy").expanduser(),
    Path("~/thesis/results/covariance_kernel_energy").expanduser(),
    Path("~/thesis/results/doitgen_kernel_energy").expanduser(),
    Path("~/thesis/results/fdtd2d_kernel_energy").expanduser(),
    Path("~/thesis/results/gemm_kernel_energy").expanduser(),
    Path("~/thesis/results/gemver_kernel_energy").expanduser(),
    Path("~/thesis/results/gesummv_kernel_energy").expanduser(),
    Path("~/thesis/results/gramschmidt_kernel_energy").expanduser(),
    Path("~/thesis/results/jacobi1d_kernel_energy").expanduser(),
    Path("~/thesis/results/jacobi2d_kernel_energy").expanduser(),
    Path("~/thesis/results/mvt_kernel_energy").expanduser(),
    Path("~/thesis/results/syr2k_kernel_energy").expanduser(),
    Path("~/thesis/results/syrk_kernel_energy").expanduser(),

    # Rodinia
    Path("~/thesis/results/hotspot_kernel_energy").expanduser(),
    Path("~/thesis/results/hotspot3d_kernel_energy").expanduser(),
    Path("~/thesis/results/kmeans_kernel_energy").expanduser(),
    Path("~/thesis/results/lavamd_kernel_energy").expanduser(),
    Path("~/thesis/results/myocyte_kernel_energy").expanduser(),
    Path("~/thesis/results/nn_kernel_energy").expanduser(),
    Path("~/thesis/results/nw_kernel_energy").expanduser(),
    Path("~/thesis/results/particlefilter_kernel_energy").expanduser(),
    Path("~/thesis/results/pathfinder_kernel_energy").expanduser(),
    Path("~/thesis/results/srad_v1_kernel_energy").expanduser(),
    Path("~/thesis/results/streamcluster_kernel_energy").expanduser(),
    Path("~/thesis/results/euler3d_kernel_energy").expanduser(),
    Path("~/thesis/results/pre_euler3d_kernel_energy").expanduser(),
]

OUTDIR = Path("~/thesis/results/master").expanduser()
AGG_OUT = OUTDIR / "aggregated_all_kernels.csv"
ORACLE_OUT = OUTDIR / "oracle_summary_all_kernels.csv"


def latest_run_dir(results_dir: Path) -> Path | None:
    if not results_dir.exists() or not results_dir.is_dir():
        return None
    runs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    if not runs:
        return None
    return runs[-1]


def read_csv_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def infer_workload_name(results_dir: Path) -> str:
    name = results_dir.name
    if name.endswith("_kernel_energy"):
        return name[:-14]
    return name


def merge_files(input_specs, output_path: Path):
    all_rows = []
    all_fields = []

    for spec in input_specs:
        csv_path = spec["csv_path"]
        workload = spec["workload"]

        if not csv_path.exists():
            print(f"SKIP missing file: {csv_path}")
            continue

        fieldnames, rows = read_csv_rows(csv_path)
        if not rows:
            print(f"SKIP empty file: {csv_path}")
            continue

        for row in rows:
            row["workload"] = workload
            all_rows.append(row)

        for field in ["workload"] + fieldnames:
            if field not in all_fields:
                all_fields.append(field)

    if not all_rows:
        raise SystemExit(f"No rows collected for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {output_path}")
    print(f"Rows: {len(all_rows)}")


def main():
    agg_inputs = []
    oracle_inputs = []

    for results_dir in RESULT_DIRS:
        latest = latest_run_dir(results_dir)
        if latest is None:
            print(f"SKIP no run dir: {results_dir}")
            continue

        workload = infer_workload_name(results_dir)

        agg_inputs.append({
            "workload": workload,
            "csv_path": latest / "aggregated_by_kernel_clock.csv",
        })

        oracle_inputs.append({
            "workload": workload,
            "csv_path": latest / "oracle_summary_by_kernel.csv",
        })

    merge_files(agg_inputs, AGG_OUT)
    merge_files(oracle_inputs, ORACLE_OUT)


if __name__ == "__main__":
    main()
