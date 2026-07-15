#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import os
import re
import subprocess
import time
from io import StringIO

import pandas as pd

POLY_ROOT = Path.home() / "polybenchGpu" / "CUDA"
RODINIA_ROOT = Path.home() / "rodinia_3.1" / "cuda"
PARBOIL_ROOT = Path.home() / "parboil_2.5" / "benchmarks"

OUT_CSV = Path.home() / "thesis/results/ncu_features_all.csv"
RAW_DIR = Path.home() / "thesis/results/ncu_raw"

RODINIA_DATA = Path.home() / "rodinia_3.1" / "data"
PARBOIL_DATA = Path.home() / "parboil_2.5" / "datasets"

NCU_SECTIONS = [
    "SpeedOfLight",
    "Occupancy",
    "LaunchStats",
    "SchedulerStats",
    "MemoryWorkloadAnalysis",
]


def norm(x: str) -> str:
    return str(x).lower().replace("-", "").replace("_", "").replace(" ", "")


def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def is_polybench_benchmark_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    n = p.name
    if n.startswith("_") or n.startswith("."):
        return False
    if n in {"scripts", "common"}:
        return False
    return any(c.isupper() for c in n)


def is_isolated_kernel_file(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".cu"
        and "_repeat" in p.stem
        and p.name[:2].isdigit()
    )


def workload_from_folder(folder: str) -> str:
    f = str(folder).upper()
    if f == "CORR":
        return "correlation"
    if f == "COVAR":
        return "covariance"
    return str(folder)


def rodinia_source_folder(cu: Path) -> str:
    stem = cu.stem

    if "_pre_euler3d_" in stem:
        return "pre_euler3d"
    if "_euler3d_" in stem:
        return "euler3d"

    parts = cu.relative_to(RODINIA_ROOT).parts

    if "srad_v1" in parts:
        return "srad_v1"

    aliases = {
        "hotspot3D": "hotspot3d",
        "lavaMD": "lavamd",
    }
    return aliases.get(cu.parent.name, cu.parent.name)


def parboil_source_folder(cu: Path) -> str:
    # parboil_2.5/benchmarks/<bench>/src/cuda/NN_*_repeat.cu -> <bench>
    return cu.relative_to(PARBOIL_ROOT).parts[0]


def kernel_from_filename(filename: str, source_folder: str) -> str:
    stem = Path(str(filename)).stem
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.replace("_repeat", "")

    folder = str(source_folder).upper()

    special = {
        "2DCONV": {"2dconv_kernel": "convolution2D_kernel"},
        "2MM": {"2mm_kernel1": "mm2_kernel1", "2mm_kernel2": "mm2_kernel2"},
        "3MM": {"3mm_kernel1": "mm3_kernel1", "3mm_kernel2": "mm3_kernel2", "3mm_kernel3": "mm3_kernel3"},
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
        "JACOBI1D": {"jacobi1d_step1": "kernel1", "jacobi1d_step2": "kernel2"},
        "JACOBI2D": {"jacobi2d_kernel1": "kernel1", "jacobi2d_kernel2": "kernel2"},
        "FDTD-2D": {"fdtd2d_step1": "step1", "fdtd2d_step2": "step2", "fdtd2d_step3": "step3"},
        "DOITGEN": {"doitgen_kernel1": "kernel1", "doitgen_kernel2": "kernel2"},
        "GRAMSCHMIDT": {
            "gramschmidt_kernel1": "kernel1",
            "gramschmidt_kernel2": "kernel2",
            "gramschmidt_kernel3": "kernel3",
        },
        "MVT": {"mvt_kernel1": "kernel1", "mvt_kernel2": "kernel2"},
    }

    if folder in special and stem in special[folder]:
        return special[folder][stem]

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


def find_executable_for_cu(cu: Path) -> Path | None:
    p = cu.parent / (cu.stem + "_profile.exe")
    if p.exists() and p.is_file() and os.access(p, os.X_OK):
        return p
    return None


def source_files(suite: str = "all"):
    if suite in {"all", "polybench"}:
        for bench in sorted(POLY_ROOT.iterdir()):
            if not is_polybench_benchmark_dir(bench):
                continue
            for cu in sorted(bench.iterdir()):
                if is_isolated_kernel_file(cu):
                    yield bench.name, cu

    if suite in {"all", "rodinia"}:
        for cu in sorted(RODINIA_ROOT.rglob("*_repeat.cu")):
            if is_isolated_kernel_file(cu):
                yield rodinia_source_folder(cu), cu

    if suite in {"all", "parboil"}:
        for cu in sorted(PARBOIL_ROOT.rglob("*_repeat.cu")):
            if is_isolated_kernel_file(cu):
                yield parboil_source_folder(cu), cu


def jobs(suite: str = "all"):
    out = []
    for source_folder, cu in source_files(suite):
        exe = find_executable_for_cu(cu)
        if exe is None:
            print(f"WARNING no executable for {cu}")
            continue
        out.append((source_folder, cu, exe))
    return out


def sanitize_metric_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return "ncu_" + name


def parse_float(value):
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.lower() in {"n/a", "nan", "none"}:
        return None

    s = s.replace(",", "")
    s = s.replace("%", "")

    m = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", s, flags=re.I)
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_ncu_csv(stdout: str) -> dict:
    lines = stdout.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID","Process ID"') or line.startswith("ID,Process ID"):
            header_idx = i
            break

    if header_idx is None:
        return {}

    csv_text = "\n".join(lines[header_idx:])

    try:
        df = pd.read_csv(StringIO(csv_text))

        if len(df) < 2:
            return {}

        values = df.iloc[1].to_dict()

        out = {}
        for k, v in values.items():
            if k in {"ID", "Process ID", "Process Name", "Host Name", "Kernel Name"}:
                continue

            val = parse_float(v)
            if val is None:
                continue

            out[sanitize_metric_name(k)] = val

        return out

    except Exception as e:
        print(f"CSV parse failed: {e}")
        return {}


def get_exe_args(source_folder: str, kernel: str, exe: Path) -> list[str]:
    sf = norm(source_folder)

    # PolyBench kernels take no CLI args.
    if "polybenchGpu" in str(exe):
        return []

    # Parboil: mirror the per-workload run_*_kernel_energy.sh launch arguments.
    if "parboil_2.5" in str(exe):
        if sf == "histo":
            return [str(PARBOIL_DATA / "histo" / "default" / "input" / "img.bin")]
        if sf == "lbm":
            return [str(PARBOIL_DATA / "lbm" / "short" / "input" / "120_120_150_ldc.of")]
        if sf == "mriq":
            return [str(PARBOIL_DATA / "mri-q" / "small" / "input" / "32_32_32_dataset.bin")]
        if sf == "sgemm":
            return [
                str(PARBOIL_DATA / "sgemm" / "small" / "input" / "matrix1.txt"),
                str(PARBOIL_DATA / "sgemm" / "small" / "input" / "matrix2t.txt"),
            ]
        if sf == "spmv":
            return [
                str(PARBOIL_DATA / "spmv" / "small" / "input" / "1138_bus.mtx"),
                str(PARBOIL_DATA / "spmv" / "small" / "input" / "vector.bin"),
            ]
        if sf == "stencil":
            return [
                str(PARBOIL_DATA / "stencil" / "small" / "input" / "128x128x32.bin"),
                "128", "128", "32",
            ]
        if sf == "tpacf":
            return [
                str(PARBOIL_DATA / "tpacf" / "small" / "input"),
                "487", "100",
            ]
        return []

    # Rodinia: mirror the per-workload run_*_kernel_energy.sh launch arguments.
    if sf == "kmeans":
        return ["1000000", "34", "32"]

    if sf == "lavamd":
        return ["10"]

    if sf == "myocyte":
        return ["100", "256"]

    if sf == "nn":
        return ["10000000"]

    if sf == "nw":
        return ["4096", "10"]

    if sf == "particlefilter":
        return ["128", "128", "10", "100000"]

    if sf == "pathfinder":
        return ["100000", "100", "20"]

    if sf == "sradv1":
        if kernel in {"srad", "srad2"}:
            return ["2048", "2048", "0.5"]
        return ["2048", "2048"]

    if sf == "streamcluster":
        return ["1000000", "32", "64"]

    if sf == "hotspot":
        return [
            "512", "1", "60",
            str(RODINIA_DATA / "hotspot" / "temp_512"),
            str(RODINIA_DATA / "hotspot" / "power_512"),
        ]

    if sf == "hotspot3d":
        return [
            "512", "8", "60",
            str(RODINIA_DATA / "hotspot3D" / "power_512x8"),
            str(RODINIA_DATA / "hotspot3D" / "temp_512x8"),
        ]

    if sf in {"euler3d", "preeuler3d"}:
        return [str(RODINIA_DATA / "cfd" / "fvcorr.domn.097K")]

    return []


def run_ncu(exe: Path, timeout_s: float, exe_args: list[str]):
    cmd = [
        "ncu",
        "--profile-from-start", "off",
        "--target-processes", "all",
        "--kernel-name-base", "function",
        "--launch-count", "1",
        "--csv",
        "--page", "raw",
    ]

    for sec in NCU_SECTIONS:
        cmd += ["--section", sec]

    cmd += [str(exe)] + exe_args

    print(" ".join(cmd))

    return subprocess.run(
        cmd,
        cwd=str(exe.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )


def write_progress_csv(rows, out_path: Path):
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tmp.replace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument(
        "--suite",
        choices=["all", "polybench", "rodinia", "parboil"],
        default="all",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=1200.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    js = jobs(args.suite)
    js = js[args.start - 1:]

    if args.limit > 0:
        js = js[:args.limit]

    rows = []

    print(f"Kernels to profile: {len(js)}")
    print(f"Parsed output: {args.out}")
    print(f"Raw output dir: {args.raw_dir}")

    for i, (source_folder, cu, exe) in enumerate(js, start=1):
        workload = norm(workload_from_folder(source_folder))
        kernel = norm(kernel_from_filename(cu.name, source_folder))

        raw_base = safe_name(f"{i:03d}_{workload}_{kernel}_{exe.stem}")
        raw_stdout_path = args.raw_dir / f"{raw_base}.ncu.csv"
        raw_stderr_path = args.raw_dir / f"{raw_base}.ncu.err"

        print(f"\n[{i}/{len(js)}] {source_folder}/{cu.name}")
        print(f"raw csv: {raw_stdout_path.name}")

        base_row = {
            "source_folder": source_folder,
            "source_cu": cu.name,
            "executable": str(exe),
            "workload": workload,
            "kernel": kernel,
            "join_workload": workload,
            "join_kernel": kernel,
            "ncu_returncode": "",
            "ncu_error": "",
            "ncu_wall_s": "",
            "ncu_raw_stdout": str(raw_stdout_path),
            "ncu_raw_stderr": str(raw_stderr_path),
            "profile_args": "",
        }

        exe_args = get_exe_args(source_folder, kernel, exe)
        base_row["profile_args"] = " ".join(exe_args)

        t0 = time.time()

        try:
            res = run_ncu(exe, timeout_s=args.timeout_s, exe_args=exe_args)

            wall_s = time.time() - t0
            base_row["ncu_wall_s"] = wall_s
            base_row["ncu_returncode"] = res.returncode

            raw_stdout_path.write_text(res.stdout, encoding="utf-8", errors="ignore")
            raw_stderr_path.write_text(res.stderr, encoding="utf-8", errors="ignore")

            if res.returncode != 0:
                base_row["ncu_error"] = res.stderr[-1000:]
                print(f"FAILED wall={wall_s:.2f}s")
                print(res.stderr[-1000:])
            else:
                feats = parse_ncu_csv(res.stdout)
                base_row.update(feats)
                print(f"metrics={len(feats)} wall={wall_s:.2f}s")

        except subprocess.TimeoutExpired as e:
            wall_s = time.time() - t0
            base_row["ncu_wall_s"] = wall_s
            base_row["ncu_returncode"] = 124
            base_row["ncu_error"] = "timeout"
            print(f"TIMEOUT wall={wall_s:.2f}s")

            if e.stdout:
                raw_stdout_path.write_text(str(e.stdout), encoding="utf-8", errors="ignore")
            if e.stderr:
                raw_stderr_path.write_text(str(e.stderr), encoding="utf-8", errors="ignore")

        rows.append(base_row)

        # Save after every kernel.
        write_progress_csv(rows, args.out)

    print(f"\nWrote {args.out}")
    print(f"Rows: {len(rows)}")
    print(f"Raw files: {args.raw_dir}")


if __name__ == "__main__":
    main()
