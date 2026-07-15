#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
source "$HOME/parboil_2.5/benchmarks/parboil_energy_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/parboil_2.5/benchmarks/spmv/src/cuda"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/spmv_kernel_energy/$RUNSTAMP"
WORKLOAD_PREFIX="spmv"

SPMV_INPUT_MTX="$HOME/parboil_2.5/datasets/spmv/small/input/1138_bus.mtx"
SPMV_INPUT_VEC="$HOME/parboil_2.5/datasets/spmv/small/input/vector.bin"

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)
REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=("jds")
KERNEL_EXES=("./01_spmv_jds_repeat.exe")
KERNEL_ARGS=("$SPMV_INPUT_MTX $SPMV_INPUT_VEC")

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"
cd "$WORKDIR"

oracle_init
oracle_run_all

reset_clocks
echo "Done."
echo "Results written to: $OUTDIR"
