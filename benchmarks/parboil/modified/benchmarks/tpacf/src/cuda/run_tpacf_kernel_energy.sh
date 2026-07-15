#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
source "$HOME/parboil_2.5/benchmarks/parboil_energy_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/parboil_2.5/benchmarks/tpacf/src/cuda"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/tpacf_kernel_energy/$RUNSTAMP"
WORKLOAD_PREFIX="tpacf"

TPACF_INPUT_DIR="$HOME/parboil_2.5/datasets/tpacf/small/input"
TPACF_NPOINTS=487
TPACF_RANDOM_COUNT=100

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)
REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=("genhists")
KERNEL_EXES=("./01_tpacf_genhists_repeat.exe")
KERNEL_ARGS=("$TPACF_INPUT_DIR $TPACF_NPOINTS $TPACF_RANDOM_COUNT")

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"
cd "$WORKDIR"

oracle_init
oracle_run_all

reset_clocks
echo "Done."
echo "Results written to: $OUTDIR"
