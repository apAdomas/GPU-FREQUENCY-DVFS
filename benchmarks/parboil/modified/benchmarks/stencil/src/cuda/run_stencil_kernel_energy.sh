#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
source "$HOME/parboil_2.5/benchmarks/parboil_energy_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/parboil_2.5/benchmarks/stencil/src/cuda"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/stencil_kernel_energy/$RUNSTAMP"
WORKLOAD_PREFIX="stencil"

STENCIL_INPUT="$HOME/parboil_2.5/datasets/stencil/small/input/128x128x32.bin"
STENCIL_NX=128
STENCIL_NY=128
STENCIL_NZ=32

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)
REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=("block2d")
KERNEL_EXES=("./01_stencil_block2d_repeat.exe")
KERNEL_ARGS=("$STENCIL_INPUT $STENCIL_NX $STENCIL_NY $STENCIL_NZ")

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"
cd "$WORKDIR"

oracle_init
oracle_run_all

reset_clocks
echo "Done."
echo "Results written to: $OUTDIR"
