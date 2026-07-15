#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
source "$HOME/parboil_2.5/benchmarks/parboil_energy_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/parboil_2.5/benchmarks/histo/src/cuda"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/histo_kernel_energy/$RUNSTAMP"
WORKLOAD_PREFIX="histo"

# All histo kernels run on the same input image.
HISTO_INPUT="$HOME/parboil_2.5/datasets/histo/default/input/img.bin"

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)
REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=("prescan" "intermediates" "main" "final")
KERNEL_EXES=(
  "./01_histo_prescan_repeat.exe"
  "./02_histo_intermediates_repeat.exe"
  "./03_histo_main_repeat.exe"
  "./04_histo_final_repeat.exe"
)
KERNEL_ARGS=("$HISTO_INPUT" "$HISTO_INPUT" "$HISTO_INPUT" "$HISTO_INPUT")

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"
cd "$WORKDIR"

oracle_init
oracle_run_all

reset_clocks
echo "Done."
echo "Results written to: $OUTDIR"
