#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/polybenchGpu/CUDA/SYR2K"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/syr2k_kernel_energy/$RUNSTAMP"

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)

REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=("syr2k_kernel")
KERNEL_EXES=(
  "./01_syr2k_kernel_repeat.exe"
)

WORKLOAD_PREFIX="syr2k"

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"

cd "$WORKDIR"

for exe in "${KERNEL_EXES[@]}"; do
  [ -x "$exe" ] || { echo "Missing executable: $exe"; exit 1; }
done

[ -x "$GLOBAL_WARMUP_EXE" ] || { echo "Missing global warmup exe: $GLOBAL_WARMUP_EXE"; exit 1; }

sudo nvidia-smi -pm 1 >/dev/null

run_global_warmup "$GLOBAL_WARMUP_EXE" "$GLOBAL_WARMUP_CORE" "$GLOBAL_WARMUP_MEM" \
  | tee "$OUTDIR/global_warmup.log"

reset_clocks
sleep 1

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

extract_result() {
  local key="$1"
  local file="$2"
  grep "^RESULT ${key}=" "$file" | tail -1 | cut -d= -f2
}

run_case() {
  local KERNEL="$1"
  local EXE="$2"
  local CORE_REQ="$3"
  local MEM_REQ="$4"
  local REP="$5"

  echo "=== kernel=$KERNEL core=$CORE_REQ mem=$MEM_REQ rep=$REP ==="

  RUN_DIR="$OUTDIR/$KERNEL/core_${CORE_REQ}_mem_${MEM_REQ}_rep_${REP}"
  mkdir -p "$RUN_DIR"

  reset_clocks
  sleep 1

  if [ "$CORE_REQ" = "auto" ] && [ "$MEM_REQ" = "auto" ]; then
    :
  elif [ "$CORE_REQ" = "auto" ]; then
    lock_mem_clock "$MEM_REQ"
  elif [ "$MEM_REQ" = "auto" ]; then
    lock_core_clock "$CORE_REQ"
  else
    lock_clocks "$CORE_REQ" "$MEM_REQ"
  fi

  sleep "$SLEEP_AFTER_CLOCK_LOCK"

  log_gpu_state > "$RUN_DIR/pre_run_gpu_state.txt"

  nvidia-smi \
    --query-gpu=clocks.current.graphics,clocks.current.memory \
    --format=csv,noheader \
    > "$RUN_DIR/locked_clocks.txt"

  "$EXE" > "$RUN_DIR/program_stdout.txt"

  log_gpu_state > "$RUN_DIR/post_run_gpu_state.txt"

  MEASURED_CUDA_TIME_MS="$(extract_result measured_cuda_time_ms "$RUN_DIR/program_stdout.txt")"
  MEASURED_CUDA_TIME_S="$(extract_result measured_cuda_time_s "$RUN_DIR/program_stdout.txt")"
  MEASURED_ENERGY_MJ="$(extract_result measured_energy_mj "$RUN_DIR/program_stdout.txt")"
  MEASURED_ENERGY_J="$(extract_result measured_energy_j "$RUN_DIR/program_stdout.txt")"
  AVG_POWER_W="$(extract_result average_power_w "$RUN_DIR/program_stdout.txt")"
  WARMUP_LAUNCHES="$(extract_result warmup_launches "$RUN_DIR/program_stdout.txt")"
  MEASURED_LAUNCHES="$(extract_result measured_launches "$RUN_DIR/program_stdout.txt")"

  APPLIED_CLOCKS="$(cat "$RUN_DIR/locked_clocks.txt")"
  PRE_GPU_STATE="$(cat "$RUN_DIR/pre_run_gpu_state.txt")"
  POST_GPU_STATE="$(cat "$RUN_DIR/post_run_gpu_state.txt")"

  {
    echo "workload=${WORKLOAD_PREFIX}_${KERNEL}_repeat"
    echo "kernel=$KERNEL"
    echo "gpu_name=$GPU_NAME"
    echo "driver_version=$DRIVER_VERSION"
    echo "requested_core_clock=$CORE_REQ"
    echo "requested_mem_clock=$MEM_REQ"
    echo "applied_clocks=$APPLIED_CLOCKS"
    echo "repeat=$REP"
    echo "warmup_seconds=25"
    echo "measure_seconds=5"
    echo "warmup_launches=$WARMUP_LAUNCHES"
    echo "measured_launches=$MEASURED_LAUNCHES"
    echo "measured_cuda_time_ms=$MEASURED_CUDA_TIME_MS"
    echo "measured_cuda_time_s=$MEASURED_CUDA_TIME_S"
    echo "measured_energy_mj=$MEASURED_ENERGY_MJ"
    echo "measured_energy_j=$MEASURED_ENERGY_J"
    echo "average_power_w=$AVG_POWER_W"
    echo "pre_run_gpu_state=$PRE_GPU_STATE"
    echo "post_run_gpu_state=$POST_GPU_STATE"
  } > "$RUN_DIR/summary.log"

  reset_clocks
  sleep 1
}

for idx in "${!KERNEL_NAMES[@]}"; do
  KERNEL="${KERNEL_NAMES[$idx]}"
  EXE="${KERNEL_EXES[$idx]}"

  for REP in $(seq 1 "$REPEATS"); do
    run_case "$KERNEL" "$EXE" auto auto "$REP"

    for CORE in "${CORE_CLOCKS[@]}"; do
      run_case "$KERNEL" "$EXE" "$CORE" auto "$REP"
    done

    for MEM in "${MEM_CLOCKS[@]}"; do
      run_case "$KERNEL" "$EXE" auto "$MEM" "$REP"
    done
  done
done

reset_clocks
echo "Done."
