#!/usr/bin/env bash
set -euo pipefail

source "$HOME/polybenchGpu/CUDA/scripts/measurement_common.sh"
trap 'reset_clocks' EXIT

WORKDIR="$HOME/rodinia_3.1/cuda/nw"
RUNSTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$HOME/thesis/results/nw_kernel_energy/$RUNSTAMP"

DIMENSION=4096
PENALTY=10

CORE_CLOCKS=(210 630 1050 1470 1890 2100)
MEM_CLOCKS=(405 810 5001 9251 9501)

REPEATS=1
SLEEP_AFTER_CLOCK_LOCK=1

GLOBAL_WARMUP_CORE=2100
GLOBAL_WARMUP_MEM=9501
GLOBAL_WARMUP_EXE="$HOME/polybenchGpu/CUDA/scripts/global_warmup_fdtd_sequence.exe"

KERNEL_NAMES=(
  "shared_1"
  "shared_2"
)

KERNEL_EXES=(
  "./01_nw_shared_1_repeat.exe"
  "./02_nw_shared_2_repeat.exe"
)

WORKLOAD_PREFIX="nw"

mkdir -p "$OUTDIR"
echo "Writing results to: $OUTDIR"

cd "$WORKDIR"

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

  local value
  value="$(grep "^RESULT ${key}=" "$file" | tail -1 | cut -d= -f2 || true)"

  if [ -z "$value" ]; then
    echo "ERROR: missing RESULT ${key} in $file" >&2
    echo "---- program stdout ----" >&2
    cat "$file" >&2
    return 1
  fi

  echo "$value"
}


run_case_fail() {
  local RUN_DIR="$1"
  local KERNEL="$2"
  local CORE_REQ="$3"
  local MEM_REQ="$4"
  local REP="$5"
  local REASON="$6"

  {
    echo "status=failed"
    echo "failure_reason=$REASON"
    echo "kernel=$KERNEL"
    echo "requested_core_clock=$CORE_REQ"
    echo "requested_mem_clock=$MEM_REQ"
    echo "repeat=$REP"
  } > "$RUN_DIR/summary.log"

  reset_clocks || true
  sleep 1
  return 1
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

  set +e
  "$EXE" "$DIMENSION" "$PENALTY" > "$RUN_DIR/program_stdout.txt"
  local exe_rc=$?
  set -e

  if [ "$exe_rc" -ne 0 ]; then
    run_case_fail "$RUN_DIR" "$KERNEL" "$CORE_REQ" "$MEM_REQ" "$REP" "exe_exit_${exe_rc}"
    return 1
  fi

  log_gpu_state > "$RUN_DIR/post_run_gpu_state.txt"

  set +e
  PROGRAM_KERNEL="$(extract_result kernel "$RUN_DIR/program_stdout.txt")"
  WARMUP_SECONDS_TARGET="$(extract_result warmup_seconds_target "$RUN_DIR/program_stdout.txt")"
  MEASURE_SECONDS_TARGET="$(extract_result measure_seconds_target "$RUN_DIR/program_stdout.txt")"
  WARMUP_LAUNCHES="$(extract_result warmup_launches "$RUN_DIR/program_stdout.txt")"
  MEASURED_LAUNCHES="$(extract_result measured_launches "$RUN_DIR/program_stdout.txt")"
  MEASURED_CUDA_TIME_MS="$(extract_result measured_cuda_time_ms "$RUN_DIR/program_stdout.txt")"
  MEASURED_CUDA_TIME_S="$(extract_result measured_cuda_time_s "$RUN_DIR/program_stdout.txt")"
  MEASURED_ENERGY_MJ="$(extract_result measured_energy_mj "$RUN_DIR/program_stdout.txt")"
  MEASURED_ENERGY_J="$(extract_result measured_energy_j "$RUN_DIR/program_stdout.txt")"
  AVG_POWER_W="$(extract_result average_power_w "$RUN_DIR/program_stdout.txt")"
  set -e

  if [ -z "$PROGRAM_KERNEL" ] || [ -z "$MEASURED_ENERGY_MJ" ]; then
    run_case_fail "$RUN_DIR" "$KERNEL" "$CORE_REQ" "$MEM_REQ" "$REP" "missing_result"
    return 1
  fi

  APPLIED_CLOCKS="$(cat "$RUN_DIR/locked_clocks.txt")"
  PRE_GPU_STATE="$(cat "$RUN_DIR/pre_run_gpu_state.txt")"
  POST_GPU_STATE="$(cat "$RUN_DIR/post_run_gpu_state.txt")"

  {
    echo "workload=${WORKLOAD_PREFIX}_${KERNEL}_repeat"
    echo "kernel=$KERNEL"
    echo "program_kernel=$PROGRAM_KERNEL"
    echo "dimension=$DIMENSION"
    echo "penalty=$PENALTY"
    echo "gpu_name=$GPU_NAME"
    echo "driver_version=$DRIVER_VERSION"
    echo "requested_core_clock=$CORE_REQ"
    echo "requested_mem_clock=$MEM_REQ"
    echo "applied_clocks=$APPLIED_CLOCKS"
    echo "repeat=$REP"
    echo "warmup_seconds=$WARMUP_SECONDS_TARGET"
    echo "measure_seconds=$MEASURE_SECONDS_TARGET"
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

run_case_safe() {
  run_case "$@" || {
    echo "WARNING: run_case failed for kernel=$1 core=$3 mem=$4 rep=$5 — continuing"
    reset_clocks || true
    sleep 1
  }
}

for idx in "${!KERNEL_NAMES[@]}"; do
  KERNEL="${KERNEL_NAMES[$idx]}"
  EXE="${KERNEL_EXES[$idx]}"

  if [ ! -x "$EXE" ]; then
    echo "SKIP: missing executable for kernel=$KERNEL: $EXE"
    continue
  fi

  for REP in $(seq 1 "$REPEATS"); do

    for CORE in "${CORE_CLOCKS[@]}"; do
      for MEM in "${MEM_CLOCKS[@]}"; do
        run_case_safe "$KERNEL" "$EXE" "$CORE" "$MEM" "$REP"
      done
    done

    run_case_safe "$KERNEL" "$EXE" auto auto "$REP"

    for CORE in "${CORE_CLOCKS[@]}"; do
      run_case_safe "$KERNEL" "$EXE" "$CORE" auto "$REP"
    done

    for MEM in "${MEM_CLOCKS[@]}"; do
      run_case_safe "$KERNEL" "$EXE" auto "$MEM" "$REP"
    done

  done
done

reset_clocks
echo "Done."
echo "Results written to: $OUTDIR"