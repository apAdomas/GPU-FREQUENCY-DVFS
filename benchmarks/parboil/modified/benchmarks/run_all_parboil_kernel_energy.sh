#!/usr/bin/env bash
set -euo pipefail

MASTER_STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="$HOME/thesis/results/parboil_master_runs/$MASTER_STAMP"

mkdir -p "$MASTER_LOG_DIR"

echo "Parboil master run started at: $(date)"
echo "Logs: $MASTER_LOG_DIR"
echo

RUNNERS=(
  "$HOME/parboil_2.5/benchmarks/histo/src/cuda/run_histo_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/lbm/src/cuda/run_lbm_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/mri-q/src/cuda/run_mriq_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/sgemm/src/cuda/run_sgemm_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/spmv/src/cuda/run_spmv_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/stencil/src/cuda/run_stencil_kernel_energy.sh"
  "$HOME/parboil_2.5/benchmarks/tpacf/src/cuda/run_tpacf_kernel_energy.sh"
)

FAILED_RUNNERS=()
SKIPPED_RUNNERS=()

run_one() {
  local runner="$1"
  local name
  name="$(basename "$runner" .sh)"

  local log_file="$MASTER_LOG_DIR/${name}.log"

  echo "============================================================"
  echo "Running: $runner"
  echo "Log:     $log_file"
  echo "Start:   $(date)"
  echo "============================================================"

  if [ ! -f "$runner" ]; then
    echo "SKIP: runner does not exist: $runner" | tee "$log_file"
    SKIPPED_RUNNERS+=("$name (missing file)")
    echo
    return 0
  fi

  if [ ! -x "$runner" ]; then
    chmod +x "$runner"
  fi

  set +e
  "$runner" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -ne 0 ]; then
    echo "FAILED: $runner with exit code $status (continuing to next runner)"
    echo "See log: $log_file"
    FAILED_RUNNERS+=("$name (exit $status)")
    echo
    return 0
  fi

  echo "Finished: $runner"
  echo "End:      $(date)"
  echo
}

for runner in "${RUNNERS[@]}"; do
  run_one "$runner"
done

echo "============================================================"
echo "Parboil master run finished at: $(date)"
echo "Logs saved in: $MASTER_LOG_DIR"
echo

if [ "${#SKIPPED_RUNNERS[@]}" -gt 0 ]; then
  echo "Skipped runners:"
  printf '  - %s\n' "${SKIPPED_RUNNERS[@]}"
  echo
fi

if [ "${#FAILED_RUNNERS[@]}" -gt 0 ]; then
  echo "Failed runners (others may have completed):"
  printf '  - %s\n' "${FAILED_RUNNERS[@]}"
  echo
  exit 1
fi

echo "All Parboil kernel-energy runners completed successfully."
echo "============================================================"
