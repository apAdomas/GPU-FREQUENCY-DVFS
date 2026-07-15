#!/usr/bin/env bash
set -euo pipefail

MASTER_STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="$HOME/thesis/results/rodinia_master_runs/$MASTER_STAMP"

mkdir -p "$MASTER_LOG_DIR"

echo "Rodinia master run started at: $(date)"
echo "Logs: $MASTER_LOG_DIR"
echo

RUNNERS=(
  "$HOME/rodinia_3.1/cuda/hotspot/run_hotspot_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/hotspot3D/run_hotspot3d_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/kmeans/run_kmeans_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/lavaMD/run_lavamd_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/myocyte/run_myocyte_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/nn/run_nn_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/nw/run_nw_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/particlefilter/run_particlefilter_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/pathfinder/run_pathfinder_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/srad/srad_v1/run_srad_v1_kernel_energy.sh"
  "$HOME/rodinia_3.1/cuda/streamcluster/run_streamcluster_kernel_energy.sh"
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
echo "Rodinia master run finished at: $(date)"
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

echo "All Rodinia kernel-energy runners completed successfully."
echo "============================================================"
