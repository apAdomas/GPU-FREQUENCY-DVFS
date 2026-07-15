#!/usr/bin/env bash
set -euo pipefail

MERGE_SCRIPT="$HOME/thesis/scripts/oracle/1_merge_kernel_logs.py"
AGG_SCRIPT="$HOME/thesis/scripts/oracle/2_aggregate_kernel_logs.py"
ORACLE_SCRIPT="$HOME/thesis/scripts/oracle/3_find_kernel_oracle.py"
MERGE_ALL_SCRIPT="$HOME/thesis/scripts/oracle/4_merge_all_workloads.py"

RESULT_DIRS=(
  "$HOME/thesis/results/2dconv_kernel_energy"
  "$HOME/thesis/results/2mm_kernel_energy"
  "$HOME/thesis/results/3mm_kernel_energy"
  "$HOME/thesis/results/atax_kernel_energy"
  "$HOME/thesis/results/bicg_kernel_energy"
  "$HOME/thesis/results/correlation_kernel_energy"
  "$HOME/thesis/results/covariance_kernel_energy"
  "$HOME/thesis/results/doitgen_kernel_energy"
  "$HOME/thesis/results/fdtd2d_kernel_energy"
  "$HOME/thesis/results/gemm_kernel_energy"
  "$HOME/thesis/results/gemver_kernel_energy"
  "$HOME/thesis/results/gesummv_kernel_energy"
  "$HOME/thesis/results/gramschmidt_kernel_energy"
  "$HOME/thesis/results/jacobi1d_kernel_energy"
  "$HOME/thesis/results/jacobi2d_kernel_energy"
  "$HOME/thesis/results/mvt_kernel_energy"
  "$HOME/thesis/results/syr2k_kernel_energy"
  "$HOME/thesis/results/syrk_kernel_energy"
)

for results_dir in "${RESULT_DIRS[@]}"; do
  echo
  echo "=================================================="
  echo "PROCESSING: $results_dir"
  echo "=================================================="

  if [ ! -d "$results_dir" ]; then
    echo "SKIP: missing dir $results_dir"
    continue
  fi

  python3 "$MERGE_SCRIPT" "$results_dir"
  python3 "$AGG_SCRIPT" "$results_dir"
  python3 "$ORACLE_SCRIPT" "$results_dir"

  echo "DONE: $results_dir"
done

python3 "$MERGE_ALL_SCRIPT"

echo
echo "All result folders processed and master files updated."
