#!/usr/bin/env bash
set -u
set -o pipefail

RUNNERS=(
  # "$HOME/polybenchGpu/CUDA/2DCONV/run_2dconv_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/2MM/run_2mm_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/3MM/run_3mm_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/ATAX/run_atax_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/BICG/run_bicg_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/CORR/run_correlation_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/COVAR/run_covariance_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/DOITGEN/run_doitgen_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/FDTD-2D/run_fdtd2d_kernel_energy.sh"
  # "$HOME/polybenchGpu/CUDA/GEMM/run_gemm_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/GEMVER/run_gemver_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/GESUMMV/run_gesummv_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/GRAMSCHMIDT/run_gramschmidt_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/JACOBI1D/run_jacobi1d_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/JACOBI2D/run_jacobi2d_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/MVT/run_mvt_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/SYR2K/run_syr2k_kernel_energy.sh"
  "$HOME/polybenchGpu/CUDA/SYRK/run_syrk_kernel_energy.sh"
)

for runner in "${RUNNERS[@]}"; do
  echo
  echo "=================================================="
  echo "RUNNING: $runner"
  echo "=================================================="

  if [ ! -f "$runner" ]; then
    echo "SKIP: file not found: $runner"
    continue
  fi

  if [ ! -x "$runner" ]; then
    chmod +x "$runner" || {
      echo "SKIP: could not make executable: $runner"
      continue
    }
  fi

  bash "$runner"
  status=$?

  if [ $status -ne 0 ]; then
    echo "FAILED: $runner (exit code $status)"
    echo "Continuing to next runner..."
  else
    echo "DONE: $runner"
  fi
done

echo
echo "All requested runners attempted."
