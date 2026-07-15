#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="$HOME/thesis/results"

RESULT_DIRS=(
  "2dconv_kernel_energy"
  "2mm_kernel_energy"
  "3mm_kernel_energy"
  "atax_kernel_energy"
  "bicg_kernel_energy"
  "correlation_kernel_energy"
  "covariance_kernel_energy"
  "doitgen_kernel_energy"
  "fdtd2d_kernel_energy"
  "gemm_kernel_energy"
  "gesummv_kernel_energy"
  "gramschmidt_kernel_energy"
  "jacobi1d_kernel_energy"
  "jacobi2d_kernel_energy"
  "mvt_kernel_energy"
  "syr2k_kernel_energy"
  "syrk_kernel_energy"
)

GEMVER_DIR="gemver_kernel_energy"

MERGED_STAMP="$(date +%Y%m%d_%H%M%S)"
MERGED_SUFFIX="_merged"

copy_run_contents() {
  local src_run="$1"
  local dst_run="$2"

  mkdir -p "$dst_run"

  find "$src_run" -type f | while read -r src_file; do
    local rel
    rel="${src_file#$src_run/}"
    local dst_file="$dst_run/$rel"
    local dst_parent
    dst_parent="$(dirname "$dst_file")"

    mkdir -p "$dst_parent"

    if [[ -e "$dst_file" ]]; then
      echo "SKIP duplicate: $dst_file"
      continue
    fi

    cp -p "$src_file" "$dst_file"
  done
}

merge_last_two_runs() {
  local results_name="$1"
  local results_dir="$RESULT_ROOT/$results_name"

  if [[ ! -d "$results_dir" ]]; then
    echo "SKIP missing dir: $results_dir"
    return
  fi

  mapfile -t runs < <(
    find "$results_dir" -mindepth 1 -maxdepth 1 -type d \
      ! -name "*${MERGED_SUFFIX}" \
      -printf "%f\n" | sort
  )

  local n="${#runs[@]}"
  if (( n == 0 )); then
    echo "SKIP no run dirs: $results_dir"
    return
  fi

  if (( n == 1 )); then
    echo "ONLY one run found for $results_name, copying latest only"
    local latest="${runs[$((n-1))]}"
    local out_dir="$results_dir/${MERGED_STAMP}${MERGED_SUFFIX}"
    mkdir -p "$out_dir"
    copy_run_contents "$results_dir/$latest" "$out_dir"
    echo "WROTE merged dir: $out_dir"
    return
  fi

  local prev="${runs[$((n-2))]}"
  local latest="${runs[$((n-1))]}"
  local out_dir="$results_dir/${MERGED_STAMP}${MERGED_SUFFIX}"

  echo
  echo "=================================================="
  echo "MERGING: $results_name"
  echo "  prev:   $prev"
  echo "  latest: $latest"
  echo "  out:    $out_dir"
  echo "=================================================="

  mkdir -p "$out_dir"

  copy_run_contents "$results_dir/$prev" "$out_dir"
  copy_run_contents "$results_dir/$latest" "$out_dir"

  echo "WROTE merged dir: $out_dir"
}

copy_latest_only() {
  local results_name="$1"
  local results_dir="$RESULT_ROOT/$results_name"

  if [[ ! -d "$results_dir" ]]; then
    echo "SKIP missing dir: $results_dir"
    return
  fi

  mapfile -t runs < <(
    find "$results_dir" -mindepth 1 -maxdepth 1 -type d \
      ! -name "*${MERGED_SUFFIX}" \
      -printf "%f\n" | sort
  )

  local n="${#runs[@]}"
  if (( n == 0 )); then
    echo "SKIP no run dirs: $results_dir"
    return
  fi

  local latest="${runs[$((n-1))]}"
  local out_dir="$results_dir/${MERGED_STAMP}${MERGED_SUFFIX}"

  echo
  echo "=================================================="
  echo "COPYING latest only: $results_name"
  echo "  latest: $latest"
  echo "  out:    $out_dir"
  echo "=================================================="

  mkdir -p "$out_dir"
  copy_run_contents "$results_dir/$latest" "$out_dir"

  echo "WROTE merged dir: $out_dir"
}

for name in "${RESULT_DIRS[@]}"; do
  merge_last_two_runs "$name"
done

copy_latest_only "$GEMVER_DIR"

echo
echo "Done."
echo "Merged timestamp suffix used: ${MERGED_STAMP}${MERGED_SUFFIX}"
