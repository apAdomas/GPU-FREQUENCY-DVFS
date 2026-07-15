#!/usr/bin/env bash
set -u

BASE="$HOME/polybenchGpu/CUDA"
OUT="$HOME/thesis/results/polybench/systems"

BENCHES=(
  2DCONV 3DCONV ADI BICG COVAR FDTD-2D GEMM GESUMMV GRAMSCHM
  JACOBI1D JACOBI2D LU MVT SYR2K SYRK 2MM 3MM ATAX CORR DOITGEN GEMVER
)

mkdir -p "$OUT"

for b in "${BENCHES[@]}"; do
  echo "=== $b ==="

  DIR="$BASE/$b"
  if [ ! -d "$DIR" ]; then
    echo "Missing dir: $DIR"
    continue
  fi

  cd "$DIR" || continue

  EXE="$(find . -maxdepth 1 -type f -name '*.exe' | head -n 1)"
  if [ -z "$EXE" ]; then
    echo "No .exe found in $DIR, skipping"
    continue
  fi

  NAME="$(basename "$EXE" .exe)"
  REP="$OUT/${b}_nsys"
  TXT="$OUT/${b}_nsys_stats.txt"

  echo "Running nsys on $EXE"

  nsys profile -o "$REP" "$EXE" >/dev/null 2>&1
  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "nsys failed for $b, skipping"
    continue
  fi

  nsys stats --report cuda_api_gpu_sum,cuda_gpu_kern_sum "${REP}.nsys-rep" > "$TXT" 2>&1
  STATUS=$?

  if [ $STATUS -ne 0 ]; then
    echo "nsys stats failed for $b, skipping"
    continue
  fi

  echo "Saved:"
  echo "  ${REP}.nsys-rep"
  echo "  $TXT"
done