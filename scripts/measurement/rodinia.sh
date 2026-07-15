#!/usr/bin/env bash
set -u

BASE="$HOME/rodinia_3.1/cuda"
DATA="$HOME/rodinia_3.1/data"
OUT="$HOME/thesis/results/rodinia/nsys"
mkdir -p "$OUT"

run_one () {
  name="$1"
  shift
  echo "=== $name ==="

  rep="$OUT/${name}"
  rm -f "${rep}.nsys-rep" "${rep}.sqlite" "${rep}_kernels.txt" "${rep}_api.txt"

  nsys profile -o "$rep" "$@" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "nsys failed: $name"
    return
  fi

  nsys stats --force-export=true --report cuda_gpu_kern_sum "${rep}.nsys-rep" > "${rep}_kernels.txt" 2>&1
  nsys stats --force-export=true --report cuda_api_sum "${rep}.nsys-rep" > "${rep}_api.txt" 2>&1

  echo "saved: ${rep}.nsys-rep"
  echo "saved: ${rep}_kernels.txt"
  echo "saved: ${rep}_api.txt"
}

run_one hotspot \
  "$BASE/hotspot/hotspot" \
  512 2 2 \
  "$DATA/hotspot/temp_512" \
  "$DATA/hotspot/power_512" \
  output.out \
  output.out

run_one hotspot3D \
  "$BASE/hotspot3D/3D" \
  512 8 100 \
  "$DATA/hotspot3D/power_512x8" \
  "$DATA/hotspot3D/temp_512x8" \
  output.out

run_one bfs \
  "$BASE/bfs/bfs" \
  "$DATA/bfs/graph65536.txt"

run_one nw \
  "$BASE/nw/needle" \
  2048 10

run_one pathfinder \
  "$BASE/pathfinder/pathfinder" \
  100000 100 20 >/dev/null 2>&1