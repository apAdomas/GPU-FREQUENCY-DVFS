#!/usr/bin/env bash
set -euo pipefail

CORE_CLOCKS=(800 1200 1600)
MEM_CLOCKS=(9501 5001 810)
RUNS=3

WORKLOADS=(
  "reduction|$HOME/cuda-samples/build/Samples/2_Concepts_and_Techniques/reduction|./reduction"
  "transpose|$HOME/cuda-samples/build/Samples/6_Performance/transpose|./transpose"

  "gemm|$HOME/polybenchGpu/CUDA/GEMM|./gemm.exe"
  "2mm|$HOME/polybenchGpu/CUDA/2MM|./2mm.exe"
  "bicg|$HOME/polybenchGpu/CUDA/BICG|./bicg.exe"
  "jacobi2d|$HOME/polybenchGpu/CUDA/JACOBI2D|./jacobi2d.exe"
  "2dconv|$HOME/polybenchGpu/CUDA/2DCONV|./2DConvolution.exe"
)

mkdir -p "$HOME/thesis/results"

sudo nvidia-smi -pm 1 >/dev/null || true

for W in "${WORKLOADS[@]}"; do
  IFS='|' read -r NAME DIR EXE <<< "$W"

  if [ ! -d "$DIR" ]; then
    echo "Missing directory: $DIR"
    continue
  fi

  if [ ! -f "$DIR/${EXE#./}" ]; then
    echo "Missing executable: $DIR/${EXE#./}"
    continue
  fi

  for CORE in "${CORE_CLOCKS[@]}"; do
    for MEM in "${MEM_CLOCKS[@]}"; do
      echo "=== $NAME core=$CORE mem=$MEM ==="

      sudo nvidia-smi --lock-gpu-clocks="${CORE},${CORE}"
      sudo nvidia-smi --lock-memory-clocks="${MEM},${MEM}"
      sleep 1

      cd "$DIR"

      # warmup
      $EXE >/dev/null 2>&1 || true

      for i in $(seq 1 "$RUNS"); do
        LOGFILE="$HOME/thesis/results/${NAME}_c${CORE}_m${MEM}_run${i}.log"
        {
          echo "workload=$NAME core=$CORE mem=$MEM run=$i"
          $EXE
        } 2>&1 | tee "$LOGFILE"
      done

      sudo nvidia-smi --reset-gpu-clocks || true
      sudo nvidia-smi --reset-memory-clocks || true
    done
  done
done

sudo nvidia-smi --reset-gpu-clocks || true
sudo nvidia-smi --reset-memory-clocks || true