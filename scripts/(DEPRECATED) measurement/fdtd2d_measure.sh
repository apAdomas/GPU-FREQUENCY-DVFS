#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$HOME/polybenchGpu/CUDA/FDTD-2D"
EXE="./fdtd2d.exe"
OUTDIR="$HOME/thesis/results/fdtd2d"

CORE_CLOCKS=(800 1200 1600)
MEM_CLOCKS=(810 5001 9501)

GPU_WARMUP_SECONDS=60
KERNEL_WARMUP_SECONDS=10
MEASURE_RUNS=3
REPEATS=3

PYTHON_NVML="$HOME/thesis/.venv/bin/python"
NVML_SCRIPT="$HOME/thesis/scripts/nvml_energy.py"

mkdir -p "$OUTDIR"
cd "$WORKDIR"

sudo nvidia-smi -pm 1 >/dev/null || true

echo "=== GPU warmup at max/default clocks for ${GPU_WARMUP_SECONDS}s ==="
sudo nvidia-smi --reset-gpu-clocks || true
sudo nvidia-smi --reset-memory-clocks || true

END=$((SECONDS + GPU_WARMUP_SECONDS))
while [ $SECONDS -lt $END ]; do
  $EXE >/dev/null 2>&1 || true
done

for CORE in "${CORE_CLOCKS[@]}"; do
  for MEM in "${MEM_CLOCKS[@]}"; do
    for REP in $(seq 1 "$REPEATS"); do
      echo "=== core=$CORE mem=$MEM rep=$REP ==="

      sudo nvidia-smi --lock-gpu-clocks="${CORE},${CORE}"
      sudo nvidia-smi --lock-memory-clocks="${MEM},${MEM}"
      sleep 1

      echo "  warmup ${KERNEL_WARMUP_SECONDS}s"
      END=$((SECONDS + KERNEL_WARMUP_SECONDS))
      while [ $SECONDS -lt $END ]; do
        $EXE >/dev/null 2>&1 || true
      done

      LOG="$OUTDIR/fdtd2d_c${CORE}_m${MEM}_rep${REP}.log"

      echo "  measure ${MEASURE_RUNS} full runs"
      START_ENERGY=$("$PYTHON_NVML" "$NVML_SCRIPT" 2>/dev/null || echo "")
      START_TS=$(date +%s.%N)

      ITER=0
      for ((i=1; i<=MEASURE_RUNS; i++)); do
        $EXE >/dev/null 2>&1 || true
        ITER=$((ITER + 1))
      done

      END_TS=$(date +%s.%N)
      END_ENERGY=$("$PYTHON_NVML" "$NVML_SCRIPT" 2>/dev/null || echo "")

      ELAPSED=$(python3 - <<PY
st=float("$START_TS")
et=float("$END_TS")
print(et-st)
PY
)
      ENERGY=$(python3 - <<PY
se=float("$START_ENERGY")
ee=float("$END_ENERGY")
print(ee-se)
PY
)
      AVG_POWER=$(python3 - <<PY
elapsed=float("$ELAPSED")
energy_mj=float("$ENERGY")
print((energy_mj/1000.0)/elapsed)
PY
)

      {
        echo "workload=fdtd2d"
        echo "core_clock=$CORE"
        echo "mem_clock=$MEM"
        echo "repeat=$REP"
        echo "iterations=$ITER"
        echo "start_time_s=$START_TS"
        echo "end_time_s=$END_TS"
        echo "elapsed_s=$ELAPSED"
        echo "start_energy_mJ=$START_ENERGY"
        echo "end_energy_mJ=$END_ENERGY"
        echo "energy_mJ=$ENERGY"
        echo "avg_power_W=$AVG_POWER"
      } > "$LOG"

      sudo nvidia-smi --reset-gpu-clocks || true
      sudo nvidia-smi --reset-memory-clocks || true
      sleep 1
    done
  done
done

sudo nvidia-smi --reset-gpu-clocks || true
sudo nvidia-smi --reset-memory-clocks || true
echo "Done."