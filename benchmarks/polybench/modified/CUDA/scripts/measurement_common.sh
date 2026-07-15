#!/usr/bin/env bash
set -euo pipefail

lock_clocks() {
  local core="$1"
  local mem="$2"
  sudo nvidia-smi --lock-gpu-clocks="${core},${core}"
  sudo nvidia-smi --lock-memory-clocks="${mem},${mem}"
}

lock_core_clock() {
  local core="$1"
  sudo nvidia-smi -lgc "$core","$core" >/dev/null
}

lock_mem_clock() {
  local mem="$1"
  sudo nvidia-smi -lmc "$mem","$mem" >/dev/null
}

reset_clocks() {
  sudo nvidia-smi --reset-gpu-clocks || true
  sudo nvidia-smi --reset-memory-clocks || true
}

log_gpu_state() {
  nvidia-smi \
    --query-gpu=temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,utilization.gpu \
    --format=csv,noheader,nounits
}

run_global_warmup() {
  local exe="$1"
  local core="$2"
  local mem="$3"

  echo "=== global GPU warmup at core=$core mem=$mem ==="
  lock_clocks "$core" "$mem"
  sleep 1

  echo "before warmup: $(log_gpu_state)"
  "$exe"
  echo "after warmup:  $(log_gpu_state)"

  reset_clocks
}