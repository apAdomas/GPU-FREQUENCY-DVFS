#!/usr/bin/env bash
#
# build_profile_exes.sh
#
# Compiles NCU profiling executables (<name>_profile.exe) for every
# *_repeat.cu kernel under rodinia_3.1/cuda.
#
# Profiling build:
#   -DNCU_PROFILE      activates cudaProfilerStart()/cudaProfilerStop() markers
#   -DWARMUP_SECONDS=1 short warmup (NCU only needs one captured launch)
#   -DMEASURE_SECONDS=0.1
#
# Pair with: ncu --profile-from-start off --launch-count 1 ...
#
set -uo pipefail

ROOT="$HOME/rodinia_3.1/cuda"
SCRIPTS_INC="$HOME/polybenchGpu/CUDA/scripts"

ARCH="${ARCH:-sm_86}"
WARMUP="${WARMUP_SECONDS:-1}"
MEASURE="${MEASURE_SECONDS:-0.1}"

ok=0
fail=0
failed_list=()

while IFS= read -r cu; do
    dir="$(dirname "$cu")"
    base="$(basename "$cu" .cu)"
    out="$dir/${base}_profile.exe"

    echo "=== compiling ${base}_profile.exe ==="
    if nvcc -O3 -arch="$ARCH" \
        -DNCU_PROFILE -DWARMUP_SECONDS="$WARMUP" -DMEASURE_SECONDS="$MEASURE" \
        -I"$SCRIPTS_INC" -I"$dir" \
        "$cu" -o "$out" -lnvidia-ml; then
        echo "OK   $out"
        ok=$((ok + 1))
    else
        echo "FAILED $cu"
        fail=$((fail + 1))
        failed_list+=("$cu")
    fi
done < <(find "$ROOT" -name '*_repeat.cu' | sort)

echo
echo "Built OK: $ok    Failed: $fail"
if [ "$fail" -gt 0 ]; then
    printf 'FAILED: %s\n' "${failed_list[@]}"
    exit 1
fi
