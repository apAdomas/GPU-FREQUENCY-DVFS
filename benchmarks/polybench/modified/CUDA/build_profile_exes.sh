#!/usr/bin/env bash
#
# Build <stem>_profile.exe for each NN_*_repeat.cu under $HOME/polybenchGpu/CUDA.
#   -DNCU_PROFILE -DWARMUP_SECONDS=1 -DMEASURE_SECONDS=0.1
# Use with: ncu --profile-from-start off --launch-count 1
#
set -uo pipefail

ROOT="$HOME/polybenchGpu/CUDA"
SCRIPTS_INC="$ROOT/scripts"

ARCH="${ARCH:-sm_86}"
WARMUP="${WARMUP_SECONDS:-1}"
MEASURE="${MEASURE_SECONDS:-0.1}"

ok=0
fail=0
failed_list=()

while IFS= read -r cu; do
    base="$(basename "$cu" .cu)"

    # Only isolated kernel files (numeric prefix), mirroring the NCU script.
    case "$base" in
        [0-9]*_repeat) ;;
        *) continue ;;
    esac

    dir="$(dirname "$cu")"
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
