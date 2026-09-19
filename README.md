# GPU Kernel DVFS

Code and results for *Predicting GPU Frequencies to Reduce Energy Waste*. GPU workloads often run at clocks that add little performance while using more energy. DVFS can reduce that waste by lowering the core and memory frequency, but kernels respond differently, so the useful setting depends on the kernel. This repo isolates each kernel, measures time and energy across a clock grid, builds oracles under a runtime slowdown bound, and trains models to predict clock settings from PTX, launch configuration, and Nsight Compute features.

## Layout

```
Predicting_GPU_Frequencies_to_Reduce_Energy_Waste.pdf
benchmarks/     modified / added files only
scripts/
  oracle/         measured energy oracles from the clock sweeps
  prediction/     features through predicted clock settings
results/master/ committed aggregates and model outputs
figures/
tables/
```

## Benchmarks

This repo does not ship original PolyBench / Rodinia / Parboil sources. `benchmarks/` is only the files that were added or changed.

Each app is split into one file per measured kernel:

```
01_<kernel>_repeat.cu   ->  01_<kernel>_repeat.exe
```

The prefix is launch order. Shared harness (`measurement_common.h` / `.sh`): allocate once, 25s warmup, 5s measure, CUDA events + NVML, print `RESULT key=value`. Clocks are locked by the runner. Warmup is `global_warmup_fdtd_sequence.cu`.

Per-app runner: `run_*_kernel_energy.sh`. One `summary.log` per kernel x clock case.

## Setup

```
pip install -r requirements.txt
```

NVIDIA GPU with `nvidia-smi` clock lock (sudo). Measured on RTX 3080 Ti (`sm_86`). `nvcc` and `ncu` on PATH.

```
nvcc -O3 -arch=sm_86 -I../scripts 01_<kernel>_repeat.cu -o 01_<kernel>_repeat.exe -lnvidia-ml
nvcc -O3 -arch=sm_86 -I. global_warmup_fdtd_sequence.cu -o global_warmup_fdtd_sequence.exe
```

NCU builds (`*_profile.exe`, `-DNCU_PROFILE`):

```
benchmarks/polybench/modified/CUDA/build_profile_exes.sh
benchmarks/rodinia/modified/cuda/build_profile_exes.sh
```

## Pipeline

1. **Measure.** `run_*_kernel_energy.sh` for each app.
2. **Oracle, per workload.** Latest run under `<workload>_kernel_energy/`:

```
python3 scripts/oracle/1_merge_kernel_logs.py    <results_dir>   # -> merged_summary.csv
python3 scripts/oracle/2_aggregate_kernel_logs.py <results_dir>   # -> aggregated_by_kernel_clock.csv
```

`3_find_kernel_oracle.py` writes a per-workload oracle: lowest measured energy clock per kernel under a slowdown bound. Skip it if you only want the ML path. Train and eval build their oracles from the joined dataset, not from this file.

1. **Master merge.** `4_merge_all_workloads.py` -> `aggregated_all_kernels.csv`
2. **Collapse clocks.** `1_collapse_effective_clock_pairs.py` merges requested pairs that applied to the same (core, mem). -> `aggregated_all_kernels_effective.csv`
3. **Features.** `0_generate_ptx_files.py`, `1_extract_ptx_features.py`, `2_extract_launch_features.py`, `3_profile_ncu_features.py`
4. **Dataset.** `2_build_ptx_launch_runtime_dataset.py` joins features and adds time/energy ratios vs auto.
5. **Train.** `1_train_ptx_launch_runtime_loko.py`. Leave-one-kernel-out. RF / ExtraTrees / XGBoost on `ptx`, `launch`, `runtime`, and combinations.
6. **Evaluate.** `global_oracle_evaluate_measured.py`, then predicted local and global policy (`1_`, `2_`). Bounds 0 / 1 / 2 / 5% vs auto.
7. **Plot.** `scripts/prediction/5_plot/{3,6,7,8}_*.py`

