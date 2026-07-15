# GPU Kernel DVFS

Code and results for a thesis on fine-grained GPU dynamic voltage and frequency scaling (DVFS) at kernel granularity.

## What's here

- benchmarks/ — modified CUDA kernels and measurement scripts (apply to upstream PolyBench / Rodinia / Parboil)
- scripts/ — measurement oracle, ML pipeline, evaluation, plots
- results/master/ — aggregated measurements and model outputs
- figures/ — paper figures
- tables/ — LaTeX tables

## Setup

Install dependencies: pip install -r requirements.txt

Benchmark trees are not included. Copy files from benchmarks/*/modified/ into your local PolyBench-GPU, Rodinia 3.1, and Parboil installs.

## Hardware

NVIDIA GPU with nvidia-smi clock locking. Tested on RTX 3080 Ti (Ampere).

## Citation

If you use this work, please cite the accompanying thesis.
