# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

easy_engine is a CUDA kernel optimization workbench for 2D convolution (3x3, low-channel: 8-64 channels). The goal is to iteratively write and benchmark hand-tuned CUDA conv2d kernels, comparing against a CUTLASS reference implementation on an NVIDIA RTX A6000 (Ampere SM 8.6).

## Build and Run

```bash
cd build && make > compile.log 2>&1       # compile (redirect output to avoid flooding context)
tail -n 2 build/compile.log               # verify 100% build success
cd build && ./engine > run.log 2>&1       # run benchmarks
tail -n1 build/run.log | grep -oP 'Time: \K[0-9.]+(?= ms)'  # extract latest kernel time
```

The build system is CMake (already configured in `build/`). Targets: `engine` (main benchmark) and `cutlass_example`. CUDA arch is SM 86, C++17/CUDA 17.

To reconfigure from scratch: `mkdir -p build && cd build && cmake .. && make`

## Architecture

- **`main.cu`** — Entry point. Includes all kernels via `#include`, runs unit tests (CPU reference vs GPU), profiles each kernel, prints timing/TFLOPS. The last printed line is always the latest experimental kernel's result.
- **`kernels/`** — Read-only baseline kernels:
  - `cpu.cu` — CPU reference implementation for correctness verification
  - `0_cutlass.cu` — CUTLASS implicit GEMM conv2d (performance reference, uses TF32)
  - `1_naiveconv.cu` — Naive conv2d baseline (one thread per output element)
- **`automatic_kernels/`** — Where new experimental kernels go. Files are named `N_description.cu` with incrementing experiment numbers.
- **`thirdparty/cutlass/`** — CUTLASS v3.5.1 (git submodule, do not modify)

## Key Conventions

- **Memory layout**: Input is `[C_in, H, W]`, filter is `[C_out, C_in, 3, 3]`, output is `[C_out, out_H, out_W]` where `out_H = H-2`, `out_W = W-2` (no padding, stride 1).
- **Kernel interface**: Every kernel must provide a `launch_*` function with signature `void(T*, T const*, T const*, size_t C_in, size_t C_out, size_t H, size_t W, cudaStream_t)`.
- **Correctness**: New kernels are validated against the CPU reference via `verify_conv2d_implementation` (tolerance 1e-3). Unit tests must pass before profiling.
- **Do not modify** files in `kernels/` or `thirdparty/`.
- **Results logging**: `results.tsv` (tab-separated) tracks experiments. It is `.gitignore`d — do not commit it.

## Experiment Workflow

See `task.md` for the full autonomous experiment loop. In brief: write kernel in `automatic_kernels/`, wire it into `main.cu`, commit, build, run, log results to `results.tsv`, keep improvements or `git reset` regressions.
