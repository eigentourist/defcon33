#!/usr/bin/env bash
set -euo pipefail

# GPU arch: RTX 4000 = 75 (override by running: ARCH=86 ./makerun.sh)
ARCH="${ARCH:-75}"

# Build both (Makefile already skips CPU if the file isn’t present)
make ARCH="$ARCH"

# Run CPU version (will fail only if you *do* have cpu target and it errors)
make run_cpu || true

# Run CUDA version with each gate
make run_and
make run_or
make run_nand

# NOTE:
# If your file names differ, run this manually at the prompt (do NOT keep in script):
#   make CPU_SRC=perceptron_cpu.c CUDA_HOST=perceptron_cuda_host.cpp CUDA_KERN=perceptron_kernel.cu ARCH=75

