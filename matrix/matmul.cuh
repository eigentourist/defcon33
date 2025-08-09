#pragma once
#include <cuda_runtime.h>

// Naïve: one thread per (row,col)
extern "C" __global__
void matmul_naive_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N);

// Tiled: shared-memory 16x16
extern "C" __global__
void matmul_tiled_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N);

// Convenience launchers
extern "C" void matmul_naive_launch(const float* dA, const float* dB, float* dC, int N,
                                    cudaStream_t stream = nullptr);
extern "C" void matmul_tiled_launch(const float* dA, const float* dB, float* dC, int N,
                                    cudaStream_t stream = nullptr);

