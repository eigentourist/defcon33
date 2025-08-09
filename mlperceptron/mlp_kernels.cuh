#pragma once
#include <cuda_runtime.h>

extern "C" __global__
void forward_hidden_kernel(const float* __restrict__ inputs,
                           const float* __restrict__ wh,
                           const float* __restrict__ bh,
                           float* __restrict__ hidden,
                           int num_samples, int num_inputs, int num_hidden);

extern "C" __global__
void forward_output_kernel(const float* __restrict__ hidden,
                           const float* __restrict__ wo,
                           const float* __restrict__ bo,
                           float* __restrict__ output,
                           int num_samples, int num_hidden, int num_outputs);

// Convenience launchers (grid/block picked here)
extern "C" void mlp_forward_hidden_launch(const float* d_inputs,
                                          const float* d_wh,
                                          const float* d_bh,
                                          float* d_hidden,
                                          int num_samples, int num_inputs, int num_hidden,
                                          cudaStream_t stream = nullptr);

extern "C" void mlp_forward_output_launch(const float* d_hidden,
                                          const float* d_wo,
                                          const float* d_bo,
                                          float* d_output,
                                          int num_samples, int num_hidden, int num_outputs,
                                          cudaStream_t stream = nullptr);

