#pragma once
#include <cuda_runtime.h>

extern "C" __global__
void perceptron_forward_kernel(const float* __restrict__ inputs,  // [num_samples * num_inputs]
                               const float* __restrict__ weights, // [num_inputs]
                               float bias,
                               int* __restrict__ outputs,         // [num_samples]
                               int num_samples,
                               int num_inputs);

// Convenience launcher (1D grid over samples)
extern "C" void perceptron_forward_launch(const float* d_inputs,
                                          const float* d_weights,
                                          float bias,
                                          int* d_outputs,
                                          int num_samples,
                                          int num_inputs,
                                          cudaStream_t stream = nullptr);

