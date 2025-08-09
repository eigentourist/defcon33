#pragma once
#include <cuda_runtime.h>

extern "C" __global__
void life_step_kernel(const int* __restrict__ in,
                      int* __restrict__ out,
                      int width, int height);

// Host-callable wrapper (implemented in .cu)
extern "C" void life_step_launch(const int* d_in, int* d_out, int width, int height);

