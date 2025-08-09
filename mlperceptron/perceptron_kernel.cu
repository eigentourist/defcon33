#include "perceptron_kernel.cuh"
#include <math.h>

extern "C"
__global__ void perceptron_forward_kernel(const float* __restrict__ inputs,
                                          const float* __restrict__ weights,
                                          float bias,
                                          int* __restrict__ outputs,
                                          int num_samples,
                                          int num_inputs) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= num_samples) return;

    const float* x = inputs + sample * num_inputs;
    float sum = bias;
    #pragma unroll
    for (int i = 0; i < num_inputs; ++i) sum += x[i] * weights[i];

    outputs[sample] = (sum >= 0.0f) ? 1 : 0;
}

extern "C"
void perceptron_forward_launch(const float* d_inputs,
                               const float* d_weights,
                               float bias,
                               int* d_outputs,
                               int num_samples,
                               int num_inputs,
                               cudaStream_t stream) {
    int block = 256;
    int grid  = (num_samples + block - 1) / block;
    perceptron_forward_kernel<<<grid, block, 0, stream>>>(
        d_inputs, d_weights, bias, d_outputs, num_samples, num_inputs
    );
}

