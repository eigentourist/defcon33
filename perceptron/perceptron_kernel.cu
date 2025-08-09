#include "perceptron_kernel.cuh"

extern "C"
__global__ void perceptron_forward_kernel(const float* __restrict__ inputs,
                                          const float* __restrict__ weights,
                                          float bias,
                                          int* __restrict__ outputs,
                                          int num_samples,
                                          int num_inputs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // one thread per sample
    if (i >= num_samples) return;

    const float* x = inputs + i * num_inputs;
    float sum = bias;
    #pragma unroll
    for (int j = 0; j < num_inputs; ++j) {
        sum += x[j] * weights[j];
    }
    outputs[i] = (sum >= 0.0f) ? 1 : 0;
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

