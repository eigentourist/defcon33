// CUDA port of perceptron_mlp_kernel.cl
#include "mlp_kernels.cuh"
#include <cmath>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Each thread handles one sample; loops over neurons (same as your OpenCL)
extern "C"
__global__ void forward_hidden_kernel(const float* __restrict__ inputs,
                                      const float* __restrict__ wh,
                                      const float* __restrict__ bh,
                                      float* __restrict__ hidden,
                                      int num_samples, int num_inputs, int num_hidden) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= num_samples) return;

    // hidden[sample, j] = sigmoid( bh[j] + sum_i inputs[sample,i] * wh[i,j] )
    for (int j = 0; j < num_hidden; ++j) {
        float sum = bh[j];
        int in_row = sample * num_inputs;
        int w_col  = j; // wh laid out [i * num_hidden + j]
        for (int i = 0; i < num_inputs; ++i) {
            sum += inputs[in_row + i] * wh[i * num_hidden + w_col];
        }
        hidden[sample * num_hidden + j] = sigmoidf(sum);
    }
}

extern "C"
__global__ void forward_output_kernel(const float* __restrict__ hidden,
                                      const float* __restrict__ wo,
                                      const float* __restrict__ bo,
                                      float* __restrict__ output,
                                      int num_samples, int num_hidden, int num_outputs) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= num_samples) return;

    // output[sample, k] = sigmoid( bo[k] + sum_j hidden[sample,j] * wo[j,k] )
    for (int k = 0; k < num_outputs; ++k) {
        float sum = bo[k];
        int h_row = sample * num_hidden;
        int w_col = k; // wo laid out [j * num_outputs + k]
        for (int j = 0; j < num_hidden; ++j) {
            sum += hidden[h_row + j] * wo[j * num_outputs + w_col];
        }
        output[sample * num_outputs + k] = sigmoidf(sum);
    }
}

// Simple 1D launches; tune block size if you like
extern "C"
void mlp_forward_hidden_launch(const float* d_inputs,
                               const float* d_wh,
                               const float* d_bh,
                               float* d_hidden,
                               int num_samples, int num_inputs, int num_hidden,
                               cudaStream_t stream) {
    int block = 256;
    int grid  = (num_samples + block - 1) / block;
    forward_hidden_kernel<<<grid, block, 0, stream>>>(d_inputs, d_wh, d_bh, d_hidden,
                                                      num_samples, num_inputs, num_hidden);
}

extern "C"
void mlp_forward_output_launch(const float* d_hidden,
                               const float* d_wo,
                               const float* d_bo,
                               float* d_output,
                               int num_samples, int num_hidden, int num_outputs,
                               cudaStream_t stream) {
    int block = 256;
    int grid  = (num_samples + block - 1) / block;
    forward_output_kernel<<<grid, block, 0, stream>>>(d_hidden, d_wo, d_bo, d_output,
                                                      num_samples, num_hidden, num_outputs);
}

