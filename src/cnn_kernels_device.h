#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <float.h>   // For FLT_MAX

// CUDA block size for softmax_parallel
#define WGSIZE 64

// -------------------- KERNEL PROTOTYPES --------------------

__global__ void conv2d(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* output,
    int input_size,
    int kernel_size,
    int conv_output_size);

__global__ void conv2d_backward_accum(
    const float* __restrict__ in,           // [C, H, W]
    const float* __restrict__ grad_output,  // [F, outH, outW]
    float* grad_weights_accum,              // [F, C, k, k]
    float* grad_biases_accum,               // [F]
    float* grad_input_accum,                // [C, H, W]
    const float* __restrict__ weights,      // [F, C, k, k]
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW);

__global__ void conv2d_update(
    float* weights,
    float* biases,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int n_weights,
    int n_biases,
    float learning_rate,
    int batch_size);

__global__ void maxpool2d(
    const float* __restrict__ input,
    float* output,
    int* max_indices,
    int channels, int in_h, int in_w,
    int pool_size, int out_h, int out_w);

__global__ void maxpool2d_backward_accum(
    const float* __restrict__ grad_output,
    float* grad_input_accum,
    const int* __restrict__ max_indices,
    int channels, int out_h, int out_w);

__global__ void dense_layer(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* output,
    int input_size,
    int output_size);

__global__ void dense_backward_accum(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int input_size,
    int output_size);

__global__ void dense_update(
    float* weights,
    float* biases,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int input_size,
    int output_size,
    float learning_rate,
    int batch_size);

__global__ void argmax_cuda(
    const float* input,
    int* output,
    int length);

__global__ void softmax(
    float* input,
    float* output,
    int length);

__global__ void softmax_parallel(
    const float* input,
    float* output,
    int len);

#ifdef __cplusplus
}
#endif

#endif // CNN_KERNELS_H

