//
// GPU kernel code
// CUDA version for Nvidia GPUs
//

#include "cnn_kernels.h"
#include <cuda_runtime.h>
#include <cfloat>

//
// Convolution layer, forward pass
// Use fmaxf() for float (in CUDA).
// Block/thread launch config: you will want to launch a 3D grid (out_x, out_y, out_c).
// e.g. dim3 block(8,8,1); dim3 grid(ceil(out_x/8), ceil(out_y/8), out_c);
// Add __restrict__ for optimization (if you know no overlap).
//
__global__ void conv2d(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* output,
    int input_size,
    int kernel_size,
    int conv_output_size)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x >= conv_output_size || out_y >= conv_output_size || out_c >= gridDim.z * blockDim.z)
        return;

    float sum = biases[out_c];
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int in_x = out_x + i;
            int in_y = out_y + j;
            sum += input[in_y * input_size + in_x] *
                   weights[out_c * kernel_size * kernel_size + i * kernel_size + j];
        }
    }
    output[out_c * conv_output_size * conv_output_size + out_y * conv_output_size + out_x] = fmaxf(sum, 0.0f);
}


//
// Convolution layer, backward propagation
// This function accumulates adjustments to weights,
// which will actually be applied in a separate
// function at the end of each batch iteration.
//
__global__ void conv2d_backward_accum(
    const float* __restrict__ in,           // [C, H, W]
    const float* __restrict__ grad_output,  // [F, outH, outW]
    float* grad_weights_accum,              // [F, C, k, k]
    float* grad_biases_accum,               // [F]
    float* grad_input_accum,                // [C, H, W]
    const float* __restrict__ weights,      // [F, C, k, k]
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= outC) return;

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            int go_idx = f * outH * outW + oy * outW + ox;
            float go = grad_output[go_idx];

            atomicAdd(&grad_biases_accum[f], go);

            for (int c = 0; c < inC; ++c) {
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int iy = oy + ky;
                        int ix = ox + kx;
                        if (iy < inH && ix < inW) {
                            int in_idx = c * inH * inW + iy * inW + ix;
                            int w_idx  = f * inC * k * k + c * k * k + ky * k + kx;

                            float dL_dw = in[in_idx] * go;
                            atomicAdd(&grad_weights_accum[w_idx], dL_dw);

                            atomicAdd(&grad_input_accum[in_idx], weights[w_idx] * go);
                        }
                    }
                }
            }
        }
    }
}



//
// Convolution layer, backward propagation
// This function applies accumulated updates
// gathered by the function above, and applies
// them after the inner minibatch loop has run.
//
// This allows for smoother and less chaotic
// weight adjustment during training.
__global__ void conv2d_update(
    float* weights,
    float* biases,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int n_weights,
    int n_biases,
    float learning_rate,
    int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Update weights
    if (i < n_weights) {
        float grad_avg = grad_weights_accum[i] / batch_size;
        weights[i] -= learning_rate * grad_avg;
        grad_weights_accum[i] = 0.0f;
    }
    // Update biases
    if (i < n_biases) {
        float grad_avg = grad_biases_accum[i] / batch_size;
        biases[i] -= learning_rate * grad_avg;
        grad_biases_accum[i] = 0.0f;
    }
}




__global__ void maxpool2d(
    const float* __restrict__ input,     // [channels, in_h, in_w]
    float* output,                       // [channels, out_h, out_w]
    int* max_indices,                    // [channels, out_h, out_w]
    int channels, int in_h, int in_w,
    int pool_size, int out_h, int out_w)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.z * blockDim.z + threadIdx.z;

    if (c >= channels || oy >= out_h || ox >= out_w) return;

    float maxval = -FLT_MAX;
    int maxidx = 0;
    for (int py = 0; py < pool_size; ++py) {
        for (int px = 0; px < pool_size; ++px) {
            int iy = oy * pool_size + py;
            int ix = ox * pool_size + px;
            if (iy < in_h && ix < in_w) {
                int idx = c * in_h * in_w + iy * in_w + ix;
                float v = input[idx];
                if (v > maxval) {
                    maxval = v;
                    maxidx = idx;
                }
            }
        }
    }
    int out_idx = c * out_h * out_w + oy * out_w + ox;
    output[out_idx] = maxval;
    max_indices[out_idx] = maxidx;
}




__global__ void maxpool2d_backward_accum(
    const float* __restrict__ grad_output,   // [channels, out_h, out_w]
    float* grad_input_accum,                 // [channels, in_h, in_w]
    const int* __restrict__ max_indices,     // [channels, out_h, out_w]
    int channels, int out_h, int out_w)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.z * blockDim.z + threadIdx.z;
    if (c >= channels || oy >= out_h || ox >= out_w) return;

    int out_idx = c * out_h * out_w + oy * out_w + ox;
    int in_idx = max_indices[out_idx];
    atomicAdd(&grad_input_accum[in_idx], grad_output[out_idx]);
}



__global__ void dense_layer(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* output,
    int input_size,
    int output_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_size) return;

    float sum = biases[i];
    for (int j = 0; j < input_size; ++j) {
        sum += input[j] * weights[i * input_size + j];
    }
    output[i] = fmaxf(sum, 0.0f); // ReLU
}



__global__ void dense_backward_accum(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int input_size,
    int output_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_size) return;

    for (int j = 0; j < input_size; ++j) {
        int w_idx = i * input_size + j;
        float grad = grad_output[i] * input[j];
        atomicAdd(&grad_weights_accum[w_idx], grad);
    }
    atomicAdd(&grad_biases_accum[i], grad_output[i]);
}



__global__ void dense_update(
    float* weights,
    float* biases,
    float* grad_weights_accum,
    float* grad_biases_accum,
    int input_size,
    int output_size,
    float learning_rate,
    int batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_size) return;

    for (int j = 0; j < input_size; ++j) {
        int w_idx = i * input_size + j;
        float grad_avg = grad_weights_accum[w_idx] / batch_size;
        weights[w_idx] -= learning_rate * grad_avg;
        grad_weights_accum[w_idx] = 0.0f;
    }
    float bias_grad_avg = grad_biases_accum[i] / batch_size;
    biases[i] -= learning_rate * bias_grad_avg;
    grad_biases_accum[i] = 0.0f;
}



__global__ void argmax_cuda(
    const float* input,
    int* output,
    int length)
{
    float max_val = input[0];
    int max_idx = 0;
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    output[0] = max_idx;
}



__global__ void softmax(
    float* input,
    float* output,
    int length)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = input[0];
    for (int i = 1; i < length; ++i) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        sum += expf(input[i] - max_val);
    }

    if (gid < length) {
        output[gid] = expf(input[gid] - max_val) / sum;
    }
}


#define WGSIZE 64

__global__ void softmax_parallel(
    const float* input,
    float* output,
    int len)
{
    __shared__ float shared_max[WGSIZE];
    __shared__ float shared_sum[WGSIZE];

    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int group_start = blockIdx.x * blockDim.x;

    // Step 1: Find local maximum
    float v = (gid < len) ? input[gid] : -FLT_MAX;
    shared_max[lid] = v;
    __syncthreads();

    for (int offset = WGSIZE / 2; offset > 0; offset /= 2) {
        if (lid < offset && (group_start + lid + offset) < len) {
            shared_max[lid] = fmaxf(shared_max[lid], shared_max[lid + offset]);
        }
        __syncthreads();
    }
    float maxval = shared_max[0];
    __syncthreads();

    float expval = (gid < len) ? expf(input[gid] - maxval) : 0.0f;
    shared_sum[lid] = expval;
    __syncthreads();

    for (int offset = WGSIZE / 2; offset > 0; offset /= 2) {
        if (lid < offset && (group_start + lid + offset) < len) {
            shared_sum[lid] += shared_sum[lid + offset];
        }
        __syncthreads();
    }
    float sumval = shared_sum[0];
    __syncthreads();

    if (gid < len && sumval > 0.0f) {
        output[gid] = expval / sumval;
    }
}

// -------------------- Host-side CUDA Helper Functions --------------------

#include <stdio.h>

void conv2d_forward_cuda(
    float* d_input,
    float* d_weights,
    float* d_biases,
    float* d_output,
    int input_size, int kernel_size, int conv_output_size,
    int num_filters
) {
    dim3 blockDim(8,8,1);
    dim3 gridDim((conv_output_size+7)/8, (conv_output_size+7)/8, num_filters);
    conv2d<<<gridDim, blockDim>>>(
        d_input, d_weights, d_biases, d_output,
        input_size, kernel_size, conv_output_size
    );
    cudaDeviceSynchronize();
}

void conv2d_backward_accum_cuda(
    float* d_in,                  // [inC, inH, inW]
    float* d_grad_output,         // [outC, outH, outW]
    float* d_grad_weights_accum,  // [outC, inC, k, k]
    float* d_grad_biases_accum,   // [outC]
    float* d_grad_input_accum,    // [inC, inH, inW]
    float* d_weights,             // [outC, inC, k, k]
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW)
{
    int block = 8;
    int grid = (outC + block - 1) / block;

    conv2d_backward_accum<<<grid, block>>>(
        d_in, d_grad_output, d_grad_weights_accum, d_grad_biases_accum,
        d_grad_input_accum, d_weights,
        inC, inH, inW, outC, k, outH, outW
    );
    cudaDeviceSynchronize();
}

void conv2d_update_cuda(
    float* d_weights,
    float* d_biases,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int n_weights, int n_biases,
    float learning_rate,
    int batch_size)
{
    int block = 128;
    int grid = ((n_weights > n_biases ? n_weights : n_biases) + block - 1) / block;
    conv2d_update<<<grid, block>>>(
        d_weights, d_biases, d_grad_weights_accum, d_grad_biases_accum,
        n_weights, n_biases, learning_rate, batch_size
    );
    cudaDeviceSynchronize();
}

void maxpool2d_forward_cuda(
    float* d_input,
    float* d_output,
    int* d_max_indices,
    int channels, int in_h, int in_w,
    int pool_size, int out_h, int out_w)
{
    dim3 blockDim(4, 4, 4);
    dim3 gridDim(
        (channels + blockDim.x - 1) / blockDim.x,
        (out_h    + blockDim.y - 1) / blockDim.y,
        (out_w    + blockDim.z - 1) / blockDim.z
    );
    maxpool2d<<<gridDim, blockDim>>>(
        d_input, d_output, d_max_indices,
        channels, in_h, in_w, pool_size, out_h, out_w
    );
    cudaDeviceSynchronize();
}

void maxpool2d_backward_accum_cuda(
    float* d_grad_output,
    float* d_grad_input_accum,
    int* d_max_indices,
    int channels, int out_h, int out_w)
{
    dim3 blockDim(4, 4, 4);
    dim3 gridDim(
        (channels + blockDim.x - 1) / blockDim.x,
        (out_h    + blockDim.y - 1) / blockDim.y,
        (out_w    + blockDim.z - 1) / blockDim.z
    );

    maxpool2d_backward_accum<<<gridDim, blockDim>>>(
        d_grad_output, d_grad_input_accum, d_max_indices,
        channels, out_h, out_w
    );
    cudaDeviceSynchronize();
}

void dense_forward_cuda(
    float* d_input,
    float* d_weights,
    float* d_biases,
    float* d_output,
    int input_size,
    int output_size
) {
    int block = 128;
    int grid = (output_size + block - 1) / block;
    dense_layer<<<grid, block>>>(
        d_input, d_weights, d_biases, d_output,
        input_size, output_size
    );
    cudaDeviceSynchronize();
}

void dense_backward_accum_cuda(
    float* d_input,
    float* d_grad_output,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int input_size,
    int output_size
) {
    int block = 128;
    int grid = (output_size + block - 1) / block;
    dense_backward_accum<<<grid, block>>>(
        d_input, d_grad_output, d_grad_weights_accum, d_grad_biases_accum,
        input_size, output_size
    );
    cudaDeviceSynchronize();
}

void dense_update_cuda(
    float* d_weights,
    float* d_biases,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int input_size,
    int output_size,
    float learning_rate,
    int batch_size)
{
    int block = 128;
    int grid = (output_size + block - 1) / block;
    dense_update<<<grid, block>>>(
        d_weights,
        d_biases,
        d_grad_weights_accum,
        d_grad_biases_accum,
        input_size,
        output_size,
        learning_rate,
        batch_size
    );
    cudaDeviceSynchronize();
}

void softmax_forward_cuda(
    float* d_input,
    float* d_output,
    int output_size)
{
    int block = 128;
    int grid = (output_size + block - 1) / block;
    softmax<<<grid, block>>>(
        d_input, d_output, output_size
    );
    cudaDeviceSynchronize();
}

void softmax_parallel_cuda(
    float* d_input,
    float* d_output,
    int output_size)
{
    int block = WGSIZE;
    int grid = (output_size + block - 1) / block;
    softmax_parallel<<<grid, block>>>(
        d_input, d_output, output_size
    );
    cudaDeviceSynchronize();
}

