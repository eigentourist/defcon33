#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Host-callable CUDA wrappers (implemented in .cu file)
// These launch the CUDA kernels and handle any sync

void conv2d_forward_cuda(
    float* d_input,
    float* d_weights,
    float* d_biases,
    float* d_output,
    int input_size,
    int kernel_size,
    int conv_output_size,
    int num_filters
);

void conv2d_backward_accum_cuda(
    float* d_in,                  // [inC, inH, inW]
    float* d_grad_output,         // [outC, outH, outW]
    float* d_grad_weights_accum,  // [outC, inC, k, k]
    float* d_grad_biases_accum,   // [outC]
    float* d_grad_input_accum,    // [inC, inH, inW]
    float* d_weights,             // [outC, inC, k, k]
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW
);

void conv2d_update_cuda(
    float* d_weights,
    float* d_biases,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int n_weights,
    int n_biases,
    float learning_rate,
    int batch_size
);

void maxpool2d_forward_cuda(
    float* d_input,
    float* d_output,
    int* d_max_indices,
    int channels,
    int in_h,
    int in_w,
    int pool_size,
    int out_h,
    int out_w
);

void maxpool2d_backward_accum_cuda(
    float* d_grad_output,
    float* d_grad_input_accum,
    int* d_max_indices,
    int channels,
    int out_h,
    int out_w
);

void maxpool2d_backward_cuda(
    float* d_grad_output,
    float* d_grad_input,
    int* d_max_indices,
    int channels,
    int out_h,
    int out_w
);

void dense_forward_cuda(
    float* d_input,
    float* d_weights,
    float* d_biases,
    float* d_output,
    int input_size,
    int output_size
);

void dense_backward_accum_cuda(
    float* d_input,
    float* d_grad_output,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int input_size,
    int output_size
);

void dense_update_cuda(
    float* d_weights,
    float* d_biases,
    float* d_grad_weights_accum,
    float* d_grad_biases_accum,
    int input_size,
    int output_size,
    float learning_rate,
    int batch_size
);

void softmax_forward_cuda(
    float* d_input,
    float* d_output,
    int output_size
);

void softmax_parallel_cuda(
    float* d_input,
    float* d_output,
    int output_size
);

#ifdef __cplusplus
}
#endif

#endif // CNN_KERNELS_H

