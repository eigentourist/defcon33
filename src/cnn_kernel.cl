
__kernel void conv2d(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    int input_size,
    int kernel_size,
    int conv_output_size)
{
    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int out_c = get_global_id(2);

    if (out_x >= conv_output_size || out_y >= conv_output_size) return;

    float sum = biases[out_c];
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int in_x = out_x + i;
            int in_y = out_y + j;
            sum += input[in_y * input_size + in_x] *
                   weights[out_c * kernel_size * kernel_size + i * kernel_size + j];
        }
    }
    output[out_c * conv_output_size * conv_output_size + out_y * conv_output_size + out_x] = fmax(sum, 0.0f);
}



__kernel void conv2d_backward(
    __global const float* in,          // [C, H, W] (input to this conv layer)
    __global float* weights,           // [F, C, k, k] (weights to update)
    __global float* biases,            // [F] (biases to update)
    __global const float* grad_output, // [F, outH, outW] (gradient wrt this layer's output)
    __global float* grad_input,        // [C, H, W] (gradient wrt input, to pass back)
    int inC, int inH, int inW,         // Input channels, height, width
    int outC, int k,                   // Output channels (filters), kernel size
    int outH, int outW,                // Output feature map size
    float learning_rate
) {
    int f = get_global_id(0); // output channel (filter)
    if (f >= outC) return;

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            int go_idx = f * outH * outW + oy * outW + ox;
            float go = grad_output[go_idx]; // dL/d(output[f, oy, ox])

            // Update bias
            biases[f] -= learning_rate * go;

            for (int c = 0; c < inC; ++c) {
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int iy = oy + ky;
                        int ix = ox + kx;
                        if (iy < inH && ix < inW) {
                            int in_idx = c * inH * inW + iy * inW + ix;
                            int w_idx  = f * inC * k * k + c * k * k + ky * k + kx;

                            // Save pre-update weight for grad_input math
                            float w_pre = weights[w_idx];

                            // Weight update
                            float dL_dw = in[in_idx] * go;
                            weights[w_idx] -= learning_rate * dL_dw;

                            // Input gradient accum (host must zero before launch!)
                            grad_input[in_idx] += w_pre * go;
                        }
                    }
                }
            }
        }
    }
}



// Batch SGD Version
__kernel void conv2d_backward_accum(
    __global const float* in,            // [C, H, W]
    __global const float* grad_output,   // [F, outH, outW]
    __global float* grad_weights_accum,  // [F, C, k, k]
    __global float* grad_biases_accum,   // [F]
    __global float* grad_input_accum,    // [C, H, W]
    __global const float* weights,       // [F, C, k, k]
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW)
{
    int f = get_global_id(0);
    if (f >= outC) return;

    for (int oy = 0; oy < outH; ++oy) {
        for (int ox = 0; ox < outW; ++ox) {
            int go_idx = f * outH * outW + oy * outW + ox;
            float go = grad_output[go_idx];

            // Bias grad
            grad_biases_accum[f] += go;

            for (int c = 0; c < inC; ++c) {
                for (int ky = 0; ky < k; ++ky) {
                    for (int kx = 0; kx < k; ++kx) {
                        int iy = oy + ky;
                        int ix = ox + kx;
                        if (iy < inH && ix < inW) {
                            int in_idx = c * inH * inW + iy * inW + ix;
                            int w_idx  = f * inC * k * k + c * k * k + ky * k + kx;

                            // dL/dW accumulator
                            float dL_dw = in[in_idx] * go;
                            grad_weights_accum[w_idx] += dL_dw;

                            // dL/dInput accumulator (for downstream)
                            // (This is optional; depends if you want to accumulate over batch or just store last.)
                            grad_input_accum[in_idx] += weights[w_idx] * go;
                        }
                    }
                }
            }
        }
    }
}



// Batch SGD Version
__kernel void conv2d_update(
    __global float* weights,
    __global float* biases,
    __global float* grad_weights_accum,
    __global float* grad_biases_accum,
    int n_weights,   // total number of weights
    int n_biases,    // total number of biases
    float learning_rate,
    int batch_size)
{
    int i = get_global_id(0);
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



__kernel void maxpool2d(
    __global const float* input,         // [channels, in_h, in_w]
    __global float* output,              // [channels, out_h, out_w]
    __global int* max_indices,           // [channels, out_h, out_w]
    int channels, int in_h, int in_w,
    int pool_size, int out_h, int out_w)
{
    int c = get_global_id(0);
    int oy = get_global_id(1);
    int ox = get_global_id(2);

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



// Batch SGD Version -- but integrates with original maxpool2d_backward below
__kernel void maxpool2d_backward_accum(
    __global const float* grad_output,     // [channels, out_h, out_w] (this sample)
    __global float* grad_input_accum,      // [channels, in_h, in_w] (accumulator, not zeroed between samples)
    __global const int* max_indices,       // [channels, out_h, out_w] (from FWD pass of this sample)
    int channels, int out_h, int out_w)
{
    int c = get_global_id(0);
    int oy = get_global_id(1);
    int ox = get_global_id(2);
    if (c >= channels || oy >= out_h || ox >= out_w) return;

    int out_idx = c * out_h * out_w + oy * out_w + ox;
    int in_idx = max_indices[out_idx];
    grad_input_accum[in_idx] += grad_output[out_idx];
}



__kernel void maxpool2d_backward(
    __global const float* grad_output,     // [channels, out_h, out_w]
    __global float* grad_input,            // [channels, in_h, in_w]
    __global const int* max_indices,       // [channels, out_h, out_w]
    int channels, int out_h, int out_w
) {
    int c = get_global_id(0);
    int oy = get_global_id(1);
    int ox = get_global_id(2);

    if (c >= channels || oy >= out_h || ox >= out_w) return;

    int out_idx = c * out_h * out_w + oy * out_w + ox;
    int in_idx = max_indices[out_idx];

    // Add upstream grad to winner in grad_input
    grad_input[in_idx] += grad_output[out_idx];
}



__kernel void dense_layer(__global const float* input,
                          __global const float* weights,
                          __global const float* biases,
                          __global float* output,
                          const int input_size,
                          const int output_size) {
    int i = get_global_id(0);
    /*
    if (i == 0) {
        for (int k = 0; k < output_size; ++k) {
            printf("bias[%d] = %f\n", k, biases[k]);
        }
    }
    */
    if (i >= output_size) return;

    float sum = biases[i];
    for (int j = 0; j < input_size; ++j) {
        sum += input[j] * weights[i * input_size + j];
    }

    output[i] = fmax(sum, 0.0f); // ReLU activation
}



// Dense layer backward kernel
// Kernel: Each output neuron computes its own grad_input contribution
__kernel void dense_backward(
    __global const float* input,           // [input_size]
    __global float* weights,               // [output_size][input_size]
    __global float* biases,                // [output_size]
    __global const float* grad_output,     // [output_size]
    __global float* grad_input_accum,      // [output_size][input_size]
    int input_size,
    int output_size,
    float learning_rate)
{
    int i = get_global_id(0); // output neuron index (class)
    if (i >= output_size) return;

    for (int j = 0; j < input_size; ++j) {
        float dL_dweight = grad_output[i] * input[j];
        int w_idx = i * input_size + j;
        float w_pre = weights[w_idx];
        grad_input_accum[i * input_size + j] = grad_output[i] * w_pre;
        weights[w_idx] -= learning_rate * dL_dweight;
    }
    biases[i] -= learning_rate * grad_output[i];
}


// Batch SGD Version
// Accumulates into grad_weights_accum, grad_biases_accum (not updating weights!)
__kernel void dense_backward_accum(
    __global const float* input,
    __global const float* grad_output,
    __global float* grad_weights_accum, // accumulates dL/dW
    __global float* grad_biases_accum,  // accumulates dL/db
    int input_size,
    int output_size)
{
    int i = get_global_id(0);
    if (i >= output_size) return;

    for (int j = 0; j < input_size; ++j) {
        int w_idx = i * input_size + j;
        float grad = grad_output[i] * input[j];
        grad_weights_accum[w_idx] += grad;
    }
    grad_biases_accum[i] += grad_output[i];
}


// Batch SGD Version
__kernel void dense_update(
    __global float* weights,
    __global float* biases,
    __global float* grad_weights_accum,
    __global float* grad_biases_accum,
    int input_size,
    int output_size,
    float learning_rate,
    int batch_size)
{
    int i = get_global_id(0);
    if (i >= output_size) return;

    for (int j = 0; j < input_size; ++j) {
        int w_idx = i * input_size + j;
        float grad_avg = grad_weights_accum[w_idx] / batch_size;
        weights[w_idx] -= learning_rate * grad_avg;
        grad_weights_accum[w_idx] = 0.0f; // reset for next batch
    }
    float bias_grad_avg = grad_biases_accum[i] / batch_size;
    biases[i] -= learning_rate * bias_grad_avg;
    grad_biases_accum[i] = 0.0f; // reset for next batch
}




__kernel void argmax(
    __global const float* input,
    __global int* output,
    int length
) {
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


__kernel void softmax(
    __global float* input,
    __global float* output,
    const int length)
{
    // Compute global ID
    int gid = get_global_id(0);

    // First, find the max value (for numerical stability)
    float max_val = input[0];
    for (int i = 1; i < length; ++i)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Compute exponentials and sum them up
    float sum = 0.0f;
    for (int i = 0; i < length; ++i)
    {
        sum += exp(input[i] - max_val);
    }

    // Synchronize (though OpenCL 1.2 doesn’t support barriers across work-items in different groups,
    // this works because we're launching just 1 group for softmax on a single vector)

    // Now compute softmax
    if (gid < length)
    {
        output[gid] = exp(input[gid] - max_val) / sum;
    }
}



// Parallel softmax kernel for one vector (output of dense layer)
// Handles any length 'len' (number of classes)
// Assumes you launch with 'global size = len', 'local size = WGSIZE' (usually 32 or 64)
// For very large len, handles multiple workgroups

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Keep WGSIZE value synced with same definition in host program
// Change the value in both if your device prefers another workgroup size
#define WGSIZE 64

__kernel void softmax_parallel(
    __global const float* input,
    __global float* output,
    int len)
{
    __local float shared_max[WGSIZE];
    __local float shared_sum[WGSIZE];

    int lid = get_local_id(0);     // Local ID (0 ... WGSIZE-1)
    int gid = get_global_id(0);    // Global ID (0 ... len-1)
    int group_start = get_group_id(0) * WGSIZE;

    // Step 1: Find local maximum
    float v = (gid < len) ? input[gid] : -FLT_MAX;
    shared_max[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction to find max in each workgroup
    for (int offset = WGSIZE/2; offset > 0; offset /= 2) {
        if (lid < offset && (group_start + lid + offset) < len) {
            shared_max[lid] = fmax(shared_max[lid], shared_max[lid + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float maxval = shared_max[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Subtract max and exponentiate
    float expval = (gid < len) ? exp(input[gid] - maxval) : 0.0f;
    shared_sum[lid] = expval;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction to sum exponentials in each workgroup
    for (int offset = WGSIZE/2; offset > 0; offset /= 2) {
        if (lid < offset && (group_start + lid + offset) < len) {
            shared_sum[lid] += shared_sum[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sumval = shared_sum[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Each thread normalizes its value (for all values in vector)
    if (gid < len && sumval > 0.0f) {
        output[gid] = expval / sumval;
    }
}
