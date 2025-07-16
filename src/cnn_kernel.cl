
__kernel void conv2d(
    __global const float* input,
    __global const float* weights,
    __global float* output,
    int input_size,
    int kernel_size,
    int conv_output_size) {

    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    if (out_x >= conv_output_size || out_y >= conv_output_size) return;

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int in_x = out_x + i;
            int in_y = out_y + j;
            sum += input[in_y * input_size + in_x] * weights[i * kernel_size + j];
        }
    }
    output[out_y * conv_output_size + out_x] = fmax(sum, 0.0f);
}



__kernel void maxpool2d(__global float* input, __global float* output,
                        int input_size, int pool_size) {
    int ox = get_global_id(0);
    int oy = get_global_id(1);
    int stride = pool_size;
    int ix0 = ox * stride;
    int iy0 = oy * stride;
    float maxval = input[iy0 * input_size + ix0];
    for (int py = 0; py < pool_size; ++py) {
        for (int px = 0; px < pool_size; ++px) {
            int ix = ox * stride + px;
            int iy = oy * stride + py;
            float v = input[iy * input_size + ix];
            if (v > maxval) maxval = v;
        }
    }
    output[oy * (input_size / pool_size) + ox] = maxval;
}



__kernel void dense_layer(__global const float* input,
                          __global const float* weights,
                          __global const float* biases,
                          __global float* output,
                          const int input_size,
                          const int output_size) {
    int i = get_global_id(0);
    if (i == 0) {
        for (int k = 0; k < output_size; ++k) {
            printf("bias[%d] = %f\n", k, biases[k]);
        }
    }
    if (i >= output_size) return;

    float sum = biases[i];
    for (int j = 0; j < input_size; ++j) {
        sum += input[j] * weights[i * input_size + j];
    }

    output[i] = fmax(sum, 0.0f); // ReLU activation
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
