__kernel void conv2d(
    __global const float* input,
    __global const float* weights,
    __global float* output) {

    const int out_x = get_global_id(0);
    const int out_y = get_global_id(1);

    const int INPUT_SIZE = 16;
    const int KERNEL_SIZE = 3;
    const int CONV_OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1;

    float sum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            int in_x = out_x + i;
            int in_y = out_y + j;
            sum += input[in_y * INPUT_SIZE + in_x] * weights[i * KERNEL_SIZE + j];
        }
    }
    output[out_y * CONV_OUTPUT_SIZE + out_x] = fmax(sum, 0.0f); // ReLU activation
}

__kernel void maxpool2d(
    __global const float* input,
    __global float* output,
    const int conv_output_size,
    const int pool_size) {

    int out_x = get_global_id(0);
    int out_y = get_global_id(1);
    int pool_output_size = conv_output_size / pool_size;

    int base_x = out_x * pool_size;
    int base_y = out_y * pool_size;

    float maxval = -INFINITY;
    for (int i = 0; i < pool_size; i++) {
        for (int j = 0; j < pool_size; j++) {
            int idx = (base_y + i) * conv_output_size + (base_x + j);
            float val = input[idx];
            if (val > maxval) maxval = val;
        }
    }

    if (out_x < pool_output_size && out_y < pool_output_size) {
        output[out_y * pool_output_size + out_x] = maxval;
    }
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
