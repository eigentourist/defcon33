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
    __global float* output) {

    const int CONV_OUTPUT_SIZE = 14; // fixed size for now
    const int POOL_SIZE = 2;
    const int POOL_OUTPUT_SIZE = CONV_OUTPUT_SIZE / POOL_SIZE;

    int out_x = get_global_id(0);
    int out_y = get_global_id(1);

    int base_x = out_x * POOL_SIZE;
    int base_y = out_y * POOL_SIZE;

    float maxval = -INFINITY;
    for (int i = 0; i < POOL_SIZE; i++) {
        for (int j = 0; j < POOL_SIZE; j++) {
            int idx = (base_y + i) * CONV_OUTPUT_SIZE + (base_x + j);
            float val = input[idx];
            if (val > maxval) maxval = val;
        }
    }
    output[out_y * POOL_OUTPUT_SIZE + out_x] = maxval;
}

__kernel void dense_layer(__global const float* input,
                          __global const float* weights,
                          __global const float* biases,
                          __global float* output,
                          const int input_size,
                          const int output_size) {
    int i = get_global_id(0);
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
