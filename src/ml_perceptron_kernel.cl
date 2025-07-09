// perceptron_mlp_kernel.cl

// Compute hidden layer activations for all samples
__kernel void forward_hidden(
    __global const float* inputs,    // NUM_SAMPLES x NUM_INPUTS
    __global const float* wh,        // NUM_INPUTS x NUM_HIDDEN
    __global const float* bh,        // NUM_HIDDEN
    __global float* hidden,          // NUM_SAMPLES x NUM_HIDDEN (output)
    int num_inputs,
    int num_hidden
) {
    int sample = get_global_id(0);
    for(int j = 0; j < num_hidden; ++j) {
        float sum = bh[j];
        for(int i = 0; i < num_inputs; ++i)
            sum += inputs[sample * num_inputs + i] * wh[i * num_hidden + j];
        // sigmoid activation
        hidden[sample * num_hidden + j] = 1.0f / (1.0f + exp(-sum));
    }
}

// Compute output layer activations for all samples
__kernel void forward_output(
    __global const float* hidden,    // NUM_SAMPLES x NUM_HIDDEN
    __global const float* wo,        // NUM_HIDDEN x NUM_OUTPUTS
    __global const float* bo,        // NUM_OUTPUTS
    __global float* output,          // NUM_SAMPLES x NUM_OUTPUTS (output)
    int num_hidden,
    int num_outputs
) {
    int sample = get_global_id(0);
    for(int k = 0; k < num_outputs; ++k) {
        float sum = bo[k];
        for(int j = 0; j < num_hidden; ++j)
            sum += hidden[sample * num_hidden + j] * wo[j * num_outputs + k];
        // sigmoid activation
        output[sample * num_outputs + k] = 1.0f / (1.0f + exp(-sum));
    }
}
