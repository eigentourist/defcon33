// perceptron_kernel.cl
__kernel void perceptron_forward(
    __global const float* inputs,
    __global const float* weights,
    float bias,
    __global int* outputs,
    int num_inputs)
{
    int i = get_global_id(0); // one work-item per sample
    float sum = bias;
    for (int j = 0; j < num_inputs; ++j)
        sum += inputs[i * num_inputs + j] * weights[j];
    outputs[i] = (sum >= 0.0f) ? 1 : 0;
}
