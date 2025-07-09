#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.5
#define EPOCHS 5000

float inputs[NUM_SAMPLES][NUM_INPUTS] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};
float targets[NUM_SAMPLES][NUM_OUTPUTS] = {
    {0}, {1}, {1}, {0}
};

// Load OpenCL kernel from file
char* load_kernel_source(const char* filename, size_t* out_size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    rewind(fp);

    char* buffer = (char*)malloc(filesize + 1);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate buffer for kernel source\n");
        exit(1);
    }
    fread(buffer, 1, filesize, fp);
    buffer[filesize] = '\0';
    fclose(fp);
    if (out_size) *out_size = filesize;
    return buffer;
}

void flatten(float* dst, float arr[][NUM_HIDDEN], int rows) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < NUM_HIDDEN; ++j)
            dst[i * NUM_HIDDEN + j] = arr[i][j];
}
void flatten_wh(float* dst, float arr[][NUM_HIDDEN]) {
    for (int i = 0; i < NUM_INPUTS; ++i)
        for (int j = 0; j < NUM_HIDDEN; ++j)
            dst[i * NUM_HIDDEN + j] = arr[i][j];
}
void flatten_wo(float* dst, float arr[][NUM_OUTPUTS]) {
    for (int j = 0; j < NUM_HIDDEN; ++j)
        for (int k = 0; k < NUM_OUTPUTS; ++k)
            dst[j * NUM_OUTPUTS + k] = arr[j][k];
}

int main() {
    srand((unsigned int)time(NULL));

    // Initialize weights and biases
    float wh[NUM_INPUTS][NUM_HIDDEN];
    float bh[NUM_HIDDEN];
    float wo[NUM_HIDDEN][NUM_OUTPUTS];
    float bo[NUM_OUTPUTS];
    for(int i=0; i<NUM_INPUTS; ++i)
        for(int j=0; j<NUM_HIDDEN; ++j)
            wh[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    for(int j=0; j<NUM_HIDDEN; ++j)
        bh[j] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    for(int j=0; j<NUM_HIDDEN; ++j)
        for(int k=0; k<NUM_OUTPUTS; ++k)
            wo[j][k] = ((float)rand()/RAND_MAX - 0.5f) * 2;
    for(int k=0; k<NUM_OUTPUTS; ++k)
        bo[k] = ((float)rand()/RAND_MAX - 0.5f) * 2;

    // Flattened buffers for OpenCL
    float inputs_flat[NUM_SAMPLES * NUM_INPUTS];
    for (int i = 0; i < NUM_SAMPLES; ++i)
        for (int j = 0; j < NUM_INPUTS; ++j)
            inputs_flat[i * NUM_INPUTS + j] = inputs[i][j];
    float wh_flat[NUM_INPUTS * NUM_HIDDEN];
    float bh_flat[NUM_HIDDEN];
    float wo_flat[NUM_HIDDEN * NUM_OUTPUTS];
    float bo_flat[NUM_OUTPUTS];
    float hidden_flat[NUM_SAMPLES * NUM_HIDDEN];
    float output_flat[NUM_SAMPLES * NUM_OUTPUTS];

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel_hidden, kernel_output;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    size_t kernel_size;
    char* kernelSource = load_kernel_source("src/ml_perceptron_kernel.cl", &kernel_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernel_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        free(kernelSource);
        return 1;
    }
    kernel_hidden = clCreateKernel(program, "forward_hidden", &err);
    kernel_output = clCreateKernel(program, "forward_output", &err);

    // Buffers
    cl_mem buf_inputs = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(inputs_flat), NULL, &err);
    cl_mem buf_wh = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(wh_flat), NULL, &err);
    cl_mem buf_bh = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bh_flat), NULL, &err);
    cl_mem buf_hidden = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(hidden_flat), NULL, &err);
    cl_mem buf_wo = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(wo_flat), NULL, &err);
    cl_mem buf_bo = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bo_flat), NULL, &err);
    cl_mem buf_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output_flat), NULL, &err);

    clEnqueueWriteBuffer(queue, buf_inputs, CL_TRUE, 0, sizeof(inputs_flat), inputs_flat, 0, NULL, NULL);

    // Training loop
    for(int epoch=0; epoch<EPOCHS; ++epoch) {
        float total_error = 0.0f;

        // Update flattened weights/biases for this epoch
        flatten_wh(wh_flat, wh);
        for(int j=0; j<NUM_HIDDEN; ++j) bh_flat[j] = bh[j];
        flatten_wo(wo_flat, wo);
        for(int k=0; k<NUM_OUTPUTS; ++k) bo_flat[k] = bo[k];

        // Upload weights/biases
        clEnqueueWriteBuffer(queue, buf_wh, CL_TRUE, 0, sizeof(wh_flat), wh_flat, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, buf_bh, CL_TRUE, 0, sizeof(bh_flat), bh_flat, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, buf_wo, CL_TRUE, 0, sizeof(wo_flat), wo_flat, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, buf_bo, CL_TRUE, 0, sizeof(bo_flat), bo_flat, 0, NULL, NULL);

        // Forward pass: hidden layer
        clSetKernelArg(kernel_hidden, 0, sizeof(cl_mem), &buf_inputs);
        clSetKernelArg(kernel_hidden, 1, sizeof(cl_mem), &buf_wh);
        clSetKernelArg(kernel_hidden, 2, sizeof(cl_mem), &buf_bh);
        clSetKernelArg(kernel_hidden, 3, sizeof(cl_mem), &buf_hidden);
        int num_inputs = NUM_INPUTS, num_hidden = NUM_HIDDEN;
        clSetKernelArg(kernel_hidden, 4, sizeof(int), &num_inputs);
        clSetKernelArg(kernel_hidden, 5, sizeof(int), &num_hidden);
        size_t gw = NUM_SAMPLES;
        clEnqueueNDRangeKernel(queue, kernel_hidden, 1, NULL, &gw, NULL, 0, NULL, NULL);

        // Forward pass: output layer
        clSetKernelArg(kernel_output, 0, sizeof(cl_mem), &buf_hidden);
        clSetKernelArg(kernel_output, 1, sizeof(cl_mem), &buf_wo);
        clSetKernelArg(kernel_output, 2, sizeof(cl_mem), &buf_bo);
        clSetKernelArg(kernel_output, 3, sizeof(cl_mem), &buf_output);
        int num_outputs = NUM_OUTPUTS;
        clSetKernelArg(kernel_output, 4, sizeof(int), &num_hidden);
        clSetKernelArg(kernel_output, 5, sizeof(int), &num_outputs);
        clEnqueueNDRangeKernel(queue, kernel_output, 1, NULL, &gw, NULL, 0, NULL, NULL);

        // Download hidden & output activations
        clEnqueueReadBuffer(queue, buf_hidden, CL_TRUE, 0, sizeof(hidden_flat), hidden_flat, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, sizeof(output_flat), output_flat, 0, NULL, NULL);

        // Backpropagation & update (CPU)
        for(int n=0; n<NUM_SAMPLES; ++n) {
            // output delta
            float oo = output_flat[n * NUM_OUTPUTS + 0];
            float target = targets[n][0];
            float d_oo = (target - oo) * oo * (1.0f - oo); // sigmoid derivative

            // hidden deltas
            float d_ho[NUM_HIDDEN];
            for(int j=0; j<NUM_HIDDEN; ++j) {
                float ho = hidden_flat[n * NUM_HIDDEN + j];
                d_ho[j] = d_oo * wo[j][0] * ho * (1.0f - ho);
            }

            // update output weights/biases
            for(int j=0; j<NUM_HIDDEN; ++j)
                wo[j][0] += LEARNING_RATE * d_oo * hidden_flat[n * NUM_HIDDEN + j];
            bo[0] += LEARNING_RATE * d_oo;

            // update input-to-hidden weights/biases
            for(int j=0; j<NUM_HIDDEN; ++j) {
                for(int i=0; i<NUM_INPUTS; ++i)
                    wh[i][j] += LEARNING_RATE * d_ho[j] * inputs[n][i];
                bh[j] += LEARNING_RATE * d_ho[j];
            }

            float e = target - oo;
            total_error += e * e;
        }

        if((epoch+1) % 500 == 0)
            printf("Epoch %d | MSE: %.5f\n", epoch+1, total_error / NUM_SAMPLES);
    }

    // Test final network
    printf("\nTest results (after training):\n");
    flatten_wh(wh_flat, wh);
    for(int j=0; j<NUM_HIDDEN; ++j) bh_flat[j] = bh[j];
    flatten_wo(wo_flat, wo);
    for(int k=0; k<NUM_OUTPUTS; ++k) bo_flat[k] = bo[k];

    clEnqueueWriteBuffer(queue, buf_wh, CL_TRUE, 0, sizeof(wh_flat), wh_flat, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_bh, CL_TRUE, 0, sizeof(bh_flat), bh_flat, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_wo, CL_TRUE, 0, sizeof(wo_flat), wo_flat, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_bo, CL_TRUE, 0, sizeof(bo_flat), bo_flat, 0, NULL, NULL);

    clSetKernelArg(kernel_hidden, 0, sizeof(cl_mem), &buf_inputs);
    clSetKernelArg(kernel_hidden, 1, sizeof(cl_mem), &buf_wh);
    clSetKernelArg(kernel_hidden, 2, sizeof(cl_mem), &buf_bh);
    clSetKernelArg(kernel_hidden, 3, sizeof(cl_mem), &buf_hidden);
    int num_inputs = NUM_INPUTS, num_hidden = NUM_HIDDEN;
    clSetKernelArg(kernel_hidden, 4, sizeof(int), &num_inputs);
    clSetKernelArg(kernel_hidden, 5, sizeof(int), &num_hidden);
    size_t gw = NUM_SAMPLES;
    clEnqueueNDRangeKernel(queue, kernel_hidden, 1, NULL, &gw, NULL, 0, NULL, NULL);

    clSetKernelArg(kernel_output, 0, sizeof(cl_mem), &buf_hidden);
    clSetKernelArg(kernel_output, 1, sizeof(cl_mem), &buf_wo);
    clSetKernelArg(kernel_output, 2, sizeof(cl_mem), &buf_bo);
    clSetKernelArg(kernel_output, 3, sizeof(cl_mem), &buf_output);
    int num_outputs = NUM_OUTPUTS;
    clSetKernelArg(kernel_output, 4, sizeof(int), &num_hidden);
    clSetKernelArg(kernel_output, 5, sizeof(int), &num_outputs);
    clEnqueueNDRangeKernel(queue, kernel_output, 1, NULL, &gw, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, buf_output, CL_TRUE, 0, sizeof(output_flat), output_flat, 0, NULL, NULL);

    for(int n=0; n<NUM_SAMPLES; ++n) {
        printf("Input: [%g, %g] => Output: %.3f (Target: %.1f)\n",
            inputs[n][0], inputs[n][1], output_flat[n * NUM_OUTPUTS + 0], targets[n][0]);
    }

    // Cleanup
    clReleaseMemObject(buf_inputs);
    clReleaseMemObject(buf_wh);
    clReleaseMemObject(buf_bh);
    clReleaseMemObject(buf_hidden);
    clReleaseMemObject(buf_wo);
    clReleaseMemObject(buf_bo);
    clReleaseMemObject(buf_output);
    clReleaseKernel(kernel_hidden);
    clReleaseKernel(kernel_output);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernelSource);

    return 0;
}
