#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>

#define NUM_INPUTS 2
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.1f
#define EPOCHS 20

// Pick your gate:
int targets[NUM_SAMPLES] = {0, 0, 0, 1}; // AND
//int targets[NUM_SAMPLES] = {0, 1, 1, 1}; // OR
//int targets[NUM_SAMPLES] = {1, 1, 1, 0}; // NAND

float inputs[NUM_SAMPLES][NUM_INPUTS] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

// Function to load OpenCL kernel source from a file
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
    buffer[filesize] = '\0'; // Null-terminate
    fclose(fp);
    if (out_size) *out_size = filesize;
    return buffer;
}

void ascii_bar(float err) {
    int len = (int)(err * 60 + 0.5f); // 0..1 maps to 0..60 chars
    printf(" [");
    for (int i = 0; i < len; ++i) printf("#");
    for (int i = len; i < 60; ++i) printf(" ");
    printf("]\n");
}

int main() {
    // Flatten input array for OpenCL
    float inputs_flat[NUM_SAMPLES * NUM_INPUTS];
    for (int i = 0; i < NUM_SAMPLES; ++i)
        for (int j = 0; j < NUM_INPUTS; ++j)
            inputs_flat[i * NUM_INPUTS + j] = inputs[i][j];

    float weights[NUM_INPUTS] = {0.0f, 0.0f};
    float bias = 0.0f;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // 1. Setup OpenCL platform/device/context/queue
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue   = clCreateCommandQueue(context, device, 0, &err);

    // 2. Load kernel source from file
    size_t kernel_size;
    char* kernelSource = load_kernel_source("src/perceptron_kernel.cl", &kernel_size);

    // 3. Compile kernel
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernel_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build errors
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        free(kernelSource);
        return 1;
    }
    kernel = clCreateKernel(program, "perceptron_forward", &err);

    // 4. Allocate buffers
    cl_mem buf_inputs = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(inputs_flat), NULL, &err);
    cl_mem buf_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(weights), NULL, &err);
    cl_mem buf_outputs = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * NUM_SAMPLES, NULL, &err);

    // 5. Copy data to device
    clEnqueueWriteBuffer(queue, buf_inputs, CL_TRUE, 0, sizeof(inputs_flat), inputs_flat, 0, NULL, NULL);

    // Training loop
    printf("\nEpoch   Error   Learning Progress\n");
    printf("==========================================\n");
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Copy weights for this epoch
        clEnqueueWriteBuffer(queue, buf_weights, CL_TRUE, 0, sizeof(weights), weights, 0, NULL, NULL);

        // Set kernel args
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_inputs);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_weights);
        clSetKernelArg(kernel, 2, sizeof(float), &bias);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_outputs);
        int num_inputs = NUM_INPUTS;
        clSetKernelArg(kernel, 4, sizeof(int), &num_inputs);

        size_t global_work_size = NUM_SAMPLES;
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

        // Read outputs
        int outputs[NUM_SAMPLES];
        clEnqueueReadBuffer(queue, buf_outputs, CL_TRUE, 0, sizeof(outputs), outputs, 0, NULL, NULL);

        // Update weights (CPU)
        int sum_error = 0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            int error = targets[i] - outputs[i];
            sum_error += abs(error);

            for (int j = 0; j < NUM_INPUTS; ++j)
                weights[j] += LEARNING_RATE * error * inputs[i][j];
            bias += LEARNING_RATE * error;
        }

        float norm_err = sum_error / (float)NUM_SAMPLES;
        printf(" %2d    %.3f  ", epoch + 1, norm_err);
        ascii_bar(norm_err / 2.0f); // 0..2 error, normalize for bar
    }

    printf("\nTrained weights: %.2f %.2f\nTrained bias: %.2f\n\n", weights[0], weights[1], bias);

    // Test final perceptron
    printf("Testing perceptron after training:\n");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float sum = bias;
        for (int j = 0; j < NUM_INPUTS; ++j)
            sum += inputs[i][j] * weights[j];
        int output = (sum >= 0.0f) ? 1 : 0;
        printf("Input: [%g, %g], Output: %d, Target: %d\n",
               inputs[i][0], inputs[i][1], output, targets[i]);
    }

    // Cleanup
    clReleaseMemObject(buf_inputs);
    clReleaseMemObject(buf_weights);
    clReleaseMemObject(buf_outputs);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(kernelSource);

    return 0;
}
