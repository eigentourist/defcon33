#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 16
#define KERNEL_SIZE 3
#define CONV_OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (CONV_OUTPUT_SIZE / POOL_SIZE)
#define DENSE_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE)
#define DENSE_OUTPUT_SIZE 10

// Simple kernel loader
char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to load kernel.");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

int main() {
    float input[INPUT_SIZE * INPUT_SIZE];
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    float dense_weights[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
    float dense_biases[DENSE_OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; ++i)
        input[i] = (i % 2 == 0) ? 0.0f : 1.0f;

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
        kernel[i] = 0.111f;

    for (int i = 0; i < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++i)
        dense_weights[i] = i / (float)(DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE);

    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        dense_biases[i] = i * 0.1f;

    printf("First few input values:\n");
    for (int i = 0; i < 8; ++i) printf("%.1f ", input[i]);
    printf("\nKernel values:\n");
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) printf("%.3f ", kernel[i]);
    printf("\n");

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_mem input_buf, kernel_buf, conv_out_buf, pool_out_buf;
    cl_mem dense_w_buf, dense_b_buf, dense_out_buf, argmax_buf;

    cl_kernel conv_kernel, pool_kernel, dense_kernel, argmax_kernel;

    size_t conv_output_size = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE;
    size_t pool_output_size = POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE;

    // OpenCL init
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Kernel
    char* source = load_kernel_source("src/cnn_kernel.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL kernel build failed. Log:\n%s\n", log);
        free(log);
        exit(1);
    }
    free(source);

    // Buffers
    input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE * INPUT_SIZE, input, NULL);
    kernel_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE, kernel, NULL);
    conv_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * conv_output_size, NULL, NULL);
    pool_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * pool_output_size, NULL, NULL);
    dense_w_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_INPUT_SIZE * DENSE_OUTPUT_SIZE, dense_weights, NULL);
    dense_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_OUTPUT_SIZE, dense_biases, NULL);
    dense_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DENSE_OUTPUT_SIZE, NULL, NULL);
    argmax_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);

    // dimension constants to pass to kernels
    int input_dim = INPUT_SIZE;
    int kernel_dim = KERNEL_SIZE;
    int conv_out_dim = CONV_OUTPUT_SIZE;
    int pool_dim = POOL_SIZE;
    int pool_out_dim = POOL_OUTPUT_SIZE;
    int dense_in_dim = DENSE_INPUT_SIZE;
    int dense_out_dim = DENSE_OUTPUT_SIZE;

    // conv2d
    conv_kernel = clCreateKernel(program, "conv2d", &err);
    clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &kernel_buf);
    clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_out_buf);
    clSetKernelArg(conv_kernel, 3, sizeof(int), &input_dim);
    clSetKernelArg(conv_kernel, 4, sizeof(int), &kernel_dim);
    clSetKernelArg(conv_kernel, 5, sizeof(int), &conv_out_dim);

    size_t conv_global[2] = { CONV_OUTPUT_SIZE, CONV_OUTPUT_SIZE };
    clEnqueueNDRangeKernel(queue, conv_kernel, 2, NULL, conv_global, NULL, 0, NULL, NULL);

    // maxpool2d
    pool_kernel = clCreateKernel(program, "maxpool2d", &err);
    clSetKernelArg(pool_kernel, 0, sizeof(cl_mem), &conv_out_buf);
    clSetKernelArg(pool_kernel, 1, sizeof(cl_mem), &pool_out_buf);
    clSetKernelArg(pool_kernel, 2, sizeof(int), &conv_output_size);
    clSetKernelArg(pool_kernel, 3, sizeof(int), &pool_dim);
    clSetKernelArg(pool_kernel, 4, sizeof(int), &pool_out_dim);

    size_t pool_global[2] = { POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE };
    clEnqueueNDRangeKernel(queue, pool_kernel, 2, NULL, pool_global, NULL, 0, NULL, NULL);

    // dense
    dense_kernel = clCreateKernel(program, "dense_layer", &err);
    clSetKernelArg(dense_kernel, 0, sizeof(cl_mem), &pool_out_buf);
    clSetKernelArg(dense_kernel, 1, sizeof(cl_mem), &dense_w_buf);
    clSetKernelArg(dense_kernel, 2, sizeof(cl_mem), &dense_b_buf);
    clSetKernelArg(dense_kernel, 3, sizeof(cl_mem), &dense_out_buf);
    clSetKernelArg(dense_kernel, 4, sizeof(int), (void*)&dense_in_dim);
    clSetKernelArg(dense_kernel, 5, sizeof(int), (void*)&dense_out_dim);

    size_t dense_global = DENSE_OUTPUT_SIZE;
    clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &dense_global, NULL, 0, NULL, NULL);

    // argmax
    argmax_kernel = clCreateKernel(program, "argmax", &err);
    clSetKernelArg(argmax_kernel, 0, sizeof(cl_mem), &dense_out_buf);
    clSetKernelArg(argmax_kernel, 1, sizeof(cl_mem), &argmax_buf);
    clSetKernelArg(argmax_kernel, 2, sizeof(int), (void*)&dense_out_dim);

    clEnqueueTask(queue, argmax_kernel, 0, NULL, NULL);

    // Get result
    float output[DENSE_OUTPUT_SIZE];
    int prediction = -1;
    clEnqueueReadBuffer(queue, dense_out_buf, CL_TRUE, 0, sizeof(output), output, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, argmax_buf, CL_TRUE, 0, sizeof(int), &prediction, 0, NULL, NULL);

    printf("Dense layer output:\n");
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) printf("%.3f ", output[i]);
    printf("\nPredicted class (argmax): %d\n", prediction);

    return 0;
}
