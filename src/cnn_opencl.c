#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 16
#define KERNEL_SIZE 3
#define CONV_OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (CONV_OUTPUT_SIZE / POOL_SIZE)
#define DENSE_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE)
#define DENSE_OUTPUT_SIZE 10

// For parallelized softmax
#define WGSIZE 64

char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to load kernel file");
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


cl_int dense_layer_backprop(
    cl_context context,
    cl_command_queue queue,
    cl_kernel dense_backward_kernel,
    cl_mem pool_output_buf,
    cl_mem dense_weights_buf,
    cl_mem dense_biases_buf,
    cl_mem grad_output_buf,
    cl_mem grad_input_buf,
    float* softmax_output,
    int label,
    int input_size,
    int output_size,
    float learning_rate)
{
    // -- Compute grad_output on host
    float grad_output[output_size];
    for (int k = 0; k < output_size; ++k) {
        grad_output[k] = softmax_output[k] - (k == label ? 1.0f : 0.0f);
    }

    // -- Zero grad_input
    float grad_input[input_size];
    memset(grad_input, 0, sizeof(float) * input_size);
    cl_int err = 0;
    cl_mem grad_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE, NULL, &err);
    if (err != CL_SUCCESS) printf("grad_input_accum_buf creation error: %d\n", err);
    return err;

    // -- Copy to device
    clEnqueueWriteBuffer(queue, grad_output_buf, CL_TRUE, 0,
        sizeof(float) * output_size, grad_output, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, grad_input_buf, CL_TRUE, 0,
        sizeof(float) * input_size, grad_input, 0, NULL, NULL);

    // -- Set kernel args
    clSetKernelArg(dense_backward_kernel, 0, sizeof(cl_mem), &pool_output_buf);
    clSetKernelArg(dense_backward_kernel, 1, sizeof(cl_mem), &dense_weights_buf);
    clSetKernelArg(dense_backward_kernel, 2, sizeof(cl_mem), &dense_biases_buf);
    clSetKernelArg(dense_backward_kernel, 3, sizeof(cl_mem), &grad_output_buf);
    clSetKernelArg(dense_backward_kernel, 4, sizeof(cl_mem), &grad_input_buf);
    clSetKernelArg(dense_backward_kernel, 5, sizeof(int), &input_size);
    clSetKernelArg(dense_backward_kernel, 6, sizeof(int), &output_size);
    clSetKernelArg(dense_backward_kernel, 7, sizeof(float), &learning_rate);

    // -- Launch
    size_t global_size = output_size;
    clEnqueueNDRangeKernel(queue, dense_backward_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    float grad_input_accum[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
    clEnqueueReadBuffer(queue, grad_input_accum_buf, CL_TRUE, 0,
        sizeof(grad_input_accum), grad_input_accum, 0, NULL, NULL);
    for (int j = 0; j < DENSE_INPUT_SIZE; ++j) {
        for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) {
            grad_input[j] += grad_input_accum[i * DENSE_INPUT_SIZE + j];
        }
    }

    // Optionally, read back grad_input here if needed for further backprop
    // clEnqueueReadBuffer(queue, grad_input_buf, CL_TRUE, 0,
    //    sizeof(float) * input_size, grad_input, 0, NULL, NULL);
}


int main() {
    float input[INPUT_SIZE * INPUT_SIZE];
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    float dense_weights[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
    float dense_biases[DENSE_OUTPUT_SIZE];

    for (int row = 0; row < INPUT_SIZE; ++row) {
        for (int col = 0; col < INPUT_SIZE; ++col) {
            int i = row * INPUT_SIZE + col;
            if (row >= 8 && row < 24 && col >= 8 && col < 24)
                input[i] = 1.0f;
            else
                input[i] = 0.0f;
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
        kernel[i] = 1.0f / 9.0f;

    for (int i = 0; i < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++i)
        dense_weights[i] = 0.1f;
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        dense_biases[i] = (float)i * 0.1f;

    // Paranoia test: force a negative bias and a negative weight
    // dense_biases[0] = -2.0f;
    // dense_weights[0 * DENSE_INPUT_SIZE + 0] = -0.5f;


    printf("First few input values:\n");
    for (int i = 0; i < 8; ++i)
        printf("%.1f ", input[i]);
    printf("\nKernel values:\n");
    for (int i = 0; i < 9; ++i)
        printf("%.3f ", kernel[i]);
    printf("\n");
    printf("First few bias values:\n");
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) {
        printf("%.3f ", dense_biases[i]);
    }
    printf("\n");


    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    char* source = load_kernel_source("src/cnn_kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error:\n%s\n", log);
        free(log);
        return 1;
    }
    free(source);

    size_t conv_output_bytes = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE * sizeof(float);
    size_t pool_output_bytes = POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE * sizeof(float);
    size_t dense_output_bytes = DENSE_OUTPUT_SIZE * sizeof(float);

    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE * INPUT_SIZE, input, &err);
    if (err != CL_SUCCESS) printf("input_buf creation error: %d\n", err);

    cl_mem kernel_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE, kernel, &err);
    if (err != CL_SUCCESS) printf("kernel_buf creation error: %d\n", err);

    cl_mem conv_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, conv_output_bytes, NULL, &err);
    if (err != CL_SUCCESS) printf("conv_output_buf creation error: %d\n", err);

    cl_mem pool_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, pool_output_bytes, NULL, &err);
    if (err != CL_SUCCESS) printf("pool_output_buf creation error: %d\n", err);

    cl_mem dense_weights_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_INPUT_SIZE * DENSE_OUTPUT_SIZE, dense_weights, &err);
    if (err != CL_SUCCESS) printf("dense_weights_buf creation error: %d\n", err);

    cl_mem dense_biases_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_OUTPUT_SIZE, dense_biases, &err);
    if (err != CL_SUCCESS) printf("dense_biases_buf creation error: %d\n", err);

    cl_mem dense_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dense_output_bytes, NULL, &err);
    if (err != CL_SUCCESS) printf("dense_output_buf creation error: %d\n", err);

    cl_mem softmax_output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dense_output_bytes, NULL, &err);
    if (err != CL_SUCCESS) printf("softmax_output_buf creation error: %d\n", err);


    int input_size_val = INPUT_SIZE;
    int kernel_size_val = KERNEL_SIZE;
    int conv_out_val = CONV_OUTPUT_SIZE;
    int pool_size_val = POOL_SIZE;
    int pool_out_val = POOL_OUTPUT_SIZE;
    int dense_in_val = DENSE_INPUT_SIZE;
    int dense_out_val = DENSE_OUTPUT_SIZE;

    cl_kernel conv_kernel = clCreateKernel(program, "conv2d", &err);
    clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &kernel_buf);
    clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_output_buf);
    clSetKernelArg(conv_kernel, 3, sizeof(int), &input_size_val);           // e.g., 16
    clSetKernelArg(conv_kernel, 4, sizeof(int), &kernel_size_val);          // e.g., 3
    clSetKernelArg(conv_kernel, 5, sizeof(int), &conv_out_val);             // e.g., 14


    size_t global_size_conv[2] = { conv_out_val, conv_out_val };
    clEnqueueNDRangeKernel(queue, conv_kernel, 2, NULL, global_size_conv, NULL, 0, NULL, NULL);

    float conv_output[CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE];
    clEnqueueReadBuffer(queue, conv_output_buf, CL_TRUE, 0, sizeof(conv_output), conv_output, 0, NULL, NULL);
    printf("Convolution output before maxpool:\n");
    for (int i = 0; i < CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE; ++i)
        printf("%.3f ", conv_output[i]);
    printf("\n");

    cl_kernel pool_kernel = clCreateKernel(program, "maxpool2d", &err);
    clSetKernelArg(pool_kernel, 0, sizeof(cl_mem), &conv_output_buf);
    clSetKernelArg(pool_kernel, 1, sizeof(cl_mem), &pool_output_buf);
    clSetKernelArg(pool_kernel, 2, sizeof(int), &conv_out_val);
    clSetKernelArg(pool_kernel, 3, sizeof(int), &pool_size_val);

    size_t global_size_pool[2] = { POOL_OUTPUT_SIZE, POOL_OUTPUT_SIZE };
    clEnqueueNDRangeKernel(queue, pool_kernel, 2, NULL, global_size_pool, NULL, 0, NULL, NULL);

    float maxpool_output[POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE];
    clEnqueueReadBuffer(queue, pool_output_buf, CL_TRUE, 0,
        sizeof(maxpool_output), maxpool_output, 0, NULL, NULL);
    printf("Maxpool output before dense layer:\n");
    for (int i = 0; i < POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE; ++i)
        printf("%.3f ", maxpool_output[i]);
    printf("\n");

    cl_kernel dense_kernel = clCreateKernel(program, "dense_layer", &err);
    if (err != CL_SUCCESS) printf("dense_layer kernel creation error: %d\n", err);

    clSetKernelArg(dense_kernel, 0, sizeof(cl_mem), &pool_output_buf);
    clSetKernelArg(dense_kernel, 1, sizeof(cl_mem), &dense_weights_buf);
    clSetKernelArg(dense_kernel, 2, sizeof(cl_mem), &dense_biases_buf);
    clSetKernelArg(dense_kernel, 3, sizeof(cl_mem), &dense_output_buf);
    clSetKernelArg(dense_kernel, 4, sizeof(int), &dense_in_val);
    clSetKernelArg(dense_kernel, 5, sizeof(int), &dense_out_val);

    size_t global_size_dense = DENSE_OUTPUT_SIZE;
    clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &global_size_dense, NULL, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &global_size_dense, NULL, 0, NULL, NULL);


    printf("Dense biases after kernel:\n");
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        printf("%.3f ", dense_biases[i]);
    printf("\n");


    cl_kernel softmax_kernel = clCreateKernel(program, "softmax_parallel", &err);
    if (err != CL_SUCCESS) printf("softmax_kernel creation error: %d\n", err);

    clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &dense_output_buf);
    clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &softmax_output_buf);
    clSetKernelArg(softmax_kernel, 2, sizeof(int), &dense_out_val);

    float dense_output[DENSE_OUTPUT_SIZE];
    clEnqueueReadBuffer(queue, dense_output_buf, CL_TRUE, 0, dense_output_bytes, dense_output, 0, NULL, NULL);

    printf("Dense layer output before softmax:\n");
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        printf("%.3f ", dense_output[i]);
    printf("\n");

    // clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size_dense, NULL, 0, NULL, NULL);
    size_t local_size = WGSIZE;
    size_t global_size = ((dense_out_val + WGSIZE - 1) / WGSIZE) * WGSIZE;
    clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);


    float softmax_output[DENSE_OUTPUT_SIZE];
    clEnqueueReadBuffer(queue, softmax_output_buf, CL_TRUE, 0,
        dense_output_bytes, softmax_output, 0, NULL, NULL);

    printf("Softmax probabilities:\n");
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        printf("%.3f ", softmax_output[i]);
    printf("\n");

    float sum = 0.0f;
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) sum += softmax_output[i];
    printf("Softmax sum: %.6f\n", sum);  // Should be almost exactly 1.0

    // -- After printing softmax outputs and softmax sum --

    // 1. Create buffers for gradients if you haven't yet (can be reused)
    cl_mem grad_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * DENSE_OUTPUT_SIZE, NULL, &err);
    if (err != CL_SUCCESS) printf("grad_output_buf creation error: %d\n", err);

    cl_mem grad_input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * DENSE_INPUT_SIZE, NULL, &err);
    if (err != CL_SUCCESS) printf("grad_input_buf creation error: %d\n", err);

    // 2. Choose a label and learning rate
    int label = 2; // Use 0-9 for your test data
    float learning_rate = 0.01f;

    // 3. Create the dense backward kernel
    cl_kernel dense_backward_kernel = clCreateKernel(program, "dense_backward", &err);
    if (err != CL_SUCCESS) printf("dense_backward_kernel creation failed: %d\n", err);

    // 4. Call your backprop function
    err = dense_layer_backprop(
        context,
        queue,
        dense_backward_kernel,
        pool_output_buf,         // input to dense layer (OpenCL buffer)
        dense_weights_buf,       // weights (OpenCL buffer)
        dense_biases_buf,        // biases (OpenCL buffer)
        grad_output_buf,         // grad_output (OpenCL buffer)
        grad_input_buf,          // grad_input (OpenCL buffer)
        softmax_output,          // host pointer (result from softmax)
        label,
        DENSE_INPUT_SIZE,
        DENSE_OUTPUT_SIZE,
        learning_rate
    );
    if (err != CL_SUCCESS) printf("dense_layer_backprop failed: %d\n", err);

    // (Optional) Read back weights, biases, or grad_input to print/inspect after update!
    // Example: read weights back to host and print

    float updated_dense_weights[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
    clEnqueueReadBuffer(queue, dense_weights_buf, CL_TRUE, 0,
        sizeof(updated_dense_weights), updated_dense_weights, 0, NULL, NULL);
    printf("First updated weights: %.3f %.3f\n", updated_dense_weights[0], updated_dense_weights[1]);


    clReleaseMemObject(grad_output_buf);
    clReleaseMemObject(grad_input_buf);
    clReleaseKernel(dense_backward_kernel);


    clReleaseMemObject(input_buf);
    clReleaseMemObject(kernel_buf);
    clReleaseMemObject(conv_output_buf);
    clReleaseMemObject(pool_output_buf);
    clReleaseMemObject(dense_weights_buf);
    clReleaseMemObject(dense_biases_buf);
    clReleaseMemObject(dense_output_buf);
    clReleaseMemObject(softmax_output_buf);

    clReleaseKernel(conv_kernel);
    clReleaseKernel(pool_kernel);
    clReleaseKernel(dense_kernel);
    clReleaseKernel(softmax_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
