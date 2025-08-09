#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>

#define N 256 // Same as CPU version

// Load kernel source from file
char* load_kernel_source(const char* filename, size_t* out_size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Failed to open kernel file: %s\n", filename); exit(1); }
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp); rewind(fp);
    char* buffer = (char*)malloc(filesize + 1);
    fread(buffer, 1, filesize, fp);
    buffer[filesize] = '\0'; fclose(fp);
    if (out_size) *out_size = filesize;
    return buffer;
}

void fill_rand(float* M, int n) {
    for (int i = 0; i < n * n; ++i)
        M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    srand((unsigned int)time(NULL));

    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));
    fill_rand(A, N);
    fill_rand(B, N);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    size_t kernel_size;
    char* kernelSource = load_kernel_source("src/matrix_mult_kernel.cl", &kernel_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernel_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        free(kernelSource);
        return 1;
    }
    kernel = clCreateKernel(program, "matmul", &err);

    // Buffers
    cl_mem buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    cl_mem buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    cl_mem buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);

    clEnqueueWriteBuffer(queue, buf_A, CL_TRUE, 0, N * N * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_B, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL);

    // Set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_C);
    int n = N;
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Timing the GPU computation only (not OpenCL setup/teardown)
    clock_t t0 = clock();

    // Launch
    size_t global[2] = {N, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    clFinish(queue);

    clock_t t1 = clock();

    // Download result
    clEnqueueReadBuffer(queue, buf_C, CL_TRUE, 0, N * N * sizeof(float), C, 0, NULL, NULL);

    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Matrix multiplication (GPU/OpenCL): %dx%d x %dx%d\n", N, N, N, N);
    printf("Elapsed time (GPU only): %.4f seconds\n", elapsed);
    printf("C[0][0]=%.3f  C[N/2][N/2]=%.3f  C[N-1][N-1]=%.3f\n",
           C[0], C[(N/2)*N + (N/2)], C[N*N-1]);

    // Cleanup
    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernelSource);
    free(A); free(B); free(C);
    return 0;
}
