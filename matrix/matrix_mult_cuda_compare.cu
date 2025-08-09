// matrix_mult_cuda_compare.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 256 // Try changing to 1024 during workshop

__global__ void matmul_cuda(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

void fill_rand(float* M, int n) {
    for (int i = 0; i < n * n; ++i)
        M[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    srand((unsigned int)time(NULL));

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    fill_rand(h_A, N);
    fill_rand(h_B, N);

    // ===== CPU TIMING =====
    clock_t cpu_start = clock();
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // ===== GPU TIMING =====
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_cuda<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // ===== Results =====
    printf("Matrix multiplication (%dx%d)\n", N, N);
    printf("CPU time: %.4f s\n", cpu_time);
    printf("GPU time: %.4f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", (cpu_time * 1000.0) / gpu_time);
    printf("Sample: C[0][0]=%.3f  C[N/2][N/2]=%.3f  C[N-1][N-1]=%.3f\n",
           h_C_gpu[0], h_C_gpu[(N/2)*N + (N/2)], h_C_gpu[N*N-1]);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

