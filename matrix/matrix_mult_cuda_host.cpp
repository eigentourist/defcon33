// CUDA host equivalent of matrix_mult_opencl.c
#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifndef N
#define N 256   // Same default as the CPU/OpenCL version
#endif

static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void fill_rand(float* M, int n) {
    for (int i = 0; i < n * n; ++i)
        M[i] = float(std::rand()) / RAND_MAX * 2.0f - 1.0f;
}

int main() {
    std::srand(unsigned(std::time(nullptr)));

    const size_t bytes = size_t(N) * N * sizeof(float);

    // Host buffers
    float* A = (float*)std::malloc(bytes);
    float* B = (float*)std::malloc(bytes);
    float* C = (float*)std::malloc(bytes);
    if (!A || !B || !C) { std::fprintf(stderr, "malloc failed\n"); return 1; }

    fill_rand(A, N);
    fill_rand(B, N);

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    check(cudaMalloc(&dA, bytes), "cudaMalloc dA");
    check(cudaMalloc(&dB, bytes), "cudaMalloc dB");
    check(cudaMalloc(&dC, bytes), "cudaMalloc dC");

    check(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice), "H2D B");

    // ---- GPU timing (kernel only) ----
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop),  "event create stop");

    check(cudaEventRecord(start), "event record start");

    // Launch (choose one)
    // matmul_naive_launch(dA, dB, dC, N);
    matmul_tiled_launch(dA, dB, dC, N);

    check(cudaGetLastError(), "kernel launch");
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    // ----------------------------------

    check(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost), "D2H C");

    std::printf("Matrix multiplication (GPU/CUDA): %dx%d x %dx%d\n", N, N, N, N);
    std::printf("Elapsed time (GPU only): %.4f seconds\n", ms / 1000.0f);
    std::printf("C[0][0]=%.3f  C[N/2][N/2]=%.3f  C[N-1][N-1]=%.3f\n",
                C[0], C[(N/2)*N + (N/2)], C[N*N - 1]);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    std::free(A); std::free(B); std::free(C);
    return 0;
}

