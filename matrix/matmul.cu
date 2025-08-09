#include "matmul.cuh"

extern "C"
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // get_global_id(0) -> row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // get_global_id(1) -> col
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

#ifndef TILE
#define TILE 16
#endif

extern "C"
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Number of tiles along K dimension
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Guarded loads into shared memory
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ---- launchers ----
extern "C" void matmul_naive_launch(const float* dA, const float* dB, float* dC, int N,
                                    cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, N);
}

extern "C" void matmul_tiled_launch(const float* dA, const float* dB, float* dC, int N,
                                    cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, N);
}

