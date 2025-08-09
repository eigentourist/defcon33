#include "life_kernel.cuh"

__device__ __forceinline__ int wrap(int v, int n) { v %= n; return v < 0 ? v + n : v; }

extern "C"
__global__ void life_step_kernel(const int* __restrict__ in,
                                 int* __restrict__ out,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int count = 0;
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        #pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = wrap(x + dx, width);
            int ny = wrap(y + dy, height);
            count += in[ny * width + nx];
        }
    }
    int cell = in[idx];
    out[idx] = ((cell == 1) && (count == 2 || count == 3)) || ((cell == 0) && (count == 3)) ? 1 : 0;
}

// Plain C wrapper — choose launch config here
extern "C" void life_step_launch(const int* d_in, int* d_out, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    life_step_kernel<<<grid, block>>>(d_in, d_out, width, height);
}

