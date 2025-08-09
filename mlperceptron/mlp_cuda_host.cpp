// mlp_cuda_host.cpp — CUDA host for forward pass of a tiny MLP (XOR demo)
#include "mlp_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static float frand() {
    return float(rand()) / RAND_MAX * 2.0f - 1.0f; // [-1,1]
}

int main() {
    srand((unsigned)time(nullptr));

    // Match the CPU example
    const int NUM_INPUTS  = 2;
    const int NUM_HIDDEN  = 2;
    const int NUM_OUTPUTS = 1;
    const int NUM_SAMPLES = 4;

    // XOR data (row-major: samples x inputs/outputs)
    const float h_inputs[NUM_SAMPLES * NUM_INPUTS] = {
        0,0,  0,1,  1,0,  1,1
    };

    // Random small weights/biases (same layout as your OpenCL kernels)
    std::vector<float> h_wh(NUM_INPUTS * NUM_HIDDEN);
    std::vector<float> h_bh(NUM_HIDDEN);
    std::vector<float> h_wo(NUM_HIDDEN * NUM_OUTPUTS);
    std::vector<float> h_bo(NUM_OUTPUTS);

    std::generate(h_wh.begin(), h_wh.end(), frand);
    std::generate(h_bh.begin(), h_bh.end(), frand);
    std::generate(h_wo.begin(), h_wo.end(), frand);
    std::generate(h_bo.begin(), h_bo.end(), frand);

    // Device buffers
    float *d_inputs=nullptr, *d_wh=nullptr, *d_bh=nullptr,
          *d_hidden=nullptr, *d_wo=nullptr, *d_bo=nullptr, *d_output=nullptr;

    check(cudaMalloc(&d_inputs,  NUM_SAMPLES * NUM_INPUTS  * sizeof(float)), "cudaMalloc d_inputs");
    check(cudaMalloc(&d_wh,      NUM_INPUTS  * NUM_HIDDEN  * sizeof(float)), "cudaMalloc d_wh");
    check(cudaMalloc(&d_bh,      NUM_HIDDEN                       * sizeof(float)), "cudaMalloc d_bh");
    check(cudaMalloc(&d_hidden,  NUM_SAMPLES * NUM_HIDDEN  * sizeof(float)), "cudaMalloc d_hidden");
    check(cudaMalloc(&d_wo,      NUM_HIDDEN  * NUM_OUTPUTS * sizeof(float)), "cudaMalloc d_wo");
    check(cudaMalloc(&d_bo,      NUM_OUTPUTS                      * sizeof(float)), "cudaMalloc d_bo");
    check(cudaMalloc(&d_output,  NUM_SAMPLES * NUM_OUTPUTS * sizeof(float)), "cudaMalloc d_output");

    // H2D
    check(cudaMemcpy(d_inputs, h_inputs, sizeof(h_inputs), cudaMemcpyHostToDevice), "H2D inputs");
    check(cudaMemcpy(d_wh,     h_wh.data(), h_wh.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D wh");
    check(cudaMemcpy(d_bh,     h_bh.data(), h_bh.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D bh");
    check(cudaMemcpy(d_wo,     h_wo.data(), h_wo.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D wo");
    check(cudaMemcpy(d_bo,     h_bo.data(), h_bo.size()*sizeof(float), cudaMemcpyHostToDevice), "H2D bo");

    // Launch forward passes (same division of labor as the OpenCL kernels)
    mlp_forward_hidden_launch(d_inputs, d_wh, d_bh, d_hidden,
                              NUM_SAMPLES, NUM_INPUTS, NUM_HIDDEN, /*stream*/nullptr);
    check(cudaGetLastError(), "forward_hidden launch");

    mlp_forward_output_launch(d_hidden, d_wo, d_bo, d_output,
                              NUM_SAMPLES, NUM_HIDDEN, NUM_OUTPUTS, /*stream*/nullptr);
    check(cudaGetLastError(), "forward_output launch");

    // D2H & print
    std::vector<float> h_output(NUM_SAMPLES * NUM_OUTPUTS);
    check(cudaMemcpy(h_output.data(), d_output,
                     h_output.size()*sizeof(float), cudaMemcpyDeviceToHost), "D2H output");

    std::puts("CUDA MLP forward (random weights) on XOR:");
    for (int n = 0; n < NUM_SAMPLES; ++n) {
        float x0 = h_inputs[n*NUM_INPUTS+0];
        float x1 = h_inputs[n*NUM_INPUTS+1];
        float y  = h_output[n*NUM_OUTPUTS+0];
        std::printf("  [%.0f, %.0f] -> %.4f\n", x0, x1, y);
    }

    // Cleanup
    cudaFree(d_inputs); cudaFree(d_wh); cudaFree(d_bh);
    cudaFree(d_hidden); cudaFree(d_wo); cudaFree(d_bo); cudaFree(d_output);
    return 0;
}

