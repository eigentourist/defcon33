// perceptron_cuda_host.cpp — CUDA perceptron host with CLI flags for AND/OR/NAND
#include "perceptron_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define NUM_INPUTS   2
#define NUM_SAMPLES  4

static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

static void ascii_bar(float err) {
    int len = (int)(err * 60 + 0.5f);
    std::printf(" [");
    for (int i = 0; i < len; ++i) std::printf("#");
    for (int i = len; i < 60; ++i) std::printf(" ");
    std::printf("]\n");
}

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [--and|--or|--nand] [--epochs N] [--lr F]\n"
        "  --and    Train on AND gate (default)\n"
        "  --or     Train on OR gate\n"
        "  --nand   Train on NAND gate\n"
        "  --epochs Number of epochs (default 20)\n"
        "  --lr     Learning rate (default 0.1)\n", prog);
}

int main(int argc, char** argv) {
    // Defaults (match your original)
    int targets[NUM_SAMPLES] = {0, 0, 0, 1}; // AND
    const char* gate_name = "AND";
    int EPOCHS = 20;
    float LEARNING_RATE = 0.1f;

    // Parse CLI
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--and"))  { int tmp[4]={0,0,0,1}; std::memcpy(targets,tmp,sizeof(tmp)); gate_name="AND"; }
        else if (!std::strcmp(argv[i], "--or"))   { int tmp[4]={0,1,1,1}; std::memcpy(targets,tmp,sizeof(tmp)); gate_name="OR"; }
        else if (!std::strcmp(argv[i], "--nand")) { int tmp[4]={1,1,1,0}; std::memcpy(targets,tmp,sizeof(tmp)); gate_name="NAND"; }
        else if (!std::strcmp(argv[i], "--epochs")) {
            if (i+1 >= argc) { usage(argv[0]); return 1; }
            EPOCHS = std::atoi(argv[++i]);
        }
        else if (!std::strcmp(argv[i], "--lr")) {
            if (i+1 >= argc) { usage(argv[0]); return 1; }
            LEARNING_RATE = std::atof(argv[++i]);
        }
        else {
            usage(argv[0]); return 1;
        }
    }

    // Inputs
    float inputs[NUM_SAMPLES][NUM_INPUTS] = { {0,0}, {0,1}, {1,0}, {1,1} };
    float inputs_flat[NUM_SAMPLES * NUM_INPUTS];
    for (int i = 0; i < NUM_SAMPLES; ++i)
        for (int j = 0; j < NUM_INPUTS; ++j)
            inputs_flat[i * NUM_INPUTS + j] = inputs[i][j];

    float weights[NUM_INPUTS] = {0.0f, 0.0f};
    float bias = 0.0f;

    // Device buffers
    float *d_inputs = nullptr, *d_weights = nullptr;
    int   *d_outputs = nullptr;
    check(cudaMalloc(&d_inputs,  sizeof(inputs_flat)), "cudaMalloc d_inputs");
    check(cudaMalloc(&d_weights, sizeof(weights)),     "cudaMalloc d_weights");
    check(cudaMalloc(&d_outputs, sizeof(int) * NUM_SAMPLES), "cudaMalloc d_outputs");

    check(cudaMemcpy(d_inputs, inputs_flat, sizeof(inputs_flat), cudaMemcpyHostToDevice), "H2D inputs");

    std::printf("\nGate: %s | epochs=%d | lr=%.3f\n", gate_name, EPOCHS, LEARNING_RATE);
    std::printf("Epoch   Error   Learning Progress\n");
    std::printf("==========================================\n");

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Push weights; bias passed by value
        check(cudaMemcpy(d_weights, weights, sizeof(weights), cudaMemcpyHostToDevice), "H2D weights");

        // Forward pass on GPU
        perceptron_forward_launch(d_inputs, d_weights, bias, d_outputs, NUM_SAMPLES, NUM_INPUTS);
        check(cudaGetLastError(), "kernel launch");

        // Pull predictions
        int h_outputs[NUM_SAMPLES] = {0};
        check(cudaMemcpy(h_outputs, d_outputs, sizeof(h_outputs), cudaMemcpyDeviceToHost), "D2H outputs");

        // Perceptron rule (CPU update)
        int sum_error = 0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            int error = targets[i] - h_outputs[i];
            sum_error += std::abs(error);
            for (int j = 0; j < NUM_INPUTS; ++j)
                weights[j] += LEARNING_RATE * error * inputs[i][j];
            bias += LEARNING_RATE * error;
        }

        float norm_err = sum_error / (float)NUM_SAMPLES;
        std::printf(" %2d    %.3f  ", epoch + 1, norm_err);
        ascii_bar(norm_err / 2.0f);
    }

    std::printf("\nTrained weights: %.2f %.2f\nTrained bias: %.2f\n\n", weights[0], weights[1], bias);

    // Test (CPU)
    std::printf("Testing perceptron after training:\n");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float sum = bias;
        for (int j = 0; j < NUM_INPUTS; ++j) sum += inputs[i][j] * weights[j];
        int output = (sum >= 0.0f) ? 1 : 0;
        int target = targets[i];
        std::printf("Input: [%g, %g], Output: %d, Target: %d\n",
                    inputs[i][0], inputs[i][1], output, target);
    }

    cudaFree(d_inputs); cudaFree(d_weights); cudaFree(d_outputs);
    return 0;
}

