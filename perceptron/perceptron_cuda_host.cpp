#include "perceptron_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>

#define NUM_INPUTS   2
#define NUM_SAMPLES  4
#define LEARNING_RATE 0.1f
#define EPOCHS       20

// Training data (same for all gates)
float inputs[NUM_SAMPLES][NUM_INPUTS] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

// Target sets
int targets_and [NUM_SAMPLES] = {0, 0, 0, 1};
int targets_or  [NUM_SAMPLES] = {0, 1, 1, 1};
int targets_nand[NUM_SAMPLES] = {1, 1, 1, 0};

// Error checking helper
static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void ascii_bar(float err) {
    int len = (int)(err * 60 + 0.5f); // 0..1 -> 0..60 chars
    std::printf(" [");
    for (int i = 0; i < len; ++i) std::printf("#");
    for (int i = len; i < 60; ++i) std::printf(" ");
    std::printf("]\n");
}

int main(int argc, char** argv) {
    // Select gate type from CLI
    const int* targets = targets_and; // default
    std::string gate_name = "AND";

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--and") {
            targets = targets_and;
            gate_name = "AND";
        } else if (arg == "--or") {
            targets = targets_or;
            gate_name = "OR";
        } else if (arg == "--nand") {
            targets = targets_nand;
            gate_name = "NAND";
        } else {
            std::fprintf(stderr, "Usage: %s [--and|--or|--nand]\n", argv[0]);
            return 1;
        }
    }

    std::printf("Training perceptron on %s gate...\n", gate_name.c_str());

    // Flatten inputs (samples x inputs)
    float inputs_flat[NUM_SAMPLES * NUM_INPUTS];
    for (int i = 0; i < NUM_SAMPLES; ++i)
        for (int j = 0; j < NUM_INPUTS; ++j)
            inputs_flat[i * NUM_INPUTS + j] = inputs[i][j];

    float weights[NUM_INPUTS] = {0.0f, 0.0f};
    float bias = 0.0f;

    // --- CUDA setup & buffers ---
    float *d_inputs = nullptr, *d_weights = nullptr;
    int   *d_outputs = nullptr;

    check(cudaMalloc(&d_inputs,  sizeof(inputs_flat)), "cudaMalloc d_inputs");
    check(cudaMalloc(&d_weights, sizeof(weights)),     "cudaMalloc d_weights");
    check(cudaMalloc(&d_outputs, sizeof(int) * NUM_SAMPLES), "cudaMalloc d_outputs");

    check(cudaMemcpy(d_inputs, inputs_flat, sizeof(inputs_flat), cudaMemcpyHostToDevice), "H2D inputs");

    // --- Training loop ---
    std::printf("\nEpoch   Error   Learning Progress\n");
    std::printf("==========================================\n");

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Copy weights for this epoch
        check(cudaMemcpy(d_weights, weights, sizeof(weights), cudaMemcpyHostToDevice), "H2D weights");

        // Launch forward pass
        perceptron_forward_launch(d_inputs, d_weights, bias, d_outputs, NUM_SAMPLES, NUM_INPUTS);
        check(cudaGetLastError(), "perceptron_forward launch");

        // Read outputs
        int outputs[NUM_SAMPLES] = {0};
        check(cudaMemcpy(outputs, d_outputs, sizeof(outputs), cudaMemcpyDeviceToHost), "D2H outputs");

        // CPU update (perceptron rule)
        int sum_error = 0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            int error = targets[i] - outputs[i];
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

    // Test final perceptron (CPU)
    std::printf("Testing perceptron after training:\n");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float sum = bias;
        for (int j = 0; j < NUM_INPUTS; ++j) sum += inputs[i][j] * weights[j];
        int output = (sum >= 0.0f) ? 1 : 0;
        std::printf("Input: [%g, %g], Output: %d, Target: %d\n",
                    inputs[i][0], inputs[i][1], output, targets[i]);
    }

    cudaFree(d_inputs); cudaFree(d_weights); cudaFree(d_outputs);
    return 0;
}

