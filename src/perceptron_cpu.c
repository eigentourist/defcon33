#include <stdio.h>
#include <stdlib.h>

#define NUM_INPUTS 2
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.1
#define EPOCHS 20

// Step activation
int activation(float x) {
    return (x >= 0.0f) ? 1 : 0;
}

int main() {
    // Truth table for AND, OR, NAND
    float inputs[NUM_SAMPLES][NUM_INPUTS] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    int targets[NUM_SAMPLES] = {0, 0, 0, 1}; // AND
    // int targets[NUM_SAMPLES] = {0, 1, 1, 1}; // OR
    // int targets[NUM_SAMPLES] = {1, 1, 1, 0}; // NAND

    // Perceptron weights and bias
    float weights[NUM_INPUTS] = {0.0, 0.0};
    float bias = 0.0;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        printf("Epoch %d\n", epoch+1);
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            float sum = bias;
            for (int j = 0; j < NUM_INPUTS; ++j)
                sum += inputs[i][j] * weights[j];
            int output = activation(sum);
            int error = targets[i] - output;

            // Update weights and bias
            for (int j = 0; j < NUM_INPUTS; ++j)
                weights[j] += LEARNING_RATE * error * inputs[i][j];
            bias += LEARNING_RATE * error;

            printf("Input: [%g, %g], Output: %d, Target: %d\n",
                   inputs[i][0], inputs[i][1], output, targets[i]);
        }
    }
    printf("\nTrained weights: %g %g\nTrained bias: %g\n", weights[0], weights[1], bias);

    // Test final perceptron
    printf("\nTesting perceptron...\n");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float sum = bias;
        for (int j = 0; j < NUM_INPUTS; ++j)
            sum += inputs[i][j] * weights[j];
        int output = activation(sum);
        printf("Input: [%g, %g], Output: %d\n",
               inputs[i][0], inputs[i][1], output);
    }

    return 0;
}
