// cnn_cpu.c - CNN with single conv layer and MLP classifier (CPU-only)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT_SIZE 16
#define KERNEL_SIZE 3
#define CONV_OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (CONV_OUTPUT_SIZE / POOL_SIZE)
#define FC_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE)
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 2
#define LEARNING_RATE 0.01f
#define EPOCHS 100
#define NUM_SAMPLES 4

float relu(float x) { return fmaxf(0.0f, x); }
float drelu(float x) { return x > 0.0f ? 1.0f : 0.0f; }
float softmax(float* input, float* output, int size) {
    float max = input[0], sum = 0.0f;
    for (int i = 1; i < size; i++) if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) output[i] = expf(input[i] - max);
    for (int i = 0; i < size; i++) sum += output[i];
    for (int i = 0; i < size; i++) output[i] /= sum;
    return sum;
}

// Kernels, biases, and weights
float kernel[KERNEL_SIZE][KERNEL_SIZE];
float bias_conv = 0.0f;
float weights1[HIDDEN_SIZE][FC_INPUT_SIZE];
float bias1[HIDDEN_SIZE];
float weights2[OUTPUT_SIZE][HIDDEN_SIZE];
float bias2[OUTPUT_SIZE];

// Fake dataset: simple 16x16 images with labels 0 or 1
float dataset[NUM_SAMPLES][INPUT_SIZE][INPUT_SIZE];
int labels[NUM_SAMPLES];

void create_fake_dataset() {
    memset(dataset, 0, sizeof(dataset));

    // Sample 0: A centered X pattern -> label 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        dataset[0][i][i] = 1.0f;
        dataset[0][i][INPUT_SIZE - 1 - i] = 1.0f;
    }
    labels[0] = 1;

    // Sample 1: A centered O pattern -> label 0
    for (int i = 3; i < 13; i++) {
        dataset[1][3][i] = 1.0f;
        dataset[1][12][i] = 1.0f;
        dataset[1][i][3] = 1.0f;
        dataset[1][i][12] = 1.0f;
    }
    labels[1] = 0;

    // Sample 2: A vertical bar -> label 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        dataset[2][i][7] = 1.0f;
    }
    labels[2] = 1;

    // Sample 3: A horizontal bar -> label 0
    for (int i = 0; i < INPUT_SIZE; i++) {
        dataset[3][7][i] = 1.0f;
    }
    labels[3] = 0;
}

void initialize_weights() {
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            kernel[i][j] = ((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias1[i] = ((float)rand() / RAND_MAX - 0.5f);
        for (int j = 0; j < FC_INPUT_SIZE; j++)
            weights1[i][j] = ((float)rand() / RAND_MAX - 0.5f);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias2[i] = ((float)rand() / RAND_MAX - 0.5f);
        for (int j = 0; j < HIDDEN_SIZE; j++)
            weights2[i][j] = ((float)rand() / RAND_MAX - 0.5f);
    }
}

void convolve(float input[INPUT_SIZE][INPUT_SIZE], float output[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE]) {
    for (int i = 0; i < CONV_OUTPUT_SIZE; i++) {
        for (int j = 0; j < CONV_OUTPUT_SIZE; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = relu(sum + bias_conv);
        }
    }
}

void pool(float input[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE], float output[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE]) {
    for (int i = 0; i < POOL_OUTPUT_SIZE; i++) {
        for (int j = 0; j < POOL_OUTPUT_SIZE; j++) {
            float max = -INFINITY;
            for (int pi = 0; pi < POOL_SIZE; pi++) {
                for (int pj = 0; pj < POOL_SIZE; pj++) {
                    float val = input[i * POOL_SIZE + pi][j * POOL_SIZE + pj];
                    if (val > max) max = val;
                }
            }
            output[i][j] = max;
        }
    }
}

void flatten(float input[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE], float output[FC_INPUT_SIZE]) {
    for (int i = 0; i < POOL_OUTPUT_SIZE; i++)
        for (int j = 0; j < POOL_OUTPUT_SIZE; j++)
            output[i * POOL_OUTPUT_SIZE + j] = input[i][j];
}

void forward(float image[INPUT_SIZE][INPUT_SIZE], float* output_probs, float* flattened, float* hidden, float* logits) {
    float conv_out[CONV_OUTPUT_SIZE][CONV_OUTPUT_SIZE];
    float pool_out[POOL_OUTPUT_SIZE][POOL_OUTPUT_SIZE];

    convolve(image, conv_out);
    pool(conv_out, pool_out);
    flatten(pool_out, flattened);

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = bias1[i];
        for (int j = 0; j < FC_INPUT_SIZE; j++)
            sum += weights1[i][j] * flattened[j];
        hidden[i] = relu(sum);
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = bias2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += weights2[i][j] * hidden[j];
        logits[i] = sum;
    }
    softmax(logits, output_probs, OUTPUT_SIZE);
}

void train() {
    float flattened[FC_INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float logits[OUTPUT_SIZE];
    float output_probs[OUTPUT_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        for (int sample = 0; sample < NUM_SAMPLES; sample++) {
            forward(dataset[sample], output_probs, flattened, hidden, logits);

            int target = labels[sample];
            if ((output_probs[0] < output_probs[1]) == target)
                correct++;

            float dlogits[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; i++)
                dlogits[i] = output_probs[i] - (i == target ? 1.0f : 0.0f);

            for (int i = 0; i < OUTPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    weights2[i][j] -= LEARNING_RATE * dlogits[i] * hidden[j];
                }
                bias2[i] -= LEARNING_RATE * dlogits[i];
            }

            float dhidden[HIDDEN_SIZE] = {0};
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float grad = 0.0f;
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    grad += dlogits[i] * weights2[i][j];
                }
                grad *= drelu(hidden[j]);
                for (int k = 0; k < FC_INPUT_SIZE; k++) {
                    weights1[j][k] -= LEARNING_RATE * grad * flattened[k];
                }
                bias1[j] -= LEARNING_RATE * grad;
            }
        }
        printf("Epoch %d: Accuracy = %.2f%%\n", epoch + 1, 100.0f * correct / NUM_SAMPLES);
    }
}

int main() {
    initialize_weights();
    create_fake_dataset();
    printf("Training CNN on fake dataset...\n");
    train();
    return 0;
}
