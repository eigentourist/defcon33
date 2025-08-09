#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.5
#define EPOCHS 5000

// Sigmoid activation and derivative
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
float sigmoid_deriv(float y) {
    // y = sigmoid(x)
    return y * (1.0f - y);
}

// XOR dataset
float inputs[NUM_SAMPLES][NUM_INPUTS] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};
float targets[NUM_SAMPLES][NUM_OUTPUTS] = {
    {0}, {1}, {1}, {0}
};

int main() {
    srand((unsigned int)time(NULL));

    // Initialize weights and biases with small random values
    float wh[NUM_INPUTS][NUM_HIDDEN];   // input-to-hidden
    float bh[NUM_HIDDEN];               // hidden biases
    float wo[NUM_HIDDEN][NUM_OUTPUTS];  // hidden-to-output
    float bo[NUM_OUTPUTS];              // output biases

    for(int i=0; i<NUM_INPUTS; ++i)
        for(int j=0; j<NUM_HIDDEN; ++j)
            wh[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2;

    for(int j=0; j<NUM_HIDDEN; ++j)
        bh[j] = ((float)rand()/RAND_MAX - 0.5f) * 2;

    for(int j=0; j<NUM_HIDDEN; ++j)
        for(int k=0; k<NUM_OUTPUTS; ++k)
            wo[j][k] = ((float)rand()/RAND_MAX - 0.5f) * 2;

    for(int k=0; k<NUM_OUTPUTS; ++k)
        bo[k] = ((float)rand()/RAND_MAX - 0.5f) * 2;

    // Training loop
    for(int epoch=0; epoch<EPOCHS; ++epoch) {
        float total_error = 0.0f;

        for(int n=0; n<NUM_SAMPLES; ++n) {
            // 1. Forward pass
            float h[NUM_HIDDEN], ho[NUM_HIDDEN]; // hidden linear, hidden activation
            for(int j=0; j<NUM_HIDDEN; ++j) {
                h[j] = bh[j];
                for(int i=0; i<NUM_INPUTS; ++i)
                    h[j] += inputs[n][i] * wh[i][j];
                ho[j] = sigmoid(h[j]);
            }

            float o[NUM_OUTPUTS], oo[NUM_OUTPUTS]; // output linear, output activation
            for(int k=0; k<NUM_OUTPUTS; ++k) {
                o[k] = bo[k];
                for(int j=0; j<NUM_HIDDEN; ++j)
                    o[k] += ho[j] * wo[j][k];
                oo[k] = sigmoid(o[k]);
            }

            // 2. Error (squared error)
            for(int k=0; k<NUM_OUTPUTS; ++k) {
                float e = targets[n][k] - oo[k];
                total_error += e * e;
            }

            // 3. Backward pass (output to hidden)
            float d_oo[NUM_OUTPUTS]; // delta for output neurons
            for(int k=0; k<NUM_OUTPUTS; ++k)
                d_oo[k] = (targets[n][k] - oo[k]) * sigmoid_deriv(oo[k]);

            float d_ho[NUM_HIDDEN]; // delta for hidden neurons
            for(int j=0; j<NUM_HIDDEN; ++j) {
                float sum = 0.0f;
                for(int k=0; k<NUM_OUTPUTS; ++k)
                    sum += d_oo[k] * wo[j][k];
                d_ho[j] = sum * sigmoid_deriv(ho[j]);
            }

            // 4. Update weights and biases
            for(int k=0; k<NUM_OUTPUTS; ++k) {
                bo[k] += LEARNING_RATE * d_oo[k];
                for(int j=0; j<NUM_HIDDEN; ++j)
                    wo[j][k] += LEARNING_RATE * d_oo[k] * ho[j];
            }
            for(int j=0; j<NUM_HIDDEN; ++j) {
                bh[j] += LEARNING_RATE * d_ho[j];
                for(int i=0; i<NUM_INPUTS; ++i)
                    wh[i][j] += LEARNING_RATE * d_ho[j] * inputs[n][i];
            }
        }

        if((epoch+1) % 500 == 0)
            printf("Epoch %d | MSE: %.5f\n", epoch+1, total_error / NUM_SAMPLES);
    }

    // Test the trained network
    printf("\nTest results (after training):\n");
    for(int n=0; n<NUM_SAMPLES; ++n) {
        float h[NUM_HIDDEN], ho[NUM_HIDDEN];
        for(int j=0; j<NUM_HIDDEN; ++j) {
            h[j] = bh[j];
            for(int i=0; i<NUM_INPUTS; ++i)
                h[j] += inputs[n][i] * wh[i][j];
            ho[j] = sigmoid(h[j]);
        }
        float o[NUM_OUTPUTS], oo[NUM_OUTPUTS];
        for(int k=0; k<NUM_OUTPUTS; ++k) {
            o[k] = bo[k];
            for(int j=0; j<NUM_HIDDEN; ++j)
                o[k] += ho[j] * wo[j][k];
            oo[k] = sigmoid(o[k]);
        }
        printf("Input: [%g, %g] => Output: %.3f (Target: %.1f)\n",
                inputs[n][0], inputs[n][1], oo[0], targets[n][0]);
    }

    return 0;
}
