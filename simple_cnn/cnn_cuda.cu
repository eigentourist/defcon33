// cnn_cuda.cu — CUDA port of your CPU-only toy CNN (single conv + MLP)
// Forward pass on GPU; weight updates remain on CPU (same as original)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define INPUT_SIZE 16
#define KERNEL_SIZE 3
#define CONV_OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)   // 14
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (CONV_OUTPUT_SIZE / POOL_SIZE)   // 7
#define FC_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE) // 49
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 2
#define LEARNING_RATE 0.01f
#define EPOCHS 100
#define NUM_SAMPLES 4

// --------------------------- CPU helpers ---------------------------
static inline float relu_host(float x) { return x > 0.f ? x : 0.f; }
static inline float drelu_host(float x) { return x > 0.f ? 1.f : 0.f; }

static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// --------------------------- Data & weights (host) ---------------------------
static float kernel_w[KERNEL_SIZE][KERNEL_SIZE];
static float bias_conv = 0.0f;
static float weights1[HIDDEN_SIZE][FC_INPUT_SIZE];
static float bias1[HIDDEN_SIZE];
static float weights2[OUTPUT_SIZE][HIDDEN_SIZE];
static float bias2[OUTPUT_SIZE];

static float dataset[NUM_SAMPLES][INPUT_SIZE][INPUT_SIZE];
static int   labels[NUM_SAMPLES];

static void create_fake_dataset() {
    std::memset(dataset, 0, sizeof(dataset));
    // 0: X -> 1
    for (int i = 0; i < INPUT_SIZE; i++) {
        dataset[0][i][i] = 1.0f;
        dataset[0][i][INPUT_SIZE - 1 - i] = 1.0f;
    }
    labels[0] = 1;
    // 1: O -> 0
    for (int i = 3; i < 13; i++) {
        dataset[1][3][i] = 1.0f;
        dataset[1][12][i] = 1.0f;
        dataset[1][i][3] = 1.0f;
        dataset[1][i][12] = 1.0f;
    }
    labels[1] = 0;
    // 2: vertical bar -> 1
    for (int i = 0; i < INPUT_SIZE; i++) dataset[2][i][7] = 1.0f;
    labels[2] = 1;
    // 3: horizontal bar -> 0
    for (int i = 0; i < INPUT_SIZE; i++) dataset[3][7][i] = 1.0f;
    labels[3] = 0;
}

static void initialize_weights() {
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            kernel_w[i][j] = ((float)rand() / RAND_MAX - 0.5f);

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

// --------------------------- Device kernels ---------------------------
__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }

__global__ void conv2d_relu_kernel(const float* __restrict__ in, // 16x16
                                   const float* __restrict__ k3, // 3x3
                                   float b, float* __restrict__ out) { // 14x14
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (x >= CONV_OUTPUT_SIZE || y >= CONV_OUTPUT_SIZE) return;

    float sum = 0.f;
    #pragma unroll
    for (int ky = 0; ky < KERNEL_SIZE; ++ky)
        #pragma unroll
        for (int kx = 0; kx < KERNEL_SIZE; ++kx)
            sum += in[(y + ky) * INPUT_SIZE + (x + kx)] * k3[ky * KERNEL_SIZE + kx];

    out[y * CONV_OUTPUT_SIZE + x] = relu(sum + b);
}

__global__ void maxpool2x2_kernel(const float* __restrict__ in,  // 14x14
                                  float* __restrict__ out) {     // 7x7
    int x = blockIdx.x * blockDim.x + threadIdx.x; // col in pooled
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row in pooled
    if (x >= POOL_OUTPUT_SIZE || y >= POOL_OUTPUT_SIZE) return;

    int ix = x * POOL_SIZE;
    int iy = y * POOL_SIZE;
    float m = -INFINITY;
    #pragma unroll
    for (int dy = 0; dy < POOL_SIZE; ++dy)
        #pragma unroll
        for (int dx = 0; dx < POOL_SIZE; ++dx) {
            float v = in[(iy + dy) * CONV_OUTPUT_SIZE + (ix + dx)];
            m = v > m ? v : m;
        }
    out[y * POOL_OUTPUT_SIZE + x] = m;
}

__global__ void flatten_kernel(const float* __restrict__ in7x7, float* __restrict__ out49) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < FC_INPUT_SIZE) out49[i] = in7x7[i];
}

// weights1: [HIDDEN_SIZE][FC_INPUT_SIZE] row-major
__global__ void dense_relu_kernel(const float* __restrict__ x49,
                                  const float* __restrict__ w1, // HIDDEN_SIZE*FC_INPUT_SIZE
                                  const float* __restrict__ b1, // HIDDEN_SIZE
                                  float* __restrict__ h16) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden idx
    if (j >= HIDDEN_SIZE) return;
    const float* wrow = w1 + j * FC_INPUT_SIZE;
    float s = b1[j];
    #pragma unroll
    for (int k = 0; k < FC_INPUT_SIZE; ++k) s += wrow[k] * x49[k];
    h16[j] = relu(s);
}

// weights2: [OUTPUT_SIZE][HIDDEN_SIZE] row-major
__global__ void dense_linear_kernel(const float* __restrict__ h16,
                                    const float* __restrict__ w2, // OUTPUT_SIZE*HIDDEN_SIZE
                                    const float* __restrict__ b2, // OUTPUT_SIZE
                                    float* __restrict__ logits2) {
    int o = threadIdx.x; // OUTPUT_SIZE is tiny; use 1 block
    if (o >= OUTPUT_SIZE) return;
    const float* wrow = w2 + o * HIDDEN_SIZE;
    float s = b2[o];
    #pragma unroll
    for (int j = 0; j < HIDDEN_SIZE; ++j) s += wrow[j] * h16[j];
    logits2[o] = s;
}

__global__ void softmax_kernel(const float* __restrict__ logits2,
                               float* __restrict__ probs2) {
    // OUTPUT_SIZE==2 small path
    float a = logits2[0], b = logits2[1];
    float m = a > b ? a : b;
    float ea = expf(a - m), eb = expf(b - m);
    float s = ea + eb;
    probs2[0] = ea / s;
    probs2[1] = eb / s;
}

// --------------------------- Host training (uses device forward) ---------------------------
static void train_cuda() {
    // Device buffers (single-sample pipeline)
    float *d_input = nullptr, *d_k3 = nullptr, *d_conv = nullptr, *d_pool = nullptr;
    float *d_flat = nullptr, *d_hidden = nullptr, *d_logits = nullptr, *d_probs = nullptr;
    float *d_w1 = nullptr, *d_b1 = nullptr, *d_w2 = nullptr, *d_b2 = nullptr;

    check(cudaMalloc(&d_input,  INPUT_SIZE*INPUT_SIZE*sizeof(float)), "cudaMalloc input");
    check(cudaMalloc(&d_k3,     KERNEL_SIZE*KERNEL_SIZE*sizeof(float)), "cudaMalloc k3");
    check(cudaMalloc(&d_conv,   CONV_OUTPUT_SIZE*CONV_OUTPUT_SIZE*sizeof(float)), "cudaMalloc conv");
    check(cudaMalloc(&d_pool,   POOL_OUTPUT_SIZE*POOL_OUTPUT_SIZE*sizeof(float)), "cudaMalloc pool");
    check(cudaMalloc(&d_flat,   FC_INPUT_SIZE*sizeof(float)), "cudaMalloc flat");
    check(cudaMalloc(&d_hidden, HIDDEN_SIZE*sizeof(float)), "cudaMalloc hidden");
    check(cudaMalloc(&d_logits, OUTPUT_SIZE*sizeof(float)), "cudaMalloc logits");
    check(cudaMalloc(&d_probs,  OUTPUT_SIZE*sizeof(float)), "cudaMalloc probs");

    check(cudaMalloc(&d_w1, HIDDEN_SIZE*FC_INPUT_SIZE*sizeof(float)), "cudaMalloc w1");
    check(cudaMalloc(&d_b1, HIDDEN_SIZE*sizeof(float)), "cudaMalloc b1");
    check(cudaMalloc(&d_w2, OUTPUT_SIZE*HIDDEN_SIZE*sizeof(float)), "cudaMalloc w2");
    check(cudaMalloc(&d_b2, OUTPUT_SIZE*sizeof(float)), "cudaMalloc b2");

    // Copy the static conv kernel/bias once (we don't update them in your original)
    float h_k3[KERNEL_SIZE*KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; ++i)
        for (int j = 0; j < KERNEL_SIZE; ++j)
            h_k3[i*KERNEL_SIZE + j] = kernel_w[i][j];
    check(cudaMemcpy(d_k3, h_k3, sizeof(h_k3), cudaMemcpyHostToDevice), "H2D k3");

    dim3 b2d(16,16);
    dim3 g2d((CONV_OUTPUT_SIZE + b2d.x - 1)/b2d.x,
             (CONV_OUTPUT_SIZE + b2d.y - 1)/b2d.y);
    dim3 bpool(16,16);
    dim3 gpool((POOL_OUTPUT_SIZE + bpool.x - 1)/bpool.x,
               (POOL_OUTPUT_SIZE + bpool.y - 1)/bpool.y);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        int correct = 0;

        for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
            // Push weights (they change every sample in your loop)
            check(cudaMemcpy(d_w1, weights1, sizeof(weights1), cudaMemcpyHostToDevice), "H2D w1");
            check(cudaMemcpy(d_b1, bias1,    sizeof(bias1),    cudaMemcpyHostToDevice), "H2D b1");
            check(cudaMemcpy(d_w2, weights2, sizeof(weights2), cudaMemcpyHostToDevice), "H2D w2");
            check(cudaMemcpy(d_b2, bias2,    sizeof(bias2),    cudaMemcpyHostToDevice), "H2D b2");

            // Input
            check(cudaMemcpy(d_input, dataset[sample], INPUT_SIZE*INPUT_SIZE*sizeof(float),
                             cudaMemcpyHostToDevice), "H2D input");

            // Forward on device
            conv2d_relu_kernel<<<g2d, b2d>>>(d_input, d_k3, bias_conv, d_conv);
            check(cudaGetLastError(), "conv2d launch");

            maxpool2x2_kernel<<<gpool, bpool>>>(d_conv, d_pool);
            check(cudaGetLastError(), "maxpool launch");

            int b1 = 64;
            flatten_kernel<<<(FC_INPUT_SIZE + b1 - 1)/b1, b1>>>(d_pool, d_flat);
            check(cudaGetLastError(), "flatten launch");

            dense_relu_kernel<<<(HIDDEN_SIZE + 255)/256, 256>>>(d_flat, d_w1, d_b1, d_hidden);
            check(cudaGetLastError(), "dense1 launch");

            dense_linear_kernel<<<1, OUTPUT_SIZE>>>(d_hidden, d_w2, d_b2, d_logits);
            check(cudaGetLastError(), "dense2 launch");

            softmax_kernel<<<1, 1>>>(d_logits, d_probs);
            check(cudaGetLastError(), "softmax launch");

            // Pull probs + intermediates needed for CPU grads
            float probs[OUTPUT_SIZE], hidden[HIDDEN_SIZE], flat[FC_INPUT_SIZE];
            check(cudaMemcpy(probs,  d_probs,  sizeof(probs),  cudaMemcpyDeviceToHost), "D2H probs");
            check(cudaMemcpy(hidden, d_hidden, sizeof(hidden), cudaMemcpyDeviceToHost), "D2H hidden");
            check(cudaMemcpy(flat,   d_flat,   sizeof(flat),   cudaMemcpyDeviceToHost), "D2H flat");

            int target = labels[sample];
            if ((probs[0] < probs[1]) == target) correct++;

            // --- CPU gradient & weight update (same math as your original) ---
            float dlogits[OUTPUT_SIZE];
            for (int i = 0; i < OUTPUT_SIZE; ++i)
                dlogits[i] = probs[i] - (i == target ? 1.0f : 0.0f);

            for (int i = 0; i < OUTPUT_SIZE; ++i) {
                for (int j = 0; j < HIDDEN_SIZE; ++j) {
                    weights2[i][j] -= LEARNING_RATE * dlogits[i] * hidden[j];
                }
                bias2[i] -= LEARNING_RATE * dlogits[i];
            }

            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                float grad = 0.0f;
                for (int i = 0; i < OUTPUT_SIZE; ++i) grad += dlogits[i] * weights2[i][j];
                grad *= drelu_host(hidden[j]);
                for (int k = 0; k < FC_INPUT_SIZE; ++k) {
                    weights1[j][k] -= LEARNING_RATE * grad * flat[k];
                }
                bias1[j] -= LEARNING_RATE * grad;
            }
        }

        std::printf("Epoch %d: Accuracy = %.2f%%\n", epoch + 1, 100.0f * correct / NUM_SAMPLES);
    }

    cudaFree(d_input); cudaFree(d_k3); cudaFree(d_conv); cudaFree(d_pool);
    cudaFree(d_flat); cudaFree(d_hidden); cudaFree(d_logits); cudaFree(d_probs);
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2);
}

// --------------------------- main ---------------------------
int main() {
    srand(1234);
    initialize_weights();
    create_fake_dataset();
    std::puts("Training CNN on fake dataset (CUDA forward, CPU updates)...");
    train_cuda();
    return 0;
}

