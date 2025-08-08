
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cnn_kernels.h"


#define INPUT_SIZE 64
#define KERNEL1_SIZE 3
#define KERNEL2_SIZE 3
#define CONV1_KERNELS 8    // Number of filters for 1st layer (e.g., 8)
#define CONV2_KERNELS 16   // Number of filters for 2nd layer (e.g., 16)
#define POOL_SIZE 2

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - KERNEL1_SIZE + 1)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / POOL_SIZE)

#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - KERNEL2_SIZE + 1)
#define POOL2_OUTPUT_SIZE (CONV2_OUTPUT_SIZE / POOL_SIZE)

#define DENSE_INPUT_SIZE (POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * CONV2_KERNELS)
#define DENSE_OUTPUT_SIZE 3 // number of classes

#define BATCH_SIZE 16
#define INPUT_PIXELS (INPUT_SIZE * INPUT_SIZE)

// For parallelized softmax
#define WGSIZE 64

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"



//
// Zero buffer memory utility function
//
void cudaZeroBuffer(float* d_buf, size_t n_floats) {
    cudaMemset(d_buf, 0, n_floats * sizeof(float));
}


//
// Load greyscale image from specified path into supplied buffer,
// using given expected width and height
//
// This function makes use of the stb_image library included above.
//
int load_greyscale_image(const char* path, float* buffer, int expected_w, int expected_h) {
    int w, h, c;
    unsigned char* img = stbi_load(path, &w, &h, &c, 1); // force greyscale
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return 1;
    }
    if (w != expected_w || h != expected_h) {
        fprintf(stderr, "Unexpected size for %s: got %dx%d, expected %dx%d\n", path, w, h, expected_w, expected_h);
        stbi_image_free(img);
        return 1;
    }
    for (int i = 0; i < w * h; ++i)
        buffer[i] = img[i] / 255.0f;
    stbi_image_free(img);
    return 0; // success
}




#define MAX_PATH_LEN 256
#define MAX_SET_SIZE 1000

typedef struct {
    char path[MAX_PATH_LEN];
    int label;
} ImageData;


int load_csv(const char* csv_path, ImageData* examples, int max_examples) {
    FILE* f = fopen(csv_path, "r");
    if (!f) {
        perror("CSV open failed");
        return 0;
    }
    int n = 0;
    char line[512];
    while (fgets(line, sizeof(line), f) && n < max_examples) {
        char* comma = strchr(line, ',');
        if (!comma) continue;
        *comma = '\0';
        strncpy(examples[n].path, line, MAX_PATH_LEN-1);
        examples[n].path[MAX_PATH_LEN-1] = '\0';
        examples[n].label = atoi(comma + 1);
        // printf("Loaded %s with label %d.\n", examples[n].path, examples[n].label);
        n++;
    }
    fclose(f);
    return n;
}


// Shuffle an array of ImageData structs in-place
void shuffle_array(ImageData* examples, int n) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1); // Pick random index from 0..i
        // Swap examples[i] and examples[j]
        ImageData temp = examples[i];
        examples[i] = examples[j];
        examples[j] = temp;
    }
}


// Return the index of the largest value in the softmax output,
// which corresponds to the predicted class.
int argmax(const float* arr, int n) {
    int best = 0;
    float best_val = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return best;
}


// Compute categorical cross-entropy loss for a batch
// probs: [batch_size][num_classes] — softmax outputs
// labels: [batch_size] — true class indices
// batch_size: number of samples
// num_classes: number of output classes
float cross_entropy_loss(const float* probs, const int* labels, int batch_size, int num_classes) {
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        int true_label = labels[i];
        float p = probs[i * num_classes + true_label];
        // Clamp p to avoid log(0) (which is -inf)
        p = fmaxf(p, 1e-8f);
        total_loss += -logf(p);
    }
    return total_loss / batch_size;
}


void load_minibatch(
    ImageData *all_examples, // the array from CSV
    int start_index,            // index to start from
    int batch_size,             // how many images to load (may be < BATCH_SIZE for last batch)
    float input2d[BATCH_SIZE][INPUT_PIXELS],
    int labels[BATCH_SIZE]
) {
    for (int i = 0; i < batch_size; ++i) {
        const char *path = all_examples[start_index + i].path;
        labels[i] = all_examples[start_index + i].label;
        if (load_greyscale_image(path, input2d[i], INPUT_SIZE, INPUT_SIZE)) {
            fprintf(stderr, "Warning: Failed to load image %s, zeroing out.\n", path);
            memset(input2d[i], 0, sizeof(float) * INPUT_PIXELS); // fill with zeros on error
            labels[i] = -1; // mark as invalid
        }
    }
}

// --- Two Convolution Layers: Weight Arrays ---
// Conv1: 8 filters, each is 3x3, input channel is 1 (grayscale)
float conv1_weights[CONV1_KERNELS][KERNEL1_SIZE * KERNEL1_SIZE];
float conv1_biases[CONV1_KERNELS];

// Conv2: 16 filters, each is 3x3x8 (operates on all conv1 outputs)
float conv2_weights[CONV2_KERNELS][CONV1_KERNELS][KERNEL2_SIZE * KERNEL2_SIZE];
float conv2_biases[CONV2_KERNELS];

// Dense layer (unchanged, just has more input features)
float dense_weights[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
float dense_biases[DENSE_OUTPUT_SIZE];


// --- Device buffers for 2-layer CNN (allocate with cudaMalloc) ---
float* conv1_weights_buf;    // [CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE]
float* conv1_biases_buf;     // [CONV1_KERNELS]
float* conv1_output_buf;     // [CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE]
float* pool1_output_buf;     // [CONV1_KERNELS * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE]

float* conv2_weights_buf;    // [CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE]
float* conv2_biases_buf;     // [CONV2_KERNELS]
float* conv2_output_buf;     // [CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE]
float* pool2_output_buf;     // [CONV2_KERNELS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE]

// For storing batch gradient accumulation on device:
float* grad_conv1_weights_accum_buf;  // [CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE]
float* grad_conv1_biases_accum_buf;   // [CONV1_KERNELS]
float* grad_conv2_weights_accum_buf;  // [CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE]
float* grad_conv2_biases_accum_buf;   // [CONV2_KERNELS]

// Dense and softmax
float* dense_weights_buf;             // [DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE]
float* dense_biases_buf;              // [DENSE_OUTPUT_SIZE]
float* dense_output_buf;              // [DENSE_OUTPUT_SIZE]
float* softmax_output_buf;            // [DENSE_OUTPUT_SIZE]

// For batch gradient accumulation on device (dense):
float* grad_dense_weights_accum_buf;  // [DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE]
float* grad_dense_biases_accum_buf;   // [DENSE_OUTPUT_SIZE]

// For input and intermediate workspace:
float* input_buf;                     // [INPUT_PIXELS]
float* grad_input_accum_buf;          // [DENSE_INPUT_SIZE] or other shapes as needed
float* grad_output_buf;               // Gradient of the loss with respect to the final output of the network

// Maxpool indices (needed for backward passes)
int* maxpool1_max_indices_buf;        // [CONV1_KERNELS * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE]
int* maxpool2_max_indices_buf;        // [CONV2_KERNELS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE]

// Pool gradients
float* grad_pool2_input_accum_buf;    // [CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE]
float* grad_pool1_input_accum_buf;    // [CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE]

// Conv1 input gradients (used if you want grads wrt image)
float* grad_conv1_input_accum_buf;    // [INPUT_PIXELS]


// Host-side workspace for diagnostics (just for printing/checking)
float grad_input_accum_buf_host[DENSE_INPUT_SIZE * DENSE_OUTPUT_SIZE];
float grad_conv2_weights_host[CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
float grad_pool2_input_host[CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
float grad_pool1_input_host[CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE];
float grad_input_host[INPUT_PIXELS];


#define DATA_PATH "./data/"
#define TRAIN_CSV_FILENAME "afhq64_train.csv"
#define VAL_CSV_FILENAME "afhq64_val.csv"

int main() {
    srand((unsigned int)time(NULL));

    // ---- 1. Load CSVs ----
    ImageData trainData[MAX_SET_SIZE];
    ImageData valData[MAX_SET_SIZE];
    char train_path[MAX_PATH_LEN], val_path[MAX_PATH_LEN];
    snprintf(train_path, sizeof(train_path), "%s%s", DATA_PATH, TRAIN_CSV_FILENAME);
    snprintf(val_path, sizeof(val_path), "%s%s", DATA_PATH, VAL_CSV_FILENAME);
    int train_size = load_csv(train_path, trainData, MAX_SET_SIZE);
    int val_size = load_csv(val_path, valData, MAX_SET_SIZE);
    printf("Train size: %d, Val size: %d\n", train_size, val_size);

    printf("Training data samples:\n");
    for (int i = 0; i < train_size; ++i) {
        printf("  %s label=%d\n", trainData[i].path, trainData[i].label);
    }
    printf("Validation data samples:\n");
    for (int i = 0; i < val_size; ++i) {
	printf("  %s label=%d\n", valData[i].path, valData[i].label);
    }

    // ---- 2. Init weights & biases ----
    // (no change: your host arrays are fine)

    // ---- 3. Buffer creation ----
    size_t input_bytes      = sizeof(float) * INPUT_PIXELS;
    size_t conv1_w_bytes    = CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE * sizeof(float);
    size_t conv1_b_bytes    = CONV1_KERNELS * sizeof(float);
    size_t conv1_out_bytes  = CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * sizeof(float);
    size_t pool1_out_bytes  = CONV1_KERNELS * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * sizeof(float);
    size_t conv2_w_bytes    = CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE * sizeof(float);
    size_t conv2_b_bytes    = CONV2_KERNELS * sizeof(float);
    size_t conv2_out_bytes  = CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * sizeof(float);
    size_t pool2_out_bytes  = CONV2_KERNELS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * sizeof(float);
    size_t dense_w_bytes    = DENSE_INPUT_SIZE * DENSE_OUTPUT_SIZE * sizeof(float);
    size_t dense_b_bytes    = DENSE_OUTPUT_SIZE * sizeof(float);
    size_t dense_out_bytes  = DENSE_OUTPUT_SIZE * sizeof(float);
    size_t maxpool1_indices_bytes = CONV1_KERNELS * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * sizeof(int);
    size_t maxpool2_indices_bytes = CONV2_KERNELS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE * sizeof(int);

    // -- Device allocations --
    cudaMalloc((void**)&conv1_weights_buf, conv1_w_bytes);
    cudaMalloc((void**)&conv1_biases_buf, conv1_b_bytes);
    cudaMalloc((void**)&conv1_output_buf, conv1_out_bytes);
    cudaMalloc((void**)&pool1_output_buf, pool1_out_bytes);
    cudaMalloc((void**)&conv2_weights_buf, conv2_w_bytes);
    cudaMalloc((void**)&conv2_biases_buf, conv2_b_bytes);
    cudaMalloc((void**)&conv2_output_buf, conv2_out_bytes);
    cudaMalloc((void**)&pool2_output_buf, pool2_out_bytes);
    cudaMalloc((void**)&dense_weights_buf, dense_w_bytes);
    cudaMalloc((void**)&dense_biases_buf, dense_b_bytes);
    cudaMalloc((void**)&dense_output_buf, dense_out_bytes);
    cudaMalloc((void**)&softmax_output_buf, dense_out_bytes);

    cudaMalloc((void**)&grad_conv1_weights_accum_buf, conv1_w_bytes);
    cudaMalloc((void**)&grad_conv1_biases_accum_buf, conv1_b_bytes);
    cudaMalloc((void**)&grad_conv2_weights_accum_buf, conv2_w_bytes);
    cudaMalloc((void**)&grad_conv2_biases_accum_buf, conv2_b_bytes);
    cudaMalloc((void**)&grad_dense_weights_accum_buf, dense_w_bytes);
    cudaMalloc((void**)&grad_dense_biases_accum_buf, dense_b_bytes);

    cudaMalloc((void**)&input_buf, input_bytes);
    cudaMalloc((void**)&grad_output_buf, dense_out_bytes);
    cudaMalloc((void**)&grad_input_accum_buf, DENSE_INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&maxpool1_max_indices_buf, maxpool1_indices_bytes);
    cudaMalloc((void**)&maxpool2_max_indices_buf, maxpool2_indices_bytes);
    cudaMalloc((void**)&grad_pool2_input_accum_buf, conv2_out_bytes);
    cudaMalloc((void**)&grad_pool1_input_accum_buf, conv1_out_bytes);
    cudaMalloc((void**)&grad_conv1_input_accum_buf, input_bytes);


    // --- RANDOM WEIGHT INITIALIZATION ---
    // Helper for random floats in [-0.05, 0.05]
    float rand_uniform(float a, float b) {
        return a + (b - a) * ((float)rand() / RAND_MAX);
    }

    // Conv1 weights: [CONV1_KERNELS][KERNEL1_SIZE * KERNEL1_SIZE]
    for (int i = 0; i < CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE; ++i)
        ((float*)conv1_weights)[i] = rand_uniform(-0.05f, 0.05f);

    // Conv1 biases: can be zero or tiny random
    for (int i = 0; i < CONV1_KERNELS; ++i)
	conv1_biases[i] = 0.0f;

    // Conv2 weights: [CONV2_KERNELS][CONV1_KERNELS][KERNEL2_SIZE * KERNEL2_SIZE]
    for (int i = 0; i < CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE; ++i)
	((float*)conv2_weights)[i] = rand_uniform(-0.05f, 0.05f);

    // Conv2 biases
    for (int i = 0; i < CONV2_KERNELS; ++i)
	conv2_biases[i] = 0.0f;

    // Dense weights: [DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE]
    for (int i = 0; i < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++i)
	dense_weights[i] = rand_uniform(-0.05f, 0.05f);

    // Dense biases
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
	dense_biases[i] = 0.0f;


    // -- Copy weights/biases to device (initialization) --
    cudaMemcpy(conv1_weights_buf, conv1_weights, conv1_w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(conv1_biases_buf, conv1_biases, conv1_b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_weights_buf, conv2_weights, conv2_w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(conv2_biases_buf, conv2_biases, conv2_b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_weights_buf, dense_weights, dense_w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dense_biases_buf, dense_biases, dense_b_bytes, cudaMemcpyHostToDevice);

    float input2d[BATCH_SIZE][INPUT_PIXELS];
    int labels[BATCH_SIZE];

    int first_batch = 1;
    int num_epochs = 100;
    float learning_rate = 0.001f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        printf("==== Epoch %d ====\n", epoch+1);
        shuffle_array(trainData, train_size); // shuffle for SGD
        int correct = 0;
        float total_loss = 0.0f;

        for (int batch_start = 0; batch_start < train_size; batch_start += BATCH_SIZE) {
            int this_batch_size = (batch_start + BATCH_SIZE > train_size)
                    ? train_size - batch_start
                    : BATCH_SIZE;
            load_minibatch(trainData, batch_start, this_batch_size, input2d, labels);

            if (first_batch) {
                first_batch = 0;
                printf("Sample pixels from input2d[0]: ");
                for (int j = 0; j < 10; ++j) printf("%f ", input2d[0][j]);
                printf("\n");
            }

            // ---- Zero accumulators before batch ----
            cudaZeroBuffer(grad_conv1_weights_accum_buf, CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE);
            cudaZeroBuffer(grad_conv1_biases_accum_buf, CONV1_KERNELS);
            cudaZeroBuffer(grad_conv2_weights_accum_buf, CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE);
            cudaZeroBuffer(grad_conv2_biases_accum_buf, CONV2_KERNELS);
            cudaZeroBuffer(grad_dense_weights_accum_buf, DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE);
            cudaZeroBuffer(grad_dense_biases_accum_buf, DENSE_OUTPUT_SIZE);
            cudaZeroBuffer(grad_pool2_input_accum_buf, CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
            cudaZeroBuffer(grad_pool1_input_accum_buf, CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
            cudaZeroBuffer(grad_conv1_input_accum_buf, INPUT_PIXELS);

            for (int i = 0; i < this_batch_size; ++i) {
                // --- Write one image to device
                cudaMemcpy(input_buf, input2d[i], input_bytes, cudaMemcpyHostToDevice);

                // --- Forward Pass ---
                conv2d_forward_cuda(
                    input_buf, conv1_weights_buf, conv1_biases_buf, conv1_output_buf,
                    INPUT_SIZE, KERNEL1_SIZE, CONV1_OUTPUT_SIZE, CONV1_KERNELS);

                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error after conv1: %s\n", cudaGetErrorString(err));
                }

                printf("conv1_output: ");
                float* conv1_output_host = (float*)malloc(conv1_out_bytes);
                cudaMemcpy(conv1_output_host, conv1_output_buf, conv1_out_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 5; ++i) printf("%f ", conv1_output_host[i]);
                    printf("\n");

                printf("conv1 weights: ");
                float* conv1_weights_host = (float*)malloc(conv1_w_bytes);
                cudaMemcpy(conv1_weights_host, conv1_weights_buf, conv1_w_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 5; ++i) printf("%f ", conv1_weights_host[i]);
                    printf("\n");

                printf("conv1 biases: ");
                float* conv1_biases_host = (float*)malloc(conv1_b_bytes);
                cudaMemcpy(conv1_biases_host, conv1_biases_buf, conv1_b_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 10; ++i) printf("%f ", conv1_biases_host[i]);
                    printf("\n");


                maxpool2d_forward_cuda(
                    conv1_output_buf, pool1_output_buf, maxpool1_max_indices_buf,
                    CONV1_KERNELS, CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE,
                    POOL_SIZE, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE);

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error after maxpool1: %s\n", cudaGetErrorString(err));
                }


                printf("maxpool1 output: ");
                float* maxpool1_output_host = (float*)malloc(pool1_out_bytes);
                cudaMemcpy(maxpool1_output_host, maxpool1_max_indices_buf, pool1_out_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 5; ++i) printf("%f ", maxpool1_output_host[i]);
                    printf("\n");

                conv2d_forward_cuda(
                    pool1_output_buf, conv2_weights_buf, conv2_biases_buf, conv2_output_buf,
                    POOL1_OUTPUT_SIZE, KERNEL2_SIZE, CONV2_OUTPUT_SIZE, CONV2_KERNELS);

                printf("conv2 output: ");
                float* conv2_output_host = (float*)malloc(conv2_out_bytes);
                cudaMemcpy(conv2_output_host, conv2_output_buf, conv2_out_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 5; ++i) printf("%f ", conv2_output_host[i]);
                    printf("\n");

                maxpool2d_forward_cuda(
                    conv2_output_buf, pool2_output_buf, maxpool2_max_indices_buf,
                    CONV2_KERNELS, CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE,
                    POOL_SIZE, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE);

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error after maxpool2: %s\n", cudaGetErrorString(err));
                }


                printf("maxpool2 output: ");
                float* maxpool2_output_host = (float*)malloc(pool2_out_bytes);
                cudaMemcpy(maxpool2_output_host, maxpool2_max_indices_buf, pool2_out_bytes, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 5; ++i) printf("%f ", maxpool2_output_host[i]);
                    printf("\n");


                dense_forward_cuda(
                    pool2_output_buf, dense_weights_buf, dense_biases_buf, dense_output_buf,
                    DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE);

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error after dense_forward: %s\n", cudaGetErrorString(err));
                }


                softmax_forward_cuda(
                    dense_output_buf, softmax_output_buf, DENSE_OUTPUT_SIZE);

                // --- Read output
                float softmax_output[DENSE_OUTPUT_SIZE];
                cudaMemcpy(softmax_output, softmax_output_buf, dense_out_bytes, cudaMemcpyDeviceToHost);

                // --- Compute gradient of loss wrt softmax output ---
                float grad_output[DENSE_OUTPUT_SIZE];
                for (int j = 0; j < DENSE_OUTPUT_SIZE; ++j) {
                    grad_output[j] = softmax_output[j] - (j == labels[i] ? 1.0f : 0.0f);
                }
                cudaMemcpy(grad_output_buf, grad_output, sizeof(float) * DENSE_OUTPUT_SIZE, cudaMemcpyHostToDevice);

                int pred = argmax(softmax_output, DENSE_OUTPUT_SIZE);
                float loss = -logf(fmaxf(softmax_output[labels[i]], 1e-8f));
                total_loss += loss;
                if (pred == labels[i]) correct++;


                dense_backward_accum_cuda(
                    pool2_output_buf, grad_output_buf,
                    grad_dense_weights_accum_buf, grad_dense_biases_accum_buf,
                    DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE);

                maxpool2d_backward_accum_cuda(
                    grad_output_buf, grad_pool2_input_accum_buf, maxpool2_max_indices_buf,
                    CONV2_KERNELS, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE);

                conv2d_backward_accum_cuda(
                    pool1_output_buf, grad_pool2_input_accum_buf,
                    grad_conv2_weights_accum_buf, grad_conv2_biases_accum_buf,
                    grad_pool1_input_accum_buf, conv2_weights_buf,
                    CONV1_KERNELS, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE,
                    CONV2_KERNELS, KERNEL2_SIZE, CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE);

                maxpool2d_backward_accum_cuda(
                    grad_pool1_input_accum_buf, grad_conv1_input_accum_buf, maxpool1_max_indices_buf,
                    CONV1_KERNELS, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE);

                conv2d_backward_accum_cuda(
                    input_buf, grad_conv1_input_accum_buf,
                    grad_conv1_weights_accum_buf, grad_conv1_biases_accum_buf,
                    grad_input_accum_buf, conv1_weights_buf,
                    1, INPUT_SIZE, INPUT_SIZE,
                    CONV1_KERNELS, KERNEL1_SIZE, CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE);
            }

            // ---- SGD update at end of batch ----
            dense_update_cuda(
                dense_weights_buf, dense_biases_buf,
                grad_dense_weights_accum_buf, grad_dense_biases_accum_buf,
                DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE, learning_rate, this_batch_size);

            conv2d_update_cuda(
                conv2_weights_buf, conv2_biases_buf,
                grad_conv2_weights_accum_buf, grad_conv2_biases_accum_buf,
                CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE, CONV2_KERNELS,
                learning_rate, this_batch_size);

            conv2d_update_cuda(
                conv1_weights_buf, conv1_biases_buf,
                grad_conv1_weights_accum_buf, grad_conv1_biases_accum_buf,
                CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE, CONV1_KERNELS,
                learning_rate, this_batch_size);
        }

        // End of epoch: print stats, copy weights/biases back for inspection
        cudaMemcpy(conv1_weights, conv1_weights_buf, conv1_w_bytes, cudaMemcpyDeviceToHost);
        printf("Conv1 weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)conv1_weights)[k]);

        cudaMemcpy(conv2_weights, conv2_weights_buf, conv2_w_bytes, cudaMemcpyDeviceToHost);
        printf("Conv2 weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)conv2_weights)[k]);

        cudaMemcpy(dense_weights, dense_weights_buf, dense_w_bytes, cudaMemcpyDeviceToHost);
        printf("Dense weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)dense_weights)[k]);

        cudaMemcpy(conv1_biases, conv1_biases_buf, conv1_b_bytes, cudaMemcpyDeviceToHost);
        printf("Conv1 biases after epoch %d:\n", epoch+1);
        for (int k = 0; k < CONV1_KERNELS; ++k)
            printf("  b[%d] = %f\n", k, conv1_biases[k]);

        cudaMemcpy(conv2_biases, conv2_biases_buf, conv2_b_bytes, cudaMemcpyDeviceToHost);
        printf("Conv2 biases after epoch %d:\n", epoch+1);
        for (int k = 0; k < CONV2_KERNELS; ++k)
            printf("  b[%d] = %f\n", k, conv2_biases[k]);

        cudaMemcpy(dense_biases, dense_biases_buf, dense_b_bytes, cudaMemcpyDeviceToHost);
        printf("Dense biases after epoch %d:\n", epoch+1);
        for (int k = 0; k < DENSE_OUTPUT_SIZE; ++k)
            printf("  b[%d] = %f\n", k, dense_biases[k]);

        float minw = conv1_weights[0][0], maxw = conv1_weights[0][0];
        for (int k = 0; k < CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE; ++k) {
            float val = ((float*)conv1_weights)[k];
            if (val < minw) minw = val;
            if (val > maxw) maxw = val;
        }
        printf("Conv1 weight min: %f max: %f\n", minw, maxw);

        // ---- VALIDATION PHASE ----
        int val_correct = 0;
        int confusion[DENSE_OUTPUT_SIZE][DENSE_OUTPUT_SIZE] = {0};

        for (int i = 0; i < val_size; ++i) {
            float val_input[INPUT_PIXELS];
            int true_label = valData[i].label;
            if (load_greyscale_image(valData[i].path, val_input, INPUT_SIZE, INPUT_SIZE)) {
                fprintf(stderr, "Failed to load validation image %s\n", valData[i].path);
                continue;
            }

            cudaMemcpy(input_buf, val_input, input_bytes, cudaMemcpyHostToDevice);

            conv2d_forward_cuda(input_buf, conv1_weights_buf, conv1_biases_buf, conv1_output_buf,
                INPUT_SIZE, KERNEL1_SIZE, CONV1_OUTPUT_SIZE, CONV1_KERNELS);

            maxpool2d_forward_cuda(conv1_output_buf, pool1_output_buf, maxpool1_max_indices_buf,
                CONV1_KERNELS, CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE,
                POOL_SIZE, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE);

            conv2d_forward_cuda(pool1_output_buf, conv2_weights_buf, conv2_biases_buf, conv2_output_buf,
                POOL1_OUTPUT_SIZE, KERNEL2_SIZE, CONV2_OUTPUT_SIZE, CONV2_KERNELS);

            maxpool2d_forward_cuda(conv2_output_buf, pool2_output_buf, maxpool2_max_indices_buf,
                CONV2_KERNELS, CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE,
                POOL_SIZE, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE);

            dense_forward_cuda(pool2_output_buf, dense_weights_buf, dense_biases_buf, dense_output_buf,
                DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE);

            softmax_forward_cuda(dense_output_buf, softmax_output_buf, DENSE_OUTPUT_SIZE);

            float softmax_output[DENSE_OUTPUT_SIZE];
            cudaMemcpy(softmax_output, softmax_output_buf, dense_out_bytes, cudaMemcpyDeviceToHost);

            if (i < 5) { // Only print for the first 5 validation images
                printf("Validation img %d (label=%d) softmax: ", i, true_label);
                for (int k = 0; k < DENSE_OUTPUT_SIZE; ++k)
                    printf("%8.5f ", softmax_output[k]);
                printf("\n");
            }

            int pred = argmax(softmax_output, DENSE_OUTPUT_SIZE);
            if (pred == true_label) val_correct++;
            confusion[true_label][pred]++;
        }

        float val_acc = 100.0f * val_correct / val_size;
        printf("VALIDATION: Accuracy = %.2f%%\n", val_acc);

        printf("Confusion Matrix (rows: true, cols: pred):\n");
        for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) {
            for (int j = 0; j < DENSE_OUTPUT_SIZE; ++j) {
                printf("%4d ", confusion[i][j]);
            }
            printf("\n");
        }

}  // end of epoch loop

    // --- Cleanup ---
    cudaFree(input_buf);
    cudaFree(conv1_weights_buf);
    cudaFree(conv1_biases_buf);
    cudaFree(conv1_output_buf);
    cudaFree(pool1_output_buf);
    cudaFree(conv2_weights_buf);
    cudaFree(conv2_biases_buf);
    cudaFree(conv2_output_buf);
    cudaFree(pool2_output_buf);
    cudaFree(dense_weights_buf);
    cudaFree(dense_biases_buf);
    cudaFree(dense_output_buf);
    cudaFree(softmax_output_buf);
    cudaFree(grad_output_buf);
    cudaFree(grad_input_accum_buf);
    cudaFree(maxpool1_max_indices_buf);
    cudaFree(maxpool2_max_indices_buf);
    cudaFree(grad_pool2_input_accum_buf);
    cudaFree(grad_pool1_input_accum_buf);
    cudaFree(grad_conv1_input_accum_buf);
    cudaFree(grad_conv1_weights_accum_buf);
    cudaFree(grad_conv1_biases_accum_buf);
    cudaFree(grad_conv2_weights_accum_buf);
    cudaFree(grad_conv2_biases_accum_buf);
    cudaFree(grad_dense_weights_accum_buf);
    cudaFree(grad_dense_biases_accum_buf);

    return 0;
}

