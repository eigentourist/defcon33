#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 32
#define KERNEL_SIZE 3
#define CONV_OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)
#define POOL_SIZE 2
#define POOL_OUTPUT_SIZE (CONV_OUTPUT_SIZE / POOL_SIZE)
#define DENSE_INPUT_SIZE (POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE)
#define DENSE_OUTPUT_SIZE 3

// For parallelized softmax
#define WGSIZE 64

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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


char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to load kernel file");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}


cl_int dense_layer_backprop(
    cl_command_queue queue,
    cl_kernel dense_backward_kernel,
    cl_mem pool_output_buf,
    cl_mem dense_weights_buf,
    cl_mem dense_biases_buf,
    cl_mem grad_output_buf,
    cl_mem grad_input_accum_buf,
    float* softmax_output,
    int label,
    int input_size,
    int output_size,
    float learning_rate)
{
    // Step 1: Compute grad_output (softmax - onehot)
    float grad_output[output_size];
    for (int k = 0; k < output_size; ++k)
        grad_output[k] = softmax_output[k] - (k == label ? 1.0f : 0.0f);

    // Step 2: Copy grad_output to device
    cl_int err = clEnqueueWriteBuffer(queue, grad_output_buf, CL_TRUE, 0,
        sizeof(float) * output_size, grad_output, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("grad_output_buf write error: %d\n", err); return err; }

    // Step 3: Launch backward kernel
    size_t global_size = output_size;
    err = clSetKernelArg(dense_backward_kernel, 0, sizeof(cl_mem), &pool_output_buf);
    err |= clSetKernelArg(dense_backward_kernel, 1, sizeof(cl_mem), &dense_weights_buf);
    err |= clSetKernelArg(dense_backward_kernel, 2, sizeof(cl_mem), &dense_biases_buf);
    err |= clSetKernelArg(dense_backward_kernel, 3, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(dense_backward_kernel, 4, sizeof(cl_mem), &grad_input_accum_buf);
    err |= clSetKernelArg(dense_backward_kernel, 5, sizeof(int), &input_size);
    err |= clSetKernelArg(dense_backward_kernel, 6, sizeof(int), &output_size);
    err |= clSetKernelArg(dense_backward_kernel, 7, sizeof(float), &learning_rate);
    if (err != CL_SUCCESS) { printf("dense_backward_kernel set args error: %d\n", err); return err; }

    err = clEnqueueNDRangeKernel(queue, dense_backward_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("dense_backward_kernel launch error: %d\n", err); return err; }
    clFinish(queue);

    return CL_SUCCESS;
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


#define BATCH_SIZE 16
#define INPUT_PIXELS (INPUT_SIZE * INPUT_SIZE)

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



#define DATA_PATH "./data/"
#define TRAIN_CSV_FILENAME "afhq32_train.csv"
#define VAL_CSV_FILENAME "afhq32_val.csv"

int main() {
    // Seed PRNG with current time (once per program)
    srand((unsigned int)time(NULL));

    // --- 1. Load CSV batch ---
    ImageData imageData[MAX_SET_SIZE];

    char train_path[MAX_PATH_LEN];
    snprintf(train_path, sizeof(train_path), "%s%s", DATA_PATH, TRAIN_CSV_FILENAME);

    printf("Loading training set from %s.\n", train_path);
    int training_set_size = load_csv(train_path, imageData, MAX_SET_SIZE);
    printf("Loaded %d training items from %s.\n", training_set_size, train_path);

    // Validation set
    ImageData valData[MAX_SET_SIZE];
    char val_path[MAX_PATH_LEN] = {0};
    strncat(val_path, DATA_PATH, sizeof(DATA_PATH));
    strncat(val_path, VAL_CSV_FILENAME, sizeof(VAL_CSV_FILENAME));
    printf("Loading validation set from %s.\n", val_path);
    int val_set_size = load_csv(val_path, valData, MAX_SET_SIZE);
    printf("Loaded %d validation items from %s.\n", val_set_size, val_path);

    // --- 2. Init model weights ---
    // float input[INPUT_SIZE * INPUT_SIZE];
    float input2d[BATCH_SIZE][INPUT_SIZE * INPUT_SIZE];
    int labels[BATCH_SIZE];
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    float dense_weights[DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE];
    float dense_biases[DENSE_OUTPUT_SIZE];
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
        kernel[i] = 1.0f / 9.0f;
    for (int i = 0; i < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++i)
        dense_weights[i] = 0.1f;
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        dense_biases[i] = (float)i * 0.1f;

    // --- 3. OpenCL boilerplate (platform, device, context, queue, program) ---
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    char* source = load_kernel_source("src/cnn_kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error:\n%s\n", log);
        free(log); free(source);
        return 1;
    }
    free(source);

    // --- 4. Allocate device buffers (reused for all images) ---
    size_t conv_output_bytes = CONV_OUTPUT_SIZE * CONV_OUTPUT_SIZE * sizeof(float);
    size_t pool_output_bytes = POOL_OUTPUT_SIZE * POOL_OUTPUT_SIZE * sizeof(float);
    size_t dense_output_bytes = DENSE_OUTPUT_SIZE * sizeof(float);

    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * BATCH_SIZE * INPUT_PIXELS, NULL, &err);
    cl_mem kernel_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE, kernel, &err);
    cl_mem conv_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, conv_output_bytes, NULL, &err);
    cl_mem pool_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, pool_output_bytes, NULL, &err);
    cl_mem dense_weights_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_INPUT_SIZE * DENSE_OUTPUT_SIZE, dense_weights, &err);
    cl_mem dense_biases_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * DENSE_OUTPUT_SIZE, dense_biases, &err);
    cl_mem dense_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dense_output_bytes, NULL, &err);
    cl_mem softmax_output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dense_output_bytes, NULL, &err);

    cl_mem grad_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DENSE_OUTPUT_SIZE, NULL, &err);
    cl_mem grad_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE, NULL, &err);

    // --- 5. Create all kernels (once) ---
    cl_kernel conv_kernel = clCreateKernel(program, "conv2d", &err);
    cl_kernel pool_kernel = clCreateKernel(program, "maxpool2d", &err);
    cl_kernel dense_kernel = clCreateKernel(program, "dense_layer", &err);
    cl_kernel softmax_kernel = clCreateKernel(program, "softmax_parallel", &err);
    cl_kernel dense_backward_kernel = clCreateKernel(program, "dense_backward", &err);

    // Set constant kernel args
    int input_size_val = INPUT_SIZE, kernel_size_val = KERNEL_SIZE, conv_out_val = CONV_OUTPUT_SIZE;
    int pool_size_val = POOL_SIZE, pool_out_val = POOL_OUTPUT_SIZE;
    int dense_in_val = DENSE_INPUT_SIZE, dense_out_val = DENSE_OUTPUT_SIZE;

    clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &kernel_buf);
    clSetKernelArg(conv_kernel, 3, sizeof(int), &input_size_val);
    clSetKernelArg(conv_kernel, 4, sizeof(int), &kernel_size_val);
    clSetKernelArg(conv_kernel, 5, sizeof(int), &conv_out_val);

    clSetKernelArg(pool_kernel, 2, sizeof(int), &conv_out_val);
    clSetKernelArg(pool_kernel, 3, sizeof(int), &pool_size_val);

    clSetKernelArg(dense_kernel, 1, sizeof(cl_mem), &dense_weights_buf);
    clSetKernelArg(dense_kernel, 2, sizeof(cl_mem), &dense_biases_buf);
    clSetKernelArg(dense_kernel, 4, sizeof(int), &dense_in_val);
    clSetKernelArg(dense_kernel, 5, sizeof(int), &dense_out_val);

    clSetKernelArg(softmax_kernel, 2, sizeof(int), &dense_out_val);

    int num_epochs = 500;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        printf("==== Epoch %d ====\n", epoch+1);
        shuffle_array(imageData, training_set_size); // Reshuffle data at start of each epoch
        int correct = 0; // total correct predictions for epoch
        float total_loss = 0.0f; // total loss for epoch

        for (int batch_start = 0; batch_start < training_set_size; batch_start += BATCH_SIZE) {
            int this_batch_size = (batch_start + BATCH_SIZE > training_set_size)
                    ? training_set_size - batch_start
                    : BATCH_SIZE;

            // Load the minibatch into arrays (input2d, labels)
            load_minibatch(imageData, batch_start, this_batch_size, input2d, labels);

            for (int i = 0; i < this_batch_size; ++i) {
                // (a) Write a SINGLE image to device
                clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0,
                    sizeof(float) * INPUT_PIXELS, input2d[i], 0, NULL, NULL);

                // (b) Forward pass
                clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &input_buf);
                clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_output_buf);
                size_t global_size_conv[2] = { conv_out_val, conv_out_val };
                clEnqueueNDRangeKernel(queue, conv_kernel, 2, NULL, global_size_conv, NULL, 0, NULL, NULL);

                clSetKernelArg(pool_kernel, 0, sizeof(cl_mem), &conv_output_buf);
                clSetKernelArg(pool_kernel, 1, sizeof(cl_mem), &pool_output_buf);
                size_t global_size_pool[2] = { pool_out_val, pool_out_val };
                clEnqueueNDRangeKernel(queue, pool_kernel, 2, NULL, global_size_pool, NULL, 0, NULL, NULL);

                clSetKernelArg(dense_kernel, 0, sizeof(cl_mem), &pool_output_buf);
                clSetKernelArg(dense_kernel, 3, sizeof(cl_mem), &dense_output_buf);
                size_t global_size_dense = dense_out_val;
                clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &global_size_dense, NULL, 0, NULL, NULL);

                clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &dense_output_buf);
                clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &softmax_output_buf);
                size_t local_size = WGSIZE;
                size_t global_size = ((dense_out_val + WGSIZE - 1) / WGSIZE) * WGSIZE;
                clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

                // (c) Read output
                float softmax_output[DENSE_OUTPUT_SIZE];
                clEnqueueReadBuffer(queue, softmax_output_buf, CL_TRUE, 0,
                    sizeof(softmax_output), softmax_output, 0, NULL, NULL);

                // (d) Prediction and Loss
                int pred = argmax(softmax_output, DENSE_OUTPUT_SIZE);
                float loss = -logf(fmaxf(softmax_output[labels[i]], 1e-8f));
                total_loss += loss;
                if (pred == labels[i]) correct++;

                // (e) Backpropagation for this sample
                float learning_rate = 0.1f;
                cl_int bp_err = dense_layer_backprop(
                    queue, dense_backward_kernel,
                    pool_output_buf, dense_weights_buf, dense_biases_buf,
                    grad_output_buf, grad_input_accum_buf,
                    softmax_output, labels[i],
                    DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE,
                    learning_rate
                );
                if (bp_err != CL_SUCCESS) printf("Backprop error: %d\n", bp_err);
            }
        }
        // End of epoch: print stats
        float avg_loss = total_loss / training_set_size;
        float acc = 100.0f * correct / training_set_size;
        printf("Epoch %d Summary: Accuracy = %.2f%%, Avg Loss = %.4f\n\n", epoch+1, acc, avg_loss);

        // ---- VALIDATION PHASE ----
        int val_correct = 0;
        int confusion[DENSE_OUTPUT_SIZE][DENSE_OUTPUT_SIZE] = {0}; // [true][pred]

        for (int i = 0; i < val_set_size; ++i) {
            float val_input[INPUT_PIXELS];
            int true_label = valData[i].label;
            if (load_greyscale_image(valData[i].path, val_input, INPUT_SIZE, INPUT_SIZE)) {
                fprintf(stderr, "Failed to load validation image %s\n", valData[i].path);
                continue;
            }

            // Write image to device
            clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, sizeof(float) * INPUT_PIXELS, val_input, 0, NULL, NULL);

            // Forward pass (same as training)
            clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &input_buf);
            clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &conv_output_buf);
            size_t global_size_conv[2] = { conv_out_val, conv_out_val };
            clEnqueueNDRangeKernel(queue, conv_kernel, 2, NULL, global_size_conv, NULL, 0, NULL, NULL);

            clSetKernelArg(pool_kernel, 0, sizeof(cl_mem), &conv_output_buf);
            clSetKernelArg(pool_kernel, 1, sizeof(cl_mem), &pool_output_buf);
            size_t global_size_pool[2] = { pool_out_val, pool_out_val };
            clEnqueueNDRangeKernel(queue, pool_kernel, 2, NULL, global_size_pool, NULL, 0, NULL, NULL);

            clSetKernelArg(dense_kernel, 0, sizeof(cl_mem), &pool_output_buf);
            clSetKernelArg(dense_kernel, 3, sizeof(cl_mem), &dense_output_buf);
            size_t global_size_dense = dense_out_val;
            clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &global_size_dense, NULL, 0, NULL, NULL);

            clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &dense_output_buf);
            clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &softmax_output_buf);
            size_t local_size = WGSIZE;
            size_t global_size = ((dense_out_val + WGSIZE - 1) / WGSIZE) * WGSIZE;
            clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

            float softmax_output[DENSE_OUTPUT_SIZE];
            clEnqueueReadBuffer(queue, softmax_output_buf, CL_TRUE, 0,
                sizeof(softmax_output), softmax_output, 0, NULL, NULL);

            int pred = argmax(softmax_output, DENSE_OUTPUT_SIZE);
            if (pred == true_label) val_correct++;
            confusion[true_label][pred]++;
        }

        float val_acc = 100.0f * val_correct / val_set_size;
        printf("VALIDATION: Accuracy = %.2f%%\n", val_acc);

        // Print confusion matrix
        printf("Confusion Matrix (rows: true, cols: pred):\n");
        for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i) {
            for (int j = 0; j < DENSE_OUTPUT_SIZE; ++j) {
                printf("%4d ", confusion[i][j]);
            }
            printf("\n");
        }

    }


    // --- 7. Cleanup ---
    clReleaseMemObject(input_buf);
    clReleaseMemObject(kernel_buf);
    clReleaseMemObject(conv_output_buf);
    clReleaseMemObject(pool_output_buf);
    clReleaseMemObject(dense_weights_buf);
    clReleaseMemObject(dense_biases_buf);
    clReleaseMemObject(dense_output_buf);
    clReleaseMemObject(softmax_output_buf);
    clReleaseMemObject(grad_output_buf);
    clReleaseMemObject(grad_input_accum_buf);

    clReleaseKernel(conv_kernel);
    clReleaseKernel(pool_kernel);
    clReleaseKernel(dense_kernel);
    clReleaseKernel(softmax_kernel);
    clReleaseKernel(dense_backward_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
