#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

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


void zero_buffer(cl_command_queue queue, cl_mem buf, size_t n_floats) {
    float* zeros = calloc(n_floats, sizeof(float));
    clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, n_floats * sizeof(float), zeros, 0, NULL, NULL);
    free(zeros);
}


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
    printf("Loading GPU kernel code from %s.\n", filename);
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


cl_int conv2d_forward(
    cl_command_queue queue,
    cl_kernel conv2d_kernel,
    cl_mem input_buf,
    cl_mem weights_buf,
    cl_mem biases_buf,
    cl_mem output_buf,
    int in_channels, int input_size,
    int num_filters, int kernel_size, int output_size
) {
    cl_int err = 0;
    err  = clSetKernelArg(conv2d_kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(conv2d_kernel, 1, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(conv2d_kernel, 2, sizeof(cl_mem), &biases_buf);
    err |= clSetKernelArg(conv2d_kernel, 3, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(conv2d_kernel, 4, sizeof(int), &input_size);
    err |= clSetKernelArg(conv2d_kernel, 5, sizeof(int), &kernel_size);
    err |= clSetKernelArg(conv2d_kernel, 6, sizeof(int), &output_size);
    if (err != CL_SUCCESS) { printf("conv2d set args error: %d\n", err); return err; }

    size_t global_size[3] = { output_size, output_size, num_filters };
    err = clEnqueueNDRangeKernel(queue, conv2d_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("conv2d launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



cl_int dense_forward(
    cl_command_queue queue,
    cl_kernel dense_kernel,
    cl_mem input_buf,
    cl_mem weights_buf,
    cl_mem biases_buf,
    cl_mem output_buf,
    int input_size,
    int output_size
) {
    cl_int err = 0;
    err  = clSetKernelArg(dense_kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(dense_kernel, 1, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(dense_kernel, 2, sizeof(cl_mem), &biases_buf);
    err |= clSetKernelArg(dense_kernel, 3, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(dense_kernel, 4, sizeof(int), &input_size);
    err |= clSetKernelArg(dense_kernel, 5, sizeof(int), &output_size);
    if (err != CL_SUCCESS) { printf("dense set args error: %d\n", err); return err; }

    size_t global_size = output_size;
    err = clEnqueueNDRangeKernel(queue, dense_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("dense launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}


cl_int maxpool2d_forward (
    cl_command_queue queue,
    cl_kernel maxpool2d_kernel,
    cl_mem input_buf,
    cl_mem output_buf,
    cl_mem max_indices_buf,
    int channels, int in_h, int in_w,
    int pool_size, int out_h, int out_w)
{
    cl_int err = 0;
    err  = clSetKernelArg(maxpool2d_kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(maxpool2d_kernel, 1, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(maxpool2d_kernel, 2, sizeof(cl_mem), &max_indices_buf);
    err |= clSetKernelArg(maxpool2d_kernel, 3, sizeof(int), &channels);
    err |= clSetKernelArg(maxpool2d_kernel, 4, sizeof(int), &in_h);
    err |= clSetKernelArg(maxpool2d_kernel, 5, sizeof(int), &in_w);
    err |= clSetKernelArg(maxpool2d_kernel, 6, sizeof(int), &pool_size);
    err |= clSetKernelArg(maxpool2d_kernel, 7, sizeof(int), &out_h);
    err |= clSetKernelArg(maxpool2d_kernel, 8, sizeof(int), &out_w);

    if (err != CL_SUCCESS) { printf("maxpool2d set args error: %d\n", err); return err; }

    size_t global_size[3] = { channels, out_h, out_w };
    err = clEnqueueNDRangeKernel(queue, maxpool2d_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("maxpool2d launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}


cl_int softmax_forward(
    cl_command_queue queue,
    cl_kernel softmax_kernel,
    cl_mem input_buf,
    cl_mem output_buf,
    int output_size
) {
    cl_int err = 0;
    err  = clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(softmax_kernel, 2, sizeof(int), &output_size);
    if (err != CL_SUCCESS) { printf("softmax set args error: %d\n", err); return err; }

    size_t sm_local = WGSIZE;
    size_t sm_global = ((output_size + WGSIZE - 1) / WGSIZE) * WGSIZE;
    err = clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &sm_global, &sm_local, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("softmax launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
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


// Accumulate gradients for one sample (adds to device accum buffers)
cl_int dense_backward_accum(
    cl_command_queue queue,
    cl_kernel dense_backward_accum_kernel,
    cl_mem input_buf,
    cl_mem grad_output_buf,     // [output_size] grad wrt this sample’s dense output
    cl_mem grad_weights_accum_buf, // [output_size * input_size] accumulator
    cl_mem grad_biases_accum_buf,  // [output_size] accumulator
    cl_mem grad_input_accum_buf,   // [input_size] for input grad propagation
    int input_size,
    int output_size
) {
    cl_int err = 0;
    err  = clSetKernelArg(dense_backward_accum_kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(dense_backward_accum_kernel, 1, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(dense_backward_accum_kernel, 2, sizeof(cl_mem), &grad_weights_accum_buf);
    err |= clSetKernelArg(dense_backward_accum_kernel, 3, sizeof(cl_mem), &grad_biases_accum_buf);
    err |= clSetKernelArg(dense_backward_accum_kernel, 4, sizeof(cl_mem), &grad_input_accum_buf);
    err |= clSetKernelArg(dense_backward_accum_kernel, 5, sizeof(int), &input_size);
    err |= clSetKernelArg(dense_backward_accum_kernel, 6, sizeof(int), &output_size);
    if (err != CL_SUCCESS) { printf("dense_backward_accum set args error: %d\n", err); return err; }

    size_t global_size = output_size;
    err = clEnqueueNDRangeKernel(queue, dense_backward_accum_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("dense_backward_accum launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



// Update dense weights & biases after batch
cl_int dense_update(
    cl_command_queue queue,
    cl_kernel dense_update_kernel,
    cl_mem weights_buf,
    cl_mem biases_buf,
    cl_mem grad_weights_accum_buf,
    cl_mem grad_biases_accum_buf,
    int n_weights, int n_biases,
    float learning_rate,
    int batch_size)
{
    cl_int err = 0;
    err  = clSetKernelArg(dense_update_kernel, 0, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(dense_update_kernel, 1, sizeof(cl_mem), &biases_buf);
    err |= clSetKernelArg(dense_update_kernel, 2, sizeof(cl_mem), &grad_weights_accum_buf);
    err |= clSetKernelArg(dense_update_kernel, 3, sizeof(cl_mem), &grad_biases_accum_buf);
    err |= clSetKernelArg(dense_update_kernel, 4, sizeof(int), &n_weights);
    err |= clSetKernelArg(dense_update_kernel, 5, sizeof(int), &n_biases);
    err |= clSetKernelArg(dense_update_kernel, 6, sizeof(float), &learning_rate);
    err |= clSetKernelArg(dense_update_kernel, 7, sizeof(int), &batch_size);

    size_t global_size = (n_weights > n_biases) ? n_weights : n_biases;
    err = clEnqueueNDRangeKernel(queue, dense_update_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("dense_update launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



cl_int maxpool2d_backward(
    cl_command_queue queue,
    cl_kernel maxpool2d_backward_kernel,
    cl_mem grad_output_buf,
    cl_mem grad_input_buf,
    cl_mem max_indices_buf,
    int channels, int out_h, int out_w,
    int in_h, int in_w // <-- ADD THESE
) {
    // 1. Zero grad_input_buf
    zero_buffer(queue, grad_input_buf, channels * in_h * in_w);

    // 2. Set args (these remain unchanged)
    cl_int err = 0;
    err  = clSetKernelArg(maxpool2d_backward_kernel, 0, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(maxpool2d_backward_kernel, 1, sizeof(cl_mem), &grad_input_buf);
    err |= clSetKernelArg(maxpool2d_backward_kernel, 2, sizeof(cl_mem), &max_indices_buf);
    err |= clSetKernelArg(maxpool2d_backward_kernel, 3, sizeof(int), &channels);
    err |= clSetKernelArg(maxpool2d_backward_kernel, 4, sizeof(int), &out_h);
    err |= clSetKernelArg(maxpool2d_backward_kernel, 5, sizeof(int), &out_w);

    if (err != CL_SUCCESS) { printf("maxpool2d_backward set args error: %d\n", err); return err; }

    // 3. Launch: 3D grid
    size_t global_size[3] = { channels, out_h, out_w };
    err = clEnqueueNDRangeKernel(queue, maxpool2d_backward_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("maxpool2d_backward launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



// Backprop maxpool for one sample (accumulates in grad_input_accum_buf)
cl_int maxpool2d_backward_accum(
    cl_command_queue queue,
    cl_kernel maxpool2d_backward_accum_kernel,
    cl_mem grad_output_buf,
    cl_mem grad_input_accum_buf,
    cl_mem max_indices_buf,
    int channels, int out_h, int out_w)
{
    cl_int err = 0;
    err  = clSetKernelArg(maxpool2d_backward_accum_kernel, 0, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(maxpool2d_backward_accum_kernel, 1, sizeof(cl_mem), &grad_input_accum_buf);
    err |= clSetKernelArg(maxpool2d_backward_accum_kernel, 2, sizeof(cl_mem), &max_indices_buf);
    err |= clSetKernelArg(maxpool2d_backward_accum_kernel, 3, sizeof(int), &channels);
    err |= clSetKernelArg(maxpool2d_backward_accum_kernel, 4, sizeof(int), &out_h);
    err |= clSetKernelArg(maxpool2d_backward_accum_kernel, 5, sizeof(int), &out_w);

    size_t global_size[3] = { channels, out_h, out_w };
    err = clEnqueueNDRangeKernel(queue, maxpool2d_backward_accum_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("maxpool2d_backward_accum launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



cl_int conv2d_backprop(
    cl_command_queue queue,
    cl_kernel conv2d_backward_kernel,
    cl_mem in_buf,              // [inC, inH, inW] - pool1_output
    cl_mem weights_buf,         // [outC, inC, k, k]
    cl_mem biases_buf,          // [outC]
    cl_mem grad_output_buf,     // [outC, outH, outW] - grad from upstream (usually maxpool2d_backward)
    cl_mem grad_input_buf,      // [inC, inH, inW] - OUT (must zero before launch!)
    int inC, int inH, int inW,
    int outC, int k,
    int outH, int outW,
    float learning_rate
) {
    // 1. Zero grad_input_buf
    zero_buffer(queue, grad_input_buf, inC * inH * inW);

    // 2. Set kernel args
    cl_int err = 0;
    err  = clSetKernelArg(conv2d_backward_kernel, 0, sizeof(cl_mem), &in_buf);
    err |= clSetKernelArg(conv2d_backward_kernel, 1, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(conv2d_backward_kernel, 2, sizeof(cl_mem), &biases_buf);
    err |= clSetKernelArg(conv2d_backward_kernel, 3, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(conv2d_backward_kernel, 4, sizeof(cl_mem), &grad_input_buf);
    err |= clSetKernelArg(conv2d_backward_kernel, 5, sizeof(int), &inC);
    err |= clSetKernelArg(conv2d_backward_kernel, 6, sizeof(int), &inH);
    err |= clSetKernelArg(conv2d_backward_kernel, 7, sizeof(int), &inW);
    err |= clSetKernelArg(conv2d_backward_kernel, 8, sizeof(int), &outC);
    err |= clSetKernelArg(conv2d_backward_kernel, 9, sizeof(int), &k);
    err |= clSetKernelArg(conv2d_backward_kernel,10, sizeof(int), &outH);
    err |= clSetKernelArg(conv2d_backward_kernel,11, sizeof(int), &outW);
    err |= clSetKernelArg(conv2d_backward_kernel,12, sizeof(float), &learning_rate);

    if (err != CL_SUCCESS) { printf("conv2d_backward set args error: %d\n", err); return err; }

    // 3. Launch (one thread per output channel/filter)
    size_t global_size = outC;
    err = clEnqueueNDRangeKernel(queue, conv2d_backward_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("conv2d_backward launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}



cl_int conv2d_backward_accum(
    cl_command_queue queue,
    cl_kernel conv2d_backward_accum_kernel,
    cl_mem in_buf,              // input to this conv layer, for this sample
    cl_mem grad_output_buf,     // gradient wrt output, for this sample
    cl_mem grad_weights_accum_buf, // [F,C,k,k], accumulator (device buffer)
    cl_mem grad_biases_accum_buf,  // [F], accumulator (device buffer)
    cl_mem grad_input_accum_buf,   // [C,H,W], accumulator (device buffer)
    int inC, int inH, int inW,
    int outC, int k, int outH, int outW)
{
    cl_int err = 0;
    err  = clSetKernelArg(conv2d_backward_accum_kernel, 0, sizeof(cl_mem), &in_buf);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 1, sizeof(cl_mem), &grad_output_buf);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 2, sizeof(cl_mem), &grad_weights_accum_buf);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 3, sizeof(cl_mem), &grad_biases_accum_buf);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 4, sizeof(cl_mem), &grad_input_accum_buf);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 5, sizeof(int), &inC);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 6, sizeof(int), &inH);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 7, sizeof(int), &inW);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 8, sizeof(int), &outC);
    err |= clSetKernelArg(conv2d_backward_accum_kernel, 9, sizeof(int), &k);
    err |= clSetKernelArg(conv2d_backward_accum_kernel,10, sizeof(int), &outH);
    err |= clSetKernelArg(conv2d_backward_accum_kernel,11, sizeof(int), &outW);
    if (err != CL_SUCCESS) { printf("conv2d_backward_accum set args error: %d\n", err); return err; }

    size_t global_size = outC;
    err = clEnqueueNDRangeKernel(queue, conv2d_backward_accum_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("conv2d_backward_accum launch error: %d\n", err); return err; }
    clFinish(queue);
    return CL_SUCCESS;
}


cl_int conv2d_update(
    cl_command_queue queue,
    cl_kernel conv2d_update_kernel,
    cl_mem weights_buf,
    cl_mem biases_buf,
    cl_mem grad_weights_accum_buf,
    cl_mem grad_biases_accum_buf,
    int n_weights, int n_biases,
    float learning_rate,
    int batch_size)
{
    cl_int err = 0;
    err  = clSetKernelArg(conv2d_update_kernel, 0, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(conv2d_update_kernel, 1, sizeof(cl_mem), &biases_buf);
    err |= clSetKernelArg(conv2d_update_kernel, 2, sizeof(cl_mem), &grad_weights_accum_buf);
    err |= clSetKernelArg(conv2d_update_kernel, 3, sizeof(cl_mem), &grad_biases_accum_buf);
    err |= clSetKernelArg(conv2d_update_kernel, 4, sizeof(int), &n_weights);
    err |= clSetKernelArg(conv2d_update_kernel, 5, sizeof(int), &n_biases);
    err |= clSetKernelArg(conv2d_update_kernel, 6, sizeof(float), &learning_rate);
    err |= clSetKernelArg(conv2d_update_kernel, 7, sizeof(int), &batch_size);

    // You can launch with max(n_weights, n_biases) as the global size.
    size_t global_size = (n_weights > n_biases) ? n_weights : n_biases;
    err = clEnqueueNDRangeKernel(queue, conv2d_update_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { printf("conv2d_update launch error: %d\n", err); return err; }
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

// --- OpenCL Buffers for 2-layer CNN ---
// Layer 1
cl_mem conv1_weights_buf;    // [CONV1_KERNELS][KERNEL1_SIZE*KERNEL1_SIZE]
cl_mem conv1_biases_buf;     // [CONV1_KERNELS]
cl_mem conv1_output_buf;     // [CONV1_KERNELS][CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE]
cl_mem pool1_output_buf;     // [CONV1_KERNELS][POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE]

// Layer 2
cl_mem conv2_weights_buf;    // [CONV2_KERNELS][CONV1_KERNELS][KERNEL2_SIZE*KERNEL2_SIZE]
cl_mem conv2_biases_buf;     // [CONV2_KERNELS]
cl_mem conv2_output_buf;     // [CONV2_KERNELS][CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE]
cl_mem pool2_output_buf;     // [CONV2_KERNELS][POOL2_OUTPUT_SIZE*POOL2_OUTPUT_SIZE]
cl_mem grad_conv2_weights_buf;    // CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE

// Dense and softmax
cl_mem dense_weights_buf;
cl_mem dense_biases_buf;
cl_mem dense_output_buf;
cl_mem softmax_output_buf;

// Buffers for diagnostics
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

    // ---- 2. Init weights & biases ----
    // Conv1
    for (int f = 0; f < CONV1_KERNELS; ++f) {
        for (int k = 0; k < KERNEL1_SIZE * KERNEL1_SIZE; ++k)
            conv1_weights[f][k] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        conv1_biases[f] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    // Conv2
    for (int f = 0; f < CONV2_KERNELS; ++f) {
        for (int c = 0; c < CONV1_KERNELS; ++c)
            for (int k = 0; k < KERNEL2_SIZE * KERNEL2_SIZE; ++k)
                conv2_weights[f][c][k] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        conv2_biases[f] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    }
    // Dense
    for (int i = 0; i < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++i)
        dense_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    for (int i = 0; i < DENSE_OUTPUT_SIZE; ++i)
        dense_biases[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;

    // ---- 3. OpenCL setup ----
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
    char* src = load_kernel_source("src/cnn_kernel.cl");
    printf("Loaded kernel source!\n");
    printf("Building program...\n");
    program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build failed:\n%s\n", log);
        free(log);
        return 1;
    }
    free(src);

    // ---- 4. Buffer creation ----
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

    cl_mem input_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, input_bytes, NULL, &err);
    conv1_weights_buf         = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, conv1_w_bytes, conv1_weights, &err);
    conv1_biases_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, conv1_b_bytes, conv1_biases, &err);
    conv1_output_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, conv1_out_bytes, NULL, &err);
    pool1_output_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, pool1_out_bytes, NULL, &err);
    conv2_weights_buf         = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, conv2_w_bytes, conv2_weights, &err);
    conv2_biases_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, conv2_b_bytes, conv2_biases, &err);
    conv2_output_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, conv2_out_bytes, NULL, &err);
    pool2_output_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, pool2_out_bytes, NULL, &err);
    dense_weights_buf         = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dense_w_bytes, dense_weights, &err);
    dense_biases_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dense_b_bytes, dense_biases, &err);
    dense_output_buf          = clCreateBuffer(context, CL_MEM_READ_WRITE, dense_out_bytes, NULL, &err);
    softmax_output_buf        = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dense_out_bytes, NULL, &err);
    grad_conv2_weights_buf    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE, NULL, &err);


    cl_mem grad_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DENSE_OUTPUT_SIZE, NULL, &err);
    cl_mem grad_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE, NULL, &err);
    cl_mem maxpool1_max_indices_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, maxpool1_indices_bytes, NULL, &err);
    cl_mem maxpool2_max_indices_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * CONV2_KERNELS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE, NULL, &err);
    cl_mem grad_pool2_input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE, NULL, &err);
    cl_mem grad_pool1_input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, NULL, &err);
    cl_mem grad_input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * INPUT_PIXELS, NULL, &err);


        // ---- Conv1 accumulators ----
    cl_mem grad_conv1_weights_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV1_KERNELS * 1 * KERNEL1_SIZE * KERNEL1_SIZE, NULL, &err);
    cl_mem grad_conv1_biases_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV1_KERNELS, NULL, &err);

    // ---- Conv2 accumulators ----
    cl_mem grad_conv2_weights_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE, NULL, &err);
    cl_mem grad_conv2_biases_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV2_KERNELS, NULL, &err);

    // ---- Dense accumulators ----
    cl_mem grad_dense_weights_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE, NULL, &err);
    cl_mem grad_dense_biases_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * DENSE_OUTPUT_SIZE, NULL, &err);

    // Pool2 input gradient accumulator buffer
    cl_mem grad_pool2_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE, NULL, &err);

    // Pool1 input gradient accumulator buffer
    cl_mem grad_pool1_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, NULL, &err);

    // Conv1 input gradient accumulator buffer (for image, usually only if you want grads wrt input)
    cl_mem grad_conv1_input_accum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * INPUT_PIXELS, NULL, &err);


    // ---- 5. Kernel creation ----
    cl_kernel conv1_kernel = clCreateKernel(program, "conv2d", &err);
    cl_kernel conv1_backward_kernel = clCreateKernel(program, "conv2d_backward", &err);
    cl_kernel pool1_kernel = clCreateKernel(program, "maxpool2d", &err);
    cl_kernel conv2_kernel = clCreateKernel(program, "conv2d", &err);
    cl_kernel conv2_backward_kernel = clCreateKernel(program, "conv2d_backward", &err);
    cl_kernel pool2_kernel = clCreateKernel(program, "maxpool2d", &err);
    cl_kernel maxpool2d_backward_kernel = clCreateKernel(program, "maxpool2d_backward", &err);
    cl_kernel dense_kernel = clCreateKernel(program, "dense_layer", &err);
    cl_kernel softmax_kernel = clCreateKernel(program, "softmax_parallel", &err);
    cl_kernel dense_backward_kernel = clCreateKernel(program, "dense_backward", &err);

    // ---- SGD Batch Versions of Backward Propagation kernels ----
    cl_kernel dense_backward_accum_kernel = clCreateKernel(program, "dense_backward_accum", &err);
    cl_kernel dense_update_kernel         = clCreateKernel(program, "dense_update", &err);
    cl_kernel conv2d_backward_accum_kernel = clCreateKernel(program, "conv2d_backward_accum", &err);
    cl_kernel conv2d_update_kernel         = clCreateKernel(program, "conv2d_update", &err);
    cl_kernel maxpool2d_backward_accum_kernel = clCreateKernel(program, "maxpool2d_backward_accum", &err);


    // ---- Set up int kernel args to avoid &MACRO bugs ----
    int input_size1      = INPUT_SIZE;
    int kernel1_size     = KERNEL1_SIZE;
    int conv1_out_size   = CONV1_OUTPUT_SIZE;
    int pool1_size       = POOL_SIZE;
    int pool1_out_size   = POOL1_OUTPUT_SIZE;

    int conv1_kernels    = CONV1_KERNELS;
    int conv2_kernels    = CONV2_KERNELS;

    int kernel2_size     = KERNEL2_SIZE;
    int conv2_out_size   = CONV2_OUTPUT_SIZE;
    int pool2_size       = POOL_SIZE;
    int pool2_out_size   = POOL2_OUTPUT_SIZE;

    int dense_in_val     = DENSE_INPUT_SIZE;
    int dense_out_val    = DENSE_OUTPUT_SIZE;

    int num_epochs = 500;


    // Adjustments to this number can have a lot of effects.
    // The deeper the network, the more sensitive it will be
    // to the value set here!
    float learning_rate = 0.001f;

    size_t conv1_gsize[3];
    size_t pool1_gsize[3];
    size_t conv2_gsize[3];
    size_t pool2_gsize[3];
    size_t dense_gsize;
    size_t sm_global, sm_local;

    float input2d[BATCH_SIZE][INPUT_PIXELS];
    int labels[BATCH_SIZE];

    int first_batch = 1;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        printf("==== Epoch %d ====\n", epoch+1);
        shuffle_array(trainData, train_size); // NOTE: use trainData/train_size here
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
            zero_buffer(queue, grad_conv1_weights_accum_buf, CONV1_KERNELS * 1 * KERNEL1_SIZE * KERNEL1_SIZE);
            zero_buffer(queue, grad_conv1_biases_accum_buf, CONV1_KERNELS);
            zero_buffer(queue, grad_conv2_weights_accum_buf, CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE);
            zero_buffer(queue, grad_conv2_biases_accum_buf, CONV2_KERNELS);
            zero_buffer(queue, grad_dense_weights_accum_buf, DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE);
            zero_buffer(queue, grad_dense_biases_accum_buf, DENSE_OUTPUT_SIZE);
            zero_buffer(queue, grad_pool2_input_accum_buf, CONV2_KERNELS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
            zero_buffer(queue, grad_pool1_input_accum_buf, CONV1_KERNELS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
            zero_buffer(queue, grad_conv1_input_accum_buf, INPUT_PIXELS);


            for (int i = 0; i < this_batch_size; ++i) {

                // --- Write one image to buffer for GPU usage ---
                clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0,
                    sizeof(float) * INPUT_PIXELS, input2d[i], 0, NULL, NULL);

                //
                // Forward Pass - process the image pixels as inputs to the network,
                // passing the output of each layer to the next layer until we reach
                // the output layer
                //

                // 1. Conv1
                conv2d_forward(queue, conv1_kernel, input_buf, conv1_weights_buf, conv1_biases_buf, conv1_output_buf, 1, INPUT_SIZE, CONV1_KERNELS, KERNEL1_SIZE, CONV1_OUTPUT_SIZE);

                // 2. Pool1
                maxpool2d_forward(queue, pool1_kernel, conv1_output_buf, pool1_output_buf, maxpool1_max_indices_buf, CONV1_KERNELS, CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE, POOL_SIZE, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE);

                // 3. Conv2
                conv2d_forward(queue, conv2_kernel, pool1_output_buf, conv2_weights_buf, conv2_biases_buf, conv2_output_buf, CONV1_KERNELS, POOL1_OUTPUT_SIZE, CONV2_KERNELS, KERNEL2_SIZE, CONV2_OUTPUT_SIZE);

                // 4. Pool2
                maxpool2d_forward(queue, pool2_kernel, conv2_output_buf, pool2_output_buf, maxpool2_max_indices_buf, CONV2_KERNELS, CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE, POOL_SIZE, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE);

                // 5. Dense
                dense_forward(queue, dense_kernel, pool2_output_buf, dense_weights_buf, dense_biases_buf, dense_output_buf, DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE);

                // 6. Softmax
                softmax_forward(queue, softmax_kernel, dense_output_buf, softmax_output_buf, DENSE_OUTPUT_SIZE);

                //
                // End of Forward Pass
                //

                // --- Read output
                float softmax_output[DENSE_OUTPUT_SIZE];
                clEnqueueReadBuffer(queue, softmax_output_buf, CL_TRUE, 0,
                    sizeof(softmax_output), softmax_output, 0, NULL, NULL);

                // --- Prediction & Loss
                int pred = argmax(softmax_output, DENSE_OUTPUT_SIZE);
                float loss = -logf(fmaxf(softmax_output[labels[i]], 1e-8f));
                total_loss += loss;
                if (pred == labels[i]) correct++;


                //
                // Training via backpropagation
                //

                // ---- Backward pass (accumulate) ----

                // DENSE LAYER BACKWARD (accumulate)
                dense_backward_accum(
                    queue, dense_backward_accum_kernel,
                    pool2_output_buf,        // Input to dense
                    grad_output_buf,         // [output_size], this sample's grad_output
                    grad_dense_weights_accum_buf, // ACCUMULATOR
                    grad_dense_biases_accum_buf,  // ACCUMULATOR
                    grad_input_accum_buf,         // Output: grad_input for previous layer
                    DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE
                );

                // -- MaxPool2 backward (accumulate, if needed) --
                maxpool2d_backward_accum(
                    queue, maxpool2d_backward_accum_kernel,
                    grad_input_accum_buf,         // grad_output from dense layer
                    grad_pool2_input_accum_buf,   // accum for pool2's input grads
                    maxpool2_max_indices_buf,     // from pool2 forward pass
                    CONV2_KERNELS,
                    POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE
                );

                // CONV2 LAYER BACKWARD (accumulate)
                conv2d_backward_accum(
                    queue, conv2d_backward_accum_kernel,
                    pool1_output_buf,                 // input to conv2 (from forward)
                    grad_pool2_input_accum_buf,       // grad_output (from pool2 back)
                    grad_conv2_weights_accum_buf,     // ACCUMULATOR
                    grad_conv2_biases_accum_buf,      // ACCUMULATOR
                    grad_pool1_input_accum_buf,       // accum for conv2's input grads
                    CONV1_KERNELS, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE,
                    CONV2_KERNELS, KERNEL2_SIZE,
                    CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE
                );

                // ...repeat for pool1/maxpool1 and conv1...

                // POOL1 BACKWARD (accumulate)
                maxpool2d_backward_accum(
                    queue, maxpool2d_backward_accum_kernel,
                    grad_pool1_input_accum_buf,   // grad_output from conv2 back
                    grad_conv1_input_accum_buf,   // accum for pool1's input gradients
                    maxpool1_max_indices_buf,     // from pool1 forward pass
                    CONV1_KERNELS,
                    POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE
                );

                // CONV1 LAYER BACKWARD (accumulate)
                conv2d_backward_accum(
                    queue, conv2d_backward_accum_kernel,
                    input_buf,                        // input to conv1 (from forward)
                    grad_conv1_input_accum_buf,       // grad_output (from pool1 back)
                    grad_conv1_weights_accum_buf,     // ACCUMULATOR
                    grad_conv1_biases_accum_buf,      // ACCUMULATOR
                    grad_input_accum_buf,             // (unused unless you want grads wrt image)
                    1, INPUT_SIZE, INPUT_SIZE,        // grayscale inC, H, W
                    CONV1_KERNELS, KERNEL1_SIZE,
                    CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE
                );

            }  // end of batch loop

            // ---- SGD update at end of batch ----
            dense_update(
                queue, dense_update_kernel,
                dense_weights_buf, dense_biases_buf,
                grad_dense_weights_accum_buf, grad_dense_biases_accum_buf,
                DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE,
                learning_rate, this_batch_size
            );
            conv2d_update(
                queue, conv2d_update_kernel,
                conv2_weights_buf, conv2_biases_buf,
                grad_conv2_weights_accum_buf, grad_conv2_biases_accum_buf,
                CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE, CONV2_KERNELS,
                learning_rate, this_batch_size
            );
            conv2d_update(
                queue, conv2d_update_kernel,
                conv1_weights_buf, conv1_biases_buf,
                grad_conv1_weights_accum_buf, grad_conv1_biases_accum_buf,
                CONV1_KERNELS * 1 * KERNEL1_SIZE * KERNEL1_SIZE, CONV1_KERNELS,
                learning_rate, this_batch_size
            );

        }  // end of minibatch loop


        // End of epoch: print stats
        float avg_loss = total_loss / train_size;
        float acc = 100.0f * correct / train_size;
        printf("Epoch %d Summary: Accuracy = %.2f%%, Avg Loss = %.4f\n\n", epoch+1, acc, avg_loss);

        // --- Print first 10 weights of Conv1 ---
        clEnqueueReadBuffer(queue, conv1_weights_buf, CL_TRUE, 0,
            sizeof(float) * CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE,
            conv1_weights, 0, NULL, NULL);

        printf("Conv1 weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < CONV1_KERNELS * KERNEL1_SIZE * KERNEL1_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)conv1_weights)[k]);

        // --- Print first 10 weights of Conv2 ---
        clEnqueueReadBuffer(queue, conv2_weights_buf, CL_TRUE, 0,
            sizeof(float) * CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE,
            conv2_weights, 0, NULL, NULL);

        printf("Conv2 weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < CONV2_KERNELS * CONV1_KERNELS * KERNEL2_SIZE * KERNEL2_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)conv2_weights)[k]);

        // --- Dense weights (if not already printing) ---
        clEnqueueReadBuffer(queue, dense_weights_buf, CL_TRUE, 0,
            sizeof(float) * DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE,
            dense_weights, 0, NULL, NULL);

        printf("Dense weights after epoch %d:\n", epoch+1);
        for (int k = 0; k < 10 && k < DENSE_OUTPUT_SIZE * DENSE_INPUT_SIZE; ++k)
            printf("  w[%d] = %f\n", k, ((float*)dense_weights)[k]);

        clEnqueueReadBuffer(queue, conv1_biases_buf, CL_TRUE, 0,
            sizeof(float) * CONV1_KERNELS, conv1_biases, 0, NULL, NULL);

        printf("Conv1 biases after epoch %d:\n", epoch+1);
        for (int k = 0; k < CONV1_KERNELS; ++k)
            printf("  b[%d] = %f\n", k, conv1_biases[k]);

        clEnqueueReadBuffer(queue, conv2_biases_buf, CL_TRUE, 0,
            sizeof(float) * CONV2_KERNELS, conv2_biases, 0, NULL, NULL);

        printf("Conv2 biases after epoch %d:\n", epoch+1);
        for (int k = 0; k < CONV2_KERNELS; ++k)
            printf("  b[%d] = %f\n", k, conv2_biases[k]);

        clEnqueueReadBuffer(queue, dense_biases_buf, CL_TRUE, 0,
            sizeof(float) * DENSE_OUTPUT_SIZE, dense_biases, 0, NULL, NULL);

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

            clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, sizeof(float) * INPUT_PIXELS, val_input, 0, NULL, NULL);

            conv2d_forward(queue, conv1_kernel, input_buf, conv1_weights_buf, conv1_biases_buf, conv1_output_buf, 1, INPUT_SIZE, CONV1_KERNELS, KERNEL1_SIZE, CONV1_OUTPUT_SIZE);
            maxpool2d_forward(queue, pool1_kernel, conv1_output_buf, pool1_output_buf, maxpool1_max_indices_buf, CONV1_KERNELS, CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE, POOL_SIZE, POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE);
            conv2d_forward(queue, conv2_kernel, pool1_output_buf, conv2_weights_buf, conv2_biases_buf, conv2_output_buf, CONV1_KERNELS, POOL1_OUTPUT_SIZE, CONV2_KERNELS, KERNEL2_SIZE, CONV2_OUTPUT_SIZE);
            maxpool2d_forward(queue, pool2_kernel, conv2_output_buf, pool2_output_buf, maxpool2_max_indices_buf, CONV2_KERNELS, CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE, POOL_SIZE, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE);
            dense_forward(queue, dense_kernel, pool2_output_buf, dense_weights_buf, dense_biases_buf, dense_output_buf, DENSE_INPUT_SIZE, DENSE_OUTPUT_SIZE);
            softmax_forward(queue, softmax_kernel, dense_output_buf, softmax_output_buf, DENSE_OUTPUT_SIZE);

            float softmax_output[DENSE_OUTPUT_SIZE];
            clEnqueueReadBuffer(queue, softmax_output_buf, CL_TRUE, 0,
                sizeof(softmax_output), softmax_output, 0, NULL, NULL);

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
    clReleaseMemObject(input_buf);
    clReleaseMemObject(conv1_weights_buf);
    clReleaseMemObject(conv1_biases_buf);
    clReleaseMemObject(conv1_output_buf);
    clReleaseMemObject(pool1_output_buf);
    clReleaseMemObject(conv2_weights_buf);
    clReleaseMemObject(conv2_biases_buf);
    clReleaseMemObject(conv2_output_buf);
    clReleaseMemObject(pool2_output_buf);
    clReleaseMemObject(dense_weights_buf);
    clReleaseMemObject(dense_biases_buf);
    clReleaseMemObject(dense_output_buf);
    clReleaseMemObject(softmax_output_buf);
    clReleaseMemObject(grad_output_buf);
    clReleaseMemObject(grad_input_accum_buf);
    clReleaseMemObject(maxpool1_max_indices_buf);
    clReleaseMemObject(maxpool2_max_indices_buf);
    clReleaseMemObject(grad_pool2_input_buf);
    clReleaseMemObject(grad_pool1_input_buf);
    clReleaseMemObject(grad_input_buf);
    clReleaseMemObject(grad_conv1_weights_accum_buf);
    clReleaseMemObject(grad_conv1_biases_accum_buf);
    clReleaseMemObject(grad_conv2_weights_accum_buf);
    clReleaseMemObject(grad_conv2_biases_accum_buf);
    clReleaseMemObject(grad_dense_weights_accum_buf);
    clReleaseMemObject(grad_dense_biases_accum_buf);
    clReleaseMemObject(grad_pool2_input_accum_buf);
    clReleaseMemObject(grad_pool1_input_accum_buf);
    clReleaseMemObject(grad_conv1_input_accum_buf);


    clReleaseKernel(conv1_kernel);
    clReleaseKernel(pool1_kernel);
    clReleaseKernel(conv2_kernel);
    clReleaseKernel(conv2_backward_kernel);
    clReleaseKernel(pool2_kernel);
    clReleaseKernel(maxpool2d_backward_kernel);
    clReleaseKernel(dense_kernel);
    clReleaseKernel(softmax_kernel);
    clReleaseKernel(dense_backward_kernel);
    clReleaseKernel(dense_backward_accum_kernel);
    clReleaseKernel(dense_update_kernel);
    clReleaseKernel(conv2d_backward_accum_kernel);
    clReleaseKernel(conv2d_update_kernel);
    clReleaseKernel(maxpool2d_backward_accum_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
