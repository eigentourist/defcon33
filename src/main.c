#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 32
#define HEIGHT 16
#define STEPS 100
#define DELAY_US 120000  // Slowed down from 100ms to 120ms

void load_oscillator(int* board) {
    // Clear the board
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        board[i] = 0;

    // Add a "blinker" (3-cell vertical oscillator)
    int cx = WIDTH / 2;
    int cy = HEIGHT / 2;
    board[(cy - 1) * WIDTH + cx] = 1;
    board[(cy    ) * WIDTH + cx] = 1;
    board[(cy + 1) * WIDTH + cx] = 1;
}

void load_glider(int* board) {
    // Clear board
    for (int i = 0; i < WIDTH * HEIGHT; ++i)
        board[i] = 0;

    // Add a glider near top-left corner
    // Pattern:
    // . O .
    // . . O
    // O O O

    int x = 1, y = 1;
    board[(y + 0) * WIDTH + (x + 1)] = 1;
    board[(y + 1) * WIDTH + (x + 2)] = 1;
    board[(y + 2) * WIDTH + (x + 0)] = 1;
    board[(y + 2) * WIDTH + (x + 1)] = 1;
    board[(y + 2) * WIDTH + (x + 2)] = 1;
}


int main() {
    size_t board_size = WIDTH * HEIGHT * sizeof(int);
    int* board_in = (int*)malloc(board_size);
    int* board_out = (int*)malloc(board_size);

    load_glider(board_in);

    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;

    // OpenCL setup
    // Get all platforms
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        exit(1);
    }

    // Get number of devices of all types
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "No OpenCL devices found.\n");
        exit(1);
    }

    cl_device_id *devices = malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get OpenCL devices.\n");
        free(devices);
        exit(1);
    }

    // Prefer GPU over CPU
    cl_device_id selected_device = NULL;
    cl_device_type selected_type = 0;
    for (cl_uint i = 0; i < num_devices; ++i) {
        cl_device_type dtype;
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, NULL);
        if (dtype == CL_DEVICE_TYPE_GPU) {
            selected_device = devices[i];
            selected_type = dtype;
            break;
        } else if (dtype == CL_DEVICE_TYPE_CPU && !selected_device) {
            selected_device = devices[i];
            selected_type = dtype;
        }
    }

    if (!selected_device) {
        fprintf(stderr, "No suitable OpenCL device found.\n");
        free(devices);
        exit(1);
    }

    // Print device type
    char device_name[128];
    clGetDeviceInfo(selected_device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    const char *type_str = (selected_type == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
    printf("OpenCL device: %s (%s)\n", device_name, type_str);

    device = selected_device;
    free(devices);

    // Now proceed to create context, command queue, etc. using 'device'
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    FILE* fp = fopen("src/life.cl", "r");
    if (!fp) {
        endwin();
        printf("Failed to load kernel.\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t src_size = ftell(fp);
    rewind(fp);
    char* src = (char*)malloc(src_size + 1);
    fread(src, 1, src_size, fp);
    src[src_size] = '\0';
    fclose(fp);

    program = clCreateProgramWithSource(context, 1, (const char**)&src, &src_size, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to build OpenCL program.\n");
        return 1;
    }
    free(src);

    kernel = clCreateKernel(program, "life_step", &err);
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_WRITE, board_size, NULL, &err);
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_READ_WRITE, board_size, NULL, &err);

    size_t global_work_size[2] = { WIDTH, HEIGHT };

    // ncurses setup
    initscr();
    noecho();
    curs_set(FALSE);
    timeout(0);
    keypad(stdscr, TRUE);

    // Enable color if available
    if (has_colors()) {
        start_color();
        init_pair(1, COLOR_GREEN, COLOR_BLACK);  // Alive cell
        init_pair(2, COLOR_RED, COLOR_BLACK);    // Dead cell
    }

    for (int step = 0; step < STEPS; ++step) {
        clEnqueueWriteBuffer(queue, buf_in, CL_TRUE, 0, board_size, board_in, 0, NULL, NULL);

        int w = WIDTH, h = HEIGHT;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_in);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_out);
        clSetKernelArg(kernel, 2, sizeof(int), &w);
        clSetKernelArg(kernel, 3, sizeof(int), &h);

        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        clFinish(queue);

        clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, board_size, board_out, 0, NULL, NULL);

        clear();
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                int val = board_out[y * WIDTH + x];
                if (has_colors()) {
                    attron(COLOR_PAIR(val ? 1 : 2));
                }
                mvaddch(y, x, val ? 'O' : '.');
                if (has_colors()) {
                    attroff(COLOR_PAIR(val ? 1 : 2));
                }
            }
        }
        mvprintw(HEIGHT, 0, "Step %d (press 'q' to quit)", step + 1);
        refresh();

        int ch = getch();
        if (ch == 'q' || ch == 'Q') break;

        usleep(DELAY_US);
        int* tmp = board_in;
        board_in = board_out;
        board_out = tmp;
    }

    endwin();

    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(board_in);
    free(board_out);

    return 0;
}
