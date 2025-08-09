#include "life_kernel.cuh"
#include <cuda_runtime.h>
#include <ncurses.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static double time_in_seconds() {
    timeval tv{}; gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static void load_glider(int* board, int W, int H) {
    std::fill(board, board + W*H, 0);
    int x = 1, y = 1;
    board[(y+0)*W + (x+1)] = 1;
    board[(y+1)*W + (x+2)] = 1;
    board[(y+2)*W + (x+0)] = 1;
    board[(y+2)*W + (x+1)] = 1;
    board[(y+2)*W + (x+2)] = 1;
}

static inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { endwin(); std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main(int argc, char** argv) {
    const int WIDTH = 32, HEIGHT = 16, STEPS = 100;
    const useconds_t DELAY_US = 120000;
    bool benchmark_mode = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--benchmark") == 0) benchmark_mode = true;

    const size_t N = size_t(WIDTH) * HEIGHT, BYTES = N * sizeof(int);
    int* h_in  = (int*)std::malloc(BYTES);
    int* h_out = (int*)std::malloc(BYTES);
    if (!h_in || !h_out) { std::fprintf(stderr, "malloc failed\n"); return 1; }
    load_glider(h_in, WIDTH, HEIGHT);

    int *d_a = nullptr, *d_b = nullptr;
    check(cudaMalloc(&d_a, BYTES), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, BYTES), "cudaMalloc d_b");
    check(cudaMemcpy(d_a, h_in, BYTES, cudaMemcpyHostToDevice), "H2D init");

    initscr(); noecho(); curs_set(FALSE); timeout(0); keypad(stdscr, TRUE);
    if (has_colors()) { start_color(); init_pair(1, COLOR_GREEN, COLOR_BLACK); init_pair(2, COLOR_RED, COLOR_BLACK); }

    double prev_ts = benchmark_mode ? time_in_seconds() : 0.0, bench_start = prev_ts;

    for (int step = 0; step < STEPS; ++step) {
        life_step_launch(d_a, d_b, WIDTH, HEIGHT);
        check(cudaGetLastError(), "kernel launch");
        std::swap(d_a, d_b);

        check(cudaMemcpy(h_out, d_a, BYTES, cudaMemcpyDeviceToHost), "D2H frame");

        clear();
        for (int y = 0; y < HEIGHT; ++y) for (int x = 0; x < WIDTH; ++x) {
            int val = h_out[y * WIDTH + x];
            if (has_colors()) attron(COLOR_PAIR(val ? 1 : 2));
            mvaddch(y, x, val ? 'O' : '.');
            if (has_colors()) attroff(COLOR_PAIR(val ? 1 : 2));
        }

        if (benchmark_mode) {
            double now = time_in_seconds(), dt = now - prev_ts;
            double fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
            mvprintw(HEIGHT, 0, "Step %d (%.1f FPS, 'q' to quit)", step + 1, fps);
            prev_ts = now;
        } else {
            mvprintw(HEIGHT, 0, "Step %d ('q' to quit)", step + 1);
        }

        refresh();
        int ch = getch(); if (ch == 'q' || ch == 'Q') break;
        if (!benchmark_mode) usleep(DELAY_US);
    }

    endwin();
    if (benchmark_mode) {
        double elapsed = time_in_seconds() - bench_start;
        double fps = (elapsed > 0.0) ? (STEPS / elapsed) : 0.0;
        std::printf("Benchmark: %.2f s, %.2f FPS\n", elapsed, fps);
    }

    cudaFree(d_a); cudaFree(d_b); std::free(h_in); std::free(h_out);
    return 0;
}

