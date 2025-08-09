#define _XOPEN_SOURCE 700   // or: #define _POSIX_C_SOURCE 200809L
#include <unistd.h>         // for usleep()

#include <ncurses.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef NPTS
#define NPTS 120
#endif

typedef struct { float x, y; } Pt;

static Pt data[NPTS];
static float m = 0.0f, b = 0.0f;     // line: y = m x + b (normalized 0..1)
static float lr = 0.1f;              // learning rate
static float loss = 0.0f;            // MSE
static int paused = 0;

static float frand(void) { return (float)rand() / (float)RAND_MAX; }

static void gen_data(void) {
    // True line + noise (in [0,1])
    float m_true = 0.5f + 0.8f * frand();
    float b_true = 0.1f * frand();
    float noise  = 0.06f + 0.04f * frand();
    for (int i = 0; i < NPTS; ++i) {
        float x = frand();
        // Gaussian-ish noise via CLT
        float n = 0.f;
        for (int k = 0; k < 6; ++k) n += frand();
        n = noise * (n/3.f - 1.f);
        float y = m_true * x + b_true + n;
        if (y < 0.f) y = 0.f;
        if (y > 1.f) y = 1.f;
        data[i].x = x;
        data[i].y = y;
    }
    m = 0.f;
    b = 0.f;
    loss = 0.f;
}

static void step_gd(void) {
    float dLdm = 0.f, dLdb = 0.f, L = 0.f;
    for (int i = 0; i < NPTS; ++i) {
        float x = data[i].x, y = data[i].y;
        float yhat = m * x + b;
        float e = yhat - y;
        L += e*e;
        dLdm += 2.f * e * x;
        dLdb += 2.f * e;
    }
    L /= (float)NPTS;
    dLdm /= (float)NPTS;
    dLdb /= (float)NPTS;
    m -= lr * dLdm;
    b -= lr * dLdb;
    loss = L;
}

static void draw_scene(void) {
    int H, W; getmaxyx(stdscr, H, W);
    int plotH = H - 2;
    if (plotH < 4 || W < 20) return;

    erase();

    for (int x = 0; x < W; ++x) {
        mvaddch(0, x, '-');
        mvaddch(plotH-1, x, '-');
    }
    for (int y = 0; y < plotH; ++y) {
        mvaddch(y, 0, '|');
        mvaddch(y, W-1, '|');
    }

    for (int i = 0; i < NPTS; ++i) {
        int px = 1 + (int)((W-2) * data[i].x + 0.5f);
        int py = 1 + (int)((plotH-2) * (1.f - data[i].y) + 0.5f);
        if (px > 0 && px < W && py > 0 && py < plotH) {
            mvaddch(py, px, '.');
        }
    }

    for (int x = 1; x < W-1; ++x) {
        float xn = (float)(x-1) / (float)(W-2);
        float yn = m * xn + b;
        int py = 1 + (int)((plotH-2) * (1.f - yn) + 0.5f);
        if (py > 0 && py < plotH) mvaddch(py, x, '*');
    }

    mvprintw(plotH, 0, "m=%.3f  b=%.3f  lr=%.3f  loss=%.5f  %s   [+, -] lr  [p] pause  [r] reseed  [q] quit",
             m, b, lr, loss, paused ? "[PAUSED]" : "       ");
    clrtoeol();
    refresh();
}

int main(void) {
    srand((unsigned)time(NULL));
    gen_data();

    initscr();
    noecho();
    curs_set(FALSE);
    timeout(0);
    keypad(stdscr, TRUE);

    if (has_colors()) {
        start_color();
        init_pair(1, COLOR_GREEN, COLOR_BLACK);
        init_pair(2, COLOR_CYAN,  COLOR_BLACK);
    }

    const unsigned int dt_us = 20000; // ~50 FPS
    int tick = 0;

    while (1) {
        int ch = getch();
        if (ch == 'q' || ch == 'Q') break;
        if (ch == 'p' || ch == 'P') paused = !paused;
        if (ch == '+' || ch == '=') lr *= 1.1f;
        if (ch == '-' || ch == '_') lr /= 1.1f;
        if (ch == 'r' || ch == 'R') gen_data();

        if (!paused) {
            for (int k = 0; k < 3; ++k) step_gd();
        }

        draw_scene();
        (void)tick;
        tick++;
        usleep(dt_us);
    }

    endwin();
    return 0;
}

