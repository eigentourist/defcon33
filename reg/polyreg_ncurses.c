// polyreg_ncurses.c — polynomial regression (GD) with ncurses
// Adds: L2 (ridge) regularization + mini-batch/SGD

#define _XOPEN_SOURCE 700   // or: #define _POSIX_C_SOURCE 200809L
#include <unistd.h>         // for usleep()

#include <ncurses.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef NPTS
#define NPTS 160
#endif

#ifndef MAX_DEG
#define MAX_DEG 9
#endif

typedef struct { float x, y; } Pt;

static Pt   data[NPTS];
static float a[MAX_DEG+1];     // coefficients a0..aD
static int   D = 3;            // degree
static float lr = 0.05f;       // learning rate
static float loss = 0.0f;      // reported loss (MSE (+ L2 if on))
static int   paused = 0;

// L2 (ridge)
static int   l2_on = 0;
static float lambda_ = 0.01f;  // regularization strength

// Mini-batch
static int   batch_sz = NPTS;  // cycles: N, 32, 8, 1

static float frand01(void) { return (float)rand() / (float)RAND_MAX; }
static float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

static float poly_eval_xp(float x, const float *coeff, int deg) {
    // Horner's
    float y = coeff[deg];
    for (int p = deg-1; p >= 0; --p) y = coeff[p] + x * y;
    return y;
}

static void clear_coeffs(void) {
    for (int i = 0; i <= MAX_DEG; ++i) a[i] = 0.f;
    loss = 0.f;
}

static void gen_data(void) {
    int true_deg = 2 + rand() % 3; // 2..4
    float c[MAX_DEG+1] = {0};
    for (int i = 0; i <= true_deg; ++i)
        c[i] = (frand01()*2.f - 1.f) * (i==0 ? 0.4f : (i==1 ? 0.8f : 0.3f));
    c[0] = clamp01(0.1f + 0.2f*(frand01()-0.5f));

    float noise = 0.06f + 0.04f * frand01();

    for (int i = 0; i < NPTS; ++i) {
        float x = frand01();
        float n = 0.f; for (int k = 0; k < 6; ++k) n += frand01();
        n = noise * (n/3.f - 1.f);
        float y = poly_eval_xp(x, c, true_deg);
        y = clamp01(y + n);
        data[i].x = x; data[i].y = y;
    }
    clear_coeffs();
}

// one GD step over a random mini-batch of size batch_sz
static void step_gd_batch(void) {
    double grad[MAX_DEG+1] = {0.0};
    double L = 0.0;

    // random contiguous window (simple, fine for demo)
    int start = (batch_sz >= NPTS) ? 0 : rand() % (NPTS - batch_sz + 1);

    for (int t = 0; t < batch_sz; ++t) {
        int i = start + t;
        float x = data[i].x, y = data[i].y;

        // yhat and accumulate powers
        float xp = 1.f, yhat = 0.f;
        for (int p = 0; p <= D; ++p) { yhat += a[p] * xp; xp *= x; }

        float e = yhat - y;
        L += (double)e * (double)e;

        xp = 1.f;
        for (int p = 0; p <= D; ++p) {
            grad[p] += (double)e * (double)xp * 2.0;
            xp *= x;
        }
    }

    // average over batch
    double invB = 1.0 / (double)batch_sz;

    // L2 term: add lambda * a[p] to gradient; add 0.5*lambda*||a||^2 to loss (for display)
    double l2_pen = 0.0;
    if (l2_on) {
        for (int p = 0; p <= D; ++p) {
            grad[p] += (double)lambda_ * (double)a[p];
            l2_pen += 0.5 * (double)lambda_ * (double)a[p] * (double)a[p];
        }
    }

    for (int p = 0; p <= D; ++p) {
        float g = (float)(grad[p] * invB);
        a[p] -= lr * g;
    }

    // report full-dataset style scale: show average per-sample (batch MSE) + L2 penalty
    loss = (float)((L * invB) + l2_pen);
}

static void draw_scene(void) {
    int H, W; getmaxyx(stdscr, H, W);
    int plotH = H - 3; // leave room for two HUD lines
    if (plotH < 5 || W < 40) return;

    erase();

    for (int x = 0; x < W; ++x) { mvaddch(0, x, '-'); mvaddch(plotH-1, x, '-'); }
    for (int y = 0; y < plotH; ++y) { mvaddch(y, 0, '|'); mvaddch(y, W-1, '|'); }

    for (int i = 0; i < NPTS; ++i) {
        int px = 1 + (int)((W-2) * data[i].x + 0.5f);
        int py = 1 + (int)((plotH-2) * (1.f - data[i].y) + 0.5f);
        if (px > 0 && px < W && py > 0 && py < plotH) mvaddch(py, px, '.');
    }

    for (int x = 1; x < W-1; ++x) {
        float xn = (float)(x-1) / (float)(W-2);
        // Horner on current a[0..D]
        float yn = a[D];
        for (int p = D-1; p >= 0; --p) yn = a[p] + xn * yn;
        yn = clamp01(yn);
        int py = 1 + (int)((plotH-2) * (1.f - yn) + 0.5f);
        if (py > 0 && py < plotH) mvaddch(py, x, '*');
    }

    mvprintw(plotH, 0,
      "deg=%d  lr=%.3f  loss=%.5f  batch=%d  L2:%s lam=%.4f  %s",
      D, lr, loss, batch_sz, l2_on ? "ON " : "off", lambda_, paused ? "[PAUSED]" : "       ");
    clrtoeol();

    mvprintw(plotH+1, 0,
      "[, .] lam  [l] L2 on/off  [m] cycle batch  [ , ] deg  [1-9] set deg  [+, -] lr  [p] pause  [r] reseed  [c] clear  [q] quit");
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

    const unsigned int dt_us = 20000; // ~50 FPS

    while (1) {
        int ch = getch();
        if (ch == 'q' || ch == 'Q') break;
        if (ch == 'p' || ch == 'P') paused = !paused;
        if (ch == '+' || ch == '=') lr *= 1.1f;
        if (ch == '-' || ch == '_') lr /= 1.1f;
        if (ch == 'r' || ch == 'R') gen_data();
        if (ch == 'c' || ch == 'C') clear_coeffs();

        // degree controls
        if (ch == '[') { if (D > 0) D--; }
        if (ch == ']') { if (D < MAX_DEG) D++; }
        if (ch >= '1' && ch <= '9') { int nd = ch - '0'; if (nd > MAX_DEG) nd = MAX_DEG; D = nd; }

        // L2 controls
        if (ch == 'l' || ch == 'L') l2_on = !l2_on;
        if (ch == ',') { lambda_ *= 0.5f; if (lambda_ < 1e-6f) lambda_ = 1e-6f; }
        if (ch == '.') { lambda_ *= 2.0f; if (lambda_ > 10.f)  lambda_ = 10.f; }

        // batch controls
        if (ch == 'm' || ch == 'M') {
            if (batch_sz == NPTS) batch_sz = 32;
            else if (batch_sz == 32) batch_sz = 8;
            else if (batch_sz == 8)  batch_sz = 1;
            else batch_sz = NPTS;
        }

        if (!paused) {
            // a few steps per frame for visible motion
            for (int k = 0; k < 3; ++k) step_gd_batch();
        }

        draw_scene();
        usleep(dt_us);
    }

    endwin();
    return 0;
}

