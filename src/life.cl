__kernel void life_step(__global const int* in, __global int* out, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * width + x;

    int count = 0;

    // Loop over 3x3 neighborhood with wrapping
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            int nidx = ny * width + nx;
            count += in[nidx];
        }
    }

    int cell = in[idx];
    if (cell == 1 && (count == 2 || count == 3)) {
        out[idx] = 1;
    } else if (cell == 0 && count == 3) {
        out[idx] = 1;
    } else {
        out[idx] = 0;
    }
}
