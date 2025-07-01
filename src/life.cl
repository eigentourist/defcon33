__kernel void life_step(__global const int* in, __global int* out, int width, int height) {
    int gid = get_global_id(0);
    // Placeholder: just copy data
    out[gid] = in[gid];
}
