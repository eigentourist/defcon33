#include <stdio.h>
#define CL_TARGET_OPENCL_VERSION 120  // for OpenCL 1.2
#include <CL/cl.h>

int main() {
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    printf("Number of OpenCL platforms: %u\n", platformCount);
    return 0;
}
