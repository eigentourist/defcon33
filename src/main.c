#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

int main() {
    cl_uint platformCount = 0;
    cl_platform_id platform = NULL;

    // Get platform count
    clGetPlatformIDs(1, &platform, &platformCount);
    if (platformCount == 0) {
        printf("No OpenCL platforms found.\n");
        return 1;
    }

    // Get device
    cl_device_id device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get OpenCL device.\n");
        return 1;
    }

    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("OpenCL device: %s\n", deviceName);

    // Load kernel source from file
    FILE* fp = fopen("src/life.cl", "r");
    if (!fp) {
        printf("Failed to load kernel file.\n");
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    size_t sourceSize = ftell(fp);
    rewind(fp);
    char* sourceStr = (char*)malloc(sourceSize + 1);
    fread(sourceStr, 1, sourceSize, fp);
    sourceStr[sourceSize] = '\0';
    fclose(fp);

    // Create OpenCL context and program
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&sourceStr, &sourceSize, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to build OpenCL program.\n");
        return 1;
    }

    printf("OpenCL kernel compiled successfully.\n");

    // Cleanup
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(sourceStr);

    return 0;
}
