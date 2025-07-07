#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

void print_device_type(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_CPU)
        printf("CPU ");
    if (type & CL_DEVICE_TYPE_GPU)
        printf("GPU ");
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        printf("Accelerator ");
    if (type & CL_DEVICE_TYPE_DEFAULT)
        printf("Default ");
    if (type & CL_DEVICE_TYPE_CUSTOM)
        printf("Custom ");
}

int main() {
    cl_uint platformCount;
    clGetPlatformIDs(0, NULL, &platformCount);
    printf("Number of OpenCL platforms: %u\n", platformCount);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (cl_uint i = 0; i < platformCount; ++i) {
        char name[128], vendor[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
        printf("\nPlatform %u: %s (%s)\n", i, name, vendor);

        // Get all devices on this platform
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        if (deviceCount == 0) {
            printf("  No devices found on this platform.\n");
            continue;
        }

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        for (cl_uint j = 0; j < deviceCount; ++j) {
            char deviceName[128];
            cl_device_type deviceType;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);

            printf("  Device %u: %s (", j, deviceName);
            print_device_type(deviceType);
            printf(")\n");
        }

        free(devices);
    }

    free(platforms);
    return 0;
}
