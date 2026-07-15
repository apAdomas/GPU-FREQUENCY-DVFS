#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <nvml.h>

#define GPU_DEVICE 0

static inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static inline void checkNvml(nvmlReturn_t err, const char* msg)
{
    if (err != NVML_SUCCESS) {
        fprintf(stderr, "NVML error at %s: %s\n", msg, nvmlErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static inline double now_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static inline void GPU_argv_init_measurement()
{
    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE), "cudaGetDeviceProperties");
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    checkCuda(cudaSetDevice(GPU_DEVICE), "cudaSetDevice");
}