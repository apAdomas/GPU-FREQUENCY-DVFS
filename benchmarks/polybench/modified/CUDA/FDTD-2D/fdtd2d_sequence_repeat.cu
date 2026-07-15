/**
 * fdtd2d_sequence_repeat.cu
 *
 * Purpose:
 * - run the 20-step FDTD kernel sequence repeatedly
 * - reset GPU arrays before each repeat
 * - stop after enough sequence runtime has accumulated
 * - no per-kernel timing instrumentation
 * - intended for energy measurement runs
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "fdtd2d.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define PERCENT_DIFF_ERROR_THRESHOLD 10.05
#define GPU_DEVICE 0
#define RUN_ON_CPU

#ifndef SEQUENCE_TMAX
#define SEQUENCE_TMAX 20
#endif

#ifndef TARGET_SEQUENCE_TIME_MS
#define TARGET_SEQUENCE_TIME_MS 5000.0f
#endif

void init_arrays(
    int tmax,
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
    DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int i, j;

    for (i = 0; i < tmax; i++) {
        _fict_[i] = (DATA_TYPE)i;
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            ex[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i][j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i][j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

void runFdtd(
    int tmax,
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
    DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int t, i, j;

    for (t = 0; t < tmax; t++) {
        for (j = 0; j < _PB_NY; j++) {
            ey[0][j] = _fict_[t];
        }

        for (i = 1; i < _PB_NX; i++) {
            for (j = 0; j < _PB_NY; j++) {
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[(i - 1)][j]);
            }
        }

        for (i = 0; i < _PB_NX; i++) {
            for (j = 1; j < _PB_NY; j++) {
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][(j - 1)]);
            }
        }

        for (i = 0; i < _PB_NX - 1; i++) {
            for (j = 0; j < _PB_NY - 1; j++) {
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][(j + 1)] - ex[i][j] + ey[(i + 1)][j] - ey[i][j]);
            }
        }
    }
}

void compareResults(
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_2D(hz1, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz2, NX, NY, nx, ny))
{
    int i, j, fail;
    fail = 0;

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            if (percentDiff(hz1[i][j], hz2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }

    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n",
           PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

static void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void fdtd_step1_kernel(int nx, int ny, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NX) && (j < _PB_NY)) {
        if (i == 0) {
            ey[i * NY + j] = _fict_[t];
        } else {
            ey[i * NY + j] = ey[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
        }
    }
}

__global__ void fdtd_step2_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NX) && (j < _PB_NY) && (j > 0)) {
        ex[i * NY + j] = ex[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}

__global__ void fdtd_step3_kernel(int nx, int ny, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < (_PB_NX - 1)) && (j < (_PB_NY - 1))) {
        hz[i * NY + j] = hz[i * NY + j] - 0.7f * (ex[i * NY + (j + 1)] - ex[i * NY + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}

void fdtdCuda(
    int tmax,
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
    DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz_outputFromGpu, NX, NY, nx, ny))
{
    DATA_TYPE* _fict_gpu;
    DATA_TYPE* ex_gpu;
    DATA_TYPE* ey_gpu;
    DATA_TYPE* hz_gpu;

    checkCuda(cudaMalloc((void**)&_fict_gpu, sizeof(DATA_TYPE) * tmax), "cudaMalloc _fict_gpu");
    checkCuda(cudaMalloc((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ex_gpu");
    checkCuda(cudaMalloc((void**)&ey_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ey_gpu");
    checkCuda(cudaMalloc((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc hz_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil(((float)NY) / ((float)block.x)),
              (size_t)ceil(((float)NX) / ((float)block.y)));

    cudaEvent_t repeat_start, repeat_stop;
    checkCuda(cudaEventCreate(&repeat_start), "cudaEventCreate repeat_start");
    checkCuda(cudaEventCreate(&repeat_stop), "cudaEventCreate repeat_stop");

    float total_sequence_ms = 0.0f;
    int completed_repeats = 0;

    polybench_start_instruments;

    while (total_sequence_ms < TARGET_SEQUENCE_TIME_MS) {
        checkCuda(cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice), "reset _fict_gpu");
        checkCuda(cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "reset ex_gpu");
        checkCuda(cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "reset ey_gpu");
        checkCuda(cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "reset hz_gpu");
        checkCuda(cudaDeviceSynchronize(), "sync after reset");

        checkCuda(cudaEventRecord(repeat_start), "record repeat_start");

        for (int t = 0; t < tmax; t++) {
            fdtd_step1_kernel<<<grid, block>>>(nx, ny, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
            checkCuda(cudaGetLastError(), "launch step1");

            fdtd_step2_kernel<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu, t);
            checkCuda(cudaGetLastError(), "launch step2");

            fdtd_step3_kernel<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu, t);
            checkCuda(cudaGetLastError(), "launch step3");
        }

        checkCuda(cudaEventRecord(repeat_stop), "record repeat_stop");
        checkCuda(cudaEventSynchronize(repeat_stop), "sync repeat_stop");

        float repeat_ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&repeat_ms, repeat_start, repeat_stop), "elapsed repeat");
        total_sequence_ms += repeat_ms;
        completed_repeats++;
    }

    polybench_stop_instruments;

    printf("GPU experiment time in seconds:\n");
    polybench_print_instruments;
    printf("Accumulated sequence GPU time in ms: %.3f\n", total_sequence_ms);
    printf("Completed repeats: %d\n", completed_repeats);

    checkCuda(cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost), "cudaMemcpy hz_outputFromGpu");

    checkCuda(cudaEventDestroy(repeat_start), "destroy repeat_start");
    checkCuda(cudaEventDestroy(repeat_stop), "destroy repeat_stop");

    cudaFree(_fict_gpu);
    cudaFree(ex_gpu);
    cudaFree(ey_gpu);
    cudaFree(hz_gpu);
}

/* DCE code. Must scan the entire live-out data. */
static void print_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int i, j;

    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
            if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main()
{
    int tmax = SEQUENCE_TMAX;
    int nx = NX;
    int ny = NY;

    POLYBENCH_1D_ARRAY_DECL(_fict_, DATA_TYPE, TMAX, TMAX);
    POLYBENCH_2D_ARRAY_DECL(ex, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(ey, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(hz, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu, DATA_TYPE, NX, NY, nx, ny);

    init_arrays(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

    GPU_argv_init();
    fdtdCuda(tmax, nx, ny,
             POLYBENCH_ARRAY(_fict_),
             POLYBENCH_ARRAY(ex),
             POLYBENCH_ARRAY(ey),
             POLYBENCH_ARRAY(hz),
             POLYBENCH_ARRAY(hz_outputFromGpu));

#ifdef RUN_ON_CPU
    polybench_start_instruments;
    runFdtd(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(nx, ny, POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));
#else
    print_array(nx, ny, POLYBENCH_ARRAY(hz_outputFromGpu));
#endif

    POLYBENCH_FREE_ARRAY(_fict_);
    POLYBENCH_FREE_ARRAY(ex);
    POLYBENCH_FREE_ARRAY(ey);
    POLYBENCH_FREE_ARRAY(hz);
    POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"