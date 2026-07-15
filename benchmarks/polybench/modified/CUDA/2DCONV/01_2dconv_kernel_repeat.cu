/**
 * 01_2dconv_kernel_repeat.cu
 *
 * Isolated measurement for convolution2D_kernel.
 *
 * Measures a single convolution2D_kernel launch
 * repeated many times for energy/time measurement.
 *
 * Protocol:
 * 1) allocate once
 * 2) copy/reset inputs to device
 * 3) warm up with repeated single launches
 * 4) reset inputs
 * 5) measure repeated single launches
 *    - CUDA events for runtime
 *    - NVML total energy for energy
 * 6) copy back once
 * 7) exit
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#define POLYBENCH_TIME 1

#include "2DConvolution.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

void init_arrays(
    int ni,
    int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
    int i, j;

    srand(0);

    for (i = 0; i < ni; ++i)
    {
        for (j = 0; j < nj; ++j)
        {
            A[i][j] = (DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX;
            B[i][j] = 0.0f;
        }
    }
}

__global__ void convolution2D_kernel(int ni, int nj, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +0.2f;  c21 = +0.5f;  c31 = -0.8f;
    c12 = -0.3f;  c22 = +0.6f;  c32 = -0.9f;
    c13 = +0.4f;  c23 = +0.7f;  c33 = +0.10f;

    if ((i < _PB_NI - 1) && (j < _PB_NJ - 1) && (i > 0) && (j > 0))
    {
        B[i * NJ + j] =
              c11 * A[(i - 1) * NJ + (j - 1)] + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)]
            + c12 * A[(i + 0) * NJ + (j - 1)] + c22 * A[(i + 0) * NJ + (j + 0)] + c32 * A[(i + 0) * NJ + (j + 1)]
            + c13 * A[(i + 1) * NJ + (j - 1)] + c23 * A[(i + 1) * NJ + (j + 0)] + c33 * A[(i + 1) * NJ + (j + 1)];
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    DATA_TYPE (*A)[NJ],
    DATA_TYPE (*B)[NJ])
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            B_gpu,
            B,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy B");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel_once(
    int ni,
    int nj,
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    dim3 grid,
    dim3 block)
{
    convolution2D_kernel<<<grid, block>>>(ni, nj, A_gpu, B_gpu);
    checkCuda(cudaGetLastError(), "launch convolution2D_kernel");
}

void convolution2DCudaKernelOnly(
    int ni,
    int nj,
    DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
    DATA_TYPE* A_gpu = NULL;
    DATA_TYPE* B_gpu = NULL;

    checkCuda(cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NI * NJ), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void**)&B_gpu, sizeof(DATA_TYPE) * NI * NJ), "cudaMalloc B_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NJ) / ((float)block.x)),
        (size_t)ceil(((float)NI) / ((float)block.y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, B_gpu, A, B);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel_once(ni, nj, A_gpu, B_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, B_gpu, A, B);

    cudaEvent_t measure_start, measure_stop;
    checkCuda(cudaEventCreate(&measure_start), "cudaEventCreate measure_start");
    checkCuda(cudaEventCreate(&measure_stop), "cudaEventCreate measure_stop");

    int measured_launches = 0;
    float measured_cuda_ms = 0.0f;

    checkCuda(cudaDeviceSynchronize(), "sync before measure");
    checkNvml(nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_start_mj), "energy_start");
    checkCuda(cudaEventRecord(measure_start), "record measure_start");

#ifdef NCU_PROFILE
    cudaProfilerStart();
#endif

    {
        double measure_wall_start = now_seconds();
        while ((now_seconds() - measure_wall_start) < MEASURE_SECONDS)
        {
            launch_kernel_once(ni, nj, A_gpu, B_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after measured launch");
            measured_launches++;
        }
    }

#ifdef NCU_PROFILE
    cudaProfilerStop();
#endif

    checkCuda(cudaEventRecord(measure_stop), "record measure_stop");
    checkCuda(cudaEventSynchronize(measure_stop), "sync measure_stop");
    checkCuda(cudaDeviceSynchronize(), "final sync before energy_end");
    checkNvml(nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_end_mj), "energy_end");

    checkCuda(cudaEventElapsedTime(&measured_cuda_ms, measure_start, measure_stop), "elapsed measure");

    checkCuda(
        cudaMemcpy(
            B_outputFromGpu,
            B_gpu,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyDeviceToHost),
        "copy B_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=convolution2D_kernel\n");
        printf("RESULT warmup_seconds_target=%.3f\n", (double)WARMUP_SECONDS);
        printf("RESULT measure_seconds_target=%.3f\n", (double)MEASURE_SECONDS);
        printf("RESULT warmup_launches=%d\n", warmup_launches);
        printf("RESULT measured_launches=%d\n", measured_launches);
        printf("RESULT measured_cuda_time_ms=%.3f\n", measured_cuda_ms);
        printf("RESULT measured_cuda_time_s=%.6f\n", measured_cuda_s);
        printf("RESULT measured_energy_mj=%llu\n", measured_energy_mj);
        printf("RESULT measured_energy_j=%.6f\n", measured_energy_j);
        printf("RESULT average_power_w=%.6f\n", avg_power_w);
    }

    checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
    checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");

    checkNvml(nvmlShutdown(), "nvmlShutdown");

    checkCuda(cudaFree(A_gpu), "cudaFree A_gpu");
    checkCuda(cudaFree(B_gpu), "cudaFree B_gpu");
}

int main()
{
    int ni = NI;
    int nj = NJ;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    init_arrays(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    GPU_argv_init_measurement();

    convolution2DCudaKernelOnly(
        ni,
        nj,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(B_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
