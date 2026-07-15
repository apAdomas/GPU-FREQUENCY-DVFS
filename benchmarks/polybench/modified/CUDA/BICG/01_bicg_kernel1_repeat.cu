/**
 * 01_bicg_kernel1_repeat.cu
 *
 * Isolated measurement for bicg_kernel1.
 *
 * Measures a single bicg_kernel1 launch
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
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#define POLYBENCH_TIME 1

#include "bicg.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_1D(r, NX, nx),
    DATA_TYPE POLYBENCH_1D(s, NY, ny))
{
    int i, j;

    for (i = 0; i < ny; i++)
    {
        s[i] = 0.0f;
    }

    for (i = 0; i < nx; i++)
    {
        r[i] = i * M_PI;

        for (j = 0; j < ny; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NX;
        }
    }
}

__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_NY)
    {
        s[j] = 0.0f;

        int i;
        for (i = 0; i < _PB_NX; i++)
        {
            s[j] += r[i] * A[i * NY + j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* r_gpu,
    DATA_TYPE* s_gpu,
    DATA_TYPE (*A)[NY],
    DATA_TYPE* r,
    DATA_TYPE* s)
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * NX * NY,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            r_gpu,
            r,
            sizeof(DATA_TYPE) * NX,
            cudaMemcpyHostToDevice),
        "copy r");

    checkCuda(
        cudaMemcpy(
            s_gpu,
            s,
            sizeof(DATA_TYPE) * NY,
            cudaMemcpyHostToDevice),
        "copy s");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel1_once(
    int nx,
    int ny,
    DATA_TYPE* A_gpu,
    DATA_TYPE* r_gpu,
    DATA_TYPE* s_gpu,
    dim3 grid,
    dim3 block)
{
    bicg_kernel1<<<grid, block>>>(nx, ny, A_gpu, r_gpu, s_gpu);
    checkCuda(cudaGetLastError(), "launch bicg_kernel1");
}

void bicgCudaKernel1Only(
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_1D(r, NX, nx),
    DATA_TYPE POLYBENCH_1D(s, NY, ny),
    DATA_TYPE POLYBENCH_1D(s_outputFromGpu, NY, ny))
{
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *r_gpu = NULL;
    DATA_TYPE *s_gpu = NULL;

    checkCuda(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * NX), "cudaMalloc r_gpu");
    checkCuda(cudaMalloc((void **)&s_gpu, sizeof(DATA_TYPE) * NY), "cudaMalloc s_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NY) / ((float)block.x)),
        1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, r_gpu, s_gpu, A, r, s);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel1_once(nx, ny, A_gpu, r_gpu, s_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, r_gpu, s_gpu, A, r, s);

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
            launch_kernel1_once(nx, ny, A_gpu, r_gpu, s_gpu, grid, block);
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
            s_outputFromGpu,
            s_gpu,
            sizeof(DATA_TYPE) * NY,
            cudaMemcpyDeviceToHost),
        "copy s_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=bicg_kernel1\n");
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
    checkCuda(cudaFree(r_gpu), "cudaFree r_gpu");
    checkCuda(cudaFree(s_gpu), "cudaFree s_gpu");
}

int main()
{
    int nx = NX;
    int ny = NY;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, NX, nx);
    POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu, DATA_TYPE, NY, ny);

    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s));

    GPU_argv_init_measurement();

    bicgCudaKernel1Only(
        nx,
        ny,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(r),
        POLYBENCH_ARRAY(s),
        POLYBENCH_ARRAY(s_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(r);
    POLYBENCH_FREE_ARRAY(s);
    POLYBENCH_FREE_ARRAY(s_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
