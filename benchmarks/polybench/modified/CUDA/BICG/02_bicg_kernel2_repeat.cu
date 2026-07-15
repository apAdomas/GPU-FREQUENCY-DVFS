/**
 * 02_bicg_kernel2_repeat.cu
 *
 * Isolated measurement for bicg_kernel2.
 *
 * Measures a single bicg_kernel2 launch
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
    DATA_TYPE POLYBENCH_1D(p, NY, ny),
    DATA_TYPE POLYBENCH_1D(q, NX, nx))
{
    int i, j;

    for (i = 0; i < ny; i++)
    {
        p[i] = i * M_PI;
    }

    for (i = 0; i < nx; i++)
    {
        q[i] = 0.0f;

        for (j = 0; j < ny; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NX;
        }
    }
}

__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_NX)
    {
        q[i] = 0.0f;

        int j;
        for (j = 0; j < _PB_NY; j++)
        {
            q[i] += A[i * NY + j] * p[j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* p_gpu,
    DATA_TYPE* q_gpu,
    DATA_TYPE (*A)[NY],
    DATA_TYPE* p,
    DATA_TYPE* q)
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
            p_gpu,
            p,
            sizeof(DATA_TYPE) * NY,
            cudaMemcpyHostToDevice),
        "copy p");

    checkCuda(
        cudaMemcpy(
            q_gpu,
            q,
            sizeof(DATA_TYPE) * NX,
            cudaMemcpyHostToDevice),
        "copy q");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel2_once(
    int nx,
    int ny,
    DATA_TYPE* A_gpu,
    DATA_TYPE* p_gpu,
    DATA_TYPE* q_gpu,
    dim3 grid,
    dim3 block)
{
    bicg_kernel2<<<grid, block>>>(nx, ny, A_gpu, p_gpu, q_gpu);
    checkCuda(cudaGetLastError(), "launch bicg_kernel2");
}

void bicgCudaKernel2Only(
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_1D(p, NY, ny),
    DATA_TYPE POLYBENCH_1D(q, NX, nx),
    DATA_TYPE POLYBENCH_1D(q_outputFromGpu, NX, nx))
{
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *p_gpu = NULL;
    DATA_TYPE *q_gpu = NULL;

    checkCuda(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void **)&p_gpu, sizeof(DATA_TYPE) * NY), "cudaMalloc p_gpu");
    checkCuda(cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * NX), "cudaMalloc q_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NX) / ((float)block.x)),
        1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, p_gpu, q_gpu, A, p, q);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel2_once(nx, ny, A_gpu, p_gpu, q_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, p_gpu, q_gpu, A, p, q);

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
            launch_kernel2_once(nx, ny, A_gpu, p_gpu, q_gpu, grid, block);
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
            q_outputFromGpu,
            q_gpu,
            sizeof(DATA_TYPE) * NX,
            cudaMemcpyDeviceToHost),
        "copy q_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=bicg_kernel2\n");
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
    checkCuda(cudaFree(p_gpu), "cudaFree p_gpu");
    checkCuda(cudaFree(q_gpu), "cudaFree q_gpu");
}

int main()
{
    int nx = NX;
    int ny = NY;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, NX, nx);
    POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu, DATA_TYPE, NX, nx);

    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

    GPU_argv_init_measurement();

    bicgCudaKernel2Only(
        nx,
        ny,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(p),
        POLYBENCH_ARRAY(q),
        POLYBENCH_ARRAY(q_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(p);
    POLYBENCH_FREE_ARRAY(q);
    POLYBENCH_FREE_ARRAY(q_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
