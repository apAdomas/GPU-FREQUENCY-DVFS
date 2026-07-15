/**
 * 01_jacobi2d_kernel1_repeat.cu
 *
 * Isolated measurement for kernel1.
 *
 * Measures a single runJacobiCUDA_kernel1 launch
 * repeatedly for energy/time measurement.
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

#include "jacobi2D.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

void init_array(
    int n,
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * (j + 2) + 10) / N;
            B[i][j] = ((DATA_TYPE)(i - 4) * (j - 1) + 11) / N;
        }
    }
}

__global__ void runJacobiCUDA_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 1) && (i < (_PB_N - 1)) && (j >= 1) && (j < (_PB_N - 1)))
    {
        B[i * N + j] =
            0.2f * (A[i * N + j]
                  + A[i * N + (j - 1)]
                  + A[i * N + (j + 1)]
                  + A[(i + 1) * N + j]
                  + A[(i - 1) * N + j]);
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* Agpu,
    DATA_TYPE* Bgpu,
    DATA_TYPE (*A)[N],
    DATA_TYPE (*B)[N])
{
    checkCuda(
        cudaMemcpy(
            Agpu,
            A,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            Bgpu,
            B,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyHostToDevice),
        "copy B");
    // input copy fully finished before measurement
    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel1_once(
    int n,
    DATA_TYPE* Agpu,
    DATA_TYPE* Bgpu,
    dim3 grid,
    dim3 block)
{
    runJacobiCUDA_kernel1<<<grid, block>>>(n, Agpu, Bgpu);
    checkCuda(cudaGetLastError(), "launch kernel1");
}

void runJacobi2DCUDAKernel1Only(
    int n,
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n))
{
    DATA_TYPE* Agpu = NULL;
    DATA_TYPE* Bgpu = NULL;

    checkCuda(cudaMalloc((void**)&Agpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc Agpu");
    checkCuda(cudaMalloc((void**)&Bgpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc Bgpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (unsigned int)ceil(((float)N) / ((float)block.x)),
        (unsigned int)ceil(((float)N) / ((float)block.y)));

    // inititalize nvml for energy
    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    // copy inputs to device for warmup
    copy_inputs_to_device(Agpu, Bgpu, A, B);

    // warmup
    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel1_once(n, Agpu, Bgpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    // copy inputs to device 2
    copy_inputs_to_device(Agpu, Bgpu, A, B);

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
            launch_kernel1_once(n, Agpu, Bgpu, grid, block);
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
            Bgpu,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyDeviceToHost),
        "copy B_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=jacobi2d_kernel1\n");
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

    checkCuda(cudaFree(Agpu), "cudaFree Agpu");
    checkCuda(cudaFree(Bgpu), "cudaFree Bgpu");
}

int main()
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, N, N, n, n);

    init_array(
        n,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B));

    GPU_argv_init_measurement();

    runJacobi2DCUDAKernel1Only(
        n,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(B_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
