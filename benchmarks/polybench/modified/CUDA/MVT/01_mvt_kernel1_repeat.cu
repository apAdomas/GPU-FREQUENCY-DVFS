/**
 * 01_mvt_kernel1_repeat.cu
 *
 * Isolated measurement for mvt_kernel1.
 *
 * Measures a single mvt_kernel1 launch
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

#include "mvt.cuh"
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
    DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
    DATA_TYPE POLYBENCH_1D(x1, N, n),
    DATA_TYPE POLYBENCH_1D(x2, N, n),
    DATA_TYPE POLYBENCH_1D(y1, N, n),
    DATA_TYPE POLYBENCH_1D(y2, N, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        x1[i] = ((DATA_TYPE)i) / N;
        x2[i] = ((DATA_TYPE)i + 1) / N;
        y1[i] = ((DATA_TYPE)i + 3) / N;
        y2[i] = ((DATA_TYPE)i + 4) / N;

        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
        }
    }
}

__global__ void mvt_kernel1(int n, DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for (j = 0; j < _PB_N; j++)
        {
            x1[i] += a[i * N + j] * y_1[j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* a_gpu,
    DATA_TYPE* x1_gpu,
    DATA_TYPE* y1_gpu,
    DATA_TYPE (*a)[N],
    DATA_TYPE* x1,
    DATA_TYPE* y1)
{
    checkCuda(
        cudaMemcpy(
            a_gpu,
            a,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyHostToDevice),
        "copy a");

    checkCuda(
        cudaMemcpy(
            x1_gpu,
            x1,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy x1");

    checkCuda(
        cudaMemcpy(
            y1_gpu,
            y1,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy y1");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel1_once(
    int n,
    DATA_TYPE* a_gpu,
    DATA_TYPE* x1_gpu,
    DATA_TYPE* y1_gpu,
    dim3 grid,
    dim3 block)
{
    mvt_kernel1<<<grid, block>>>(n, a_gpu, x1_gpu, y1_gpu);
    checkCuda(cudaGetLastError(), "launch kernel1");
}

void runMvtCudaKernel1Only(
    int n,
    DATA_TYPE POLYBENCH_2D(a, N, N, n, n),
    DATA_TYPE POLYBENCH_1D(x1, N, n),
    DATA_TYPE POLYBENCH_1D(y1, N, n),
    DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n))
{
    DATA_TYPE* a_gpu = NULL;
    DATA_TYPE* x1_gpu = NULL;
    DATA_TYPE* y1_gpu = NULL;

    checkCuda(cudaMalloc((void**)&a_gpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc a_gpu");
    checkCuda(cudaMalloc((void**)&x1_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc x1_gpu");
    checkCuda(cudaMalloc((void**)&y1_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc y1_gpu");

    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((size_t)ceil((float)N / ((float)block.x)), 1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(a_gpu, x1_gpu, y1_gpu, a, x1, y1);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel1_once(n, a_gpu, x1_gpu, y1_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(a_gpu, x1_gpu, y1_gpu, a, x1, y1);

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
            launch_kernel1_once(n, a_gpu, x1_gpu, y1_gpu, grid, block);
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
            x1_outputFromGpu,
            x1_gpu,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyDeviceToHost),
        "copy x1_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=mvt_kernel1\n");
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

    checkCuda(cudaFree(a_gpu), "cudaFree a_gpu");
    checkCuda(cudaFree(x1_gpu), "cudaFree x1_gpu");
    checkCuda(cudaFree(y1_gpu), "cudaFree y1_gpu");
}

int main()
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu, DATA_TYPE, N, n);

    init_array(
        n,
        POLYBENCH_ARRAY(a),
        POLYBENCH_ARRAY(x1),
        POLYBENCH_ARRAY(x2),
        POLYBENCH_ARRAY(y1),
        POLYBENCH_ARRAY(y2));

    GPU_argv_init_measurement();

    runMvtCudaKernel1Only(
        n,
        POLYBENCH_ARRAY(a),
        POLYBENCH_ARRAY(x1),
        POLYBENCH_ARRAY(y1),
        POLYBENCH_ARRAY(x1_outputFromGpu));

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(x1);
    POLYBENCH_FREE_ARRAY(x2);
    POLYBENCH_FREE_ARRAY(y1);
    POLYBENCH_FREE_ARRAY(y2);
    POLYBENCH_FREE_ARRAY(x1_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
