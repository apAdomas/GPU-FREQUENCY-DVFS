/**
 * 01_covariance_mean_kernel_repeat.cu
 *
 * Isolated measurement for mean_kernel.
 *
 * Measures a single mean_kernel launch
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
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#define POLYBENCH_TIME 1

#include "covariance.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#define FLOAT_N 3214212.01

void init_arrays(
    int m,
    int n,
    DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
    int i, j;

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            data[i][j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_M)
    {
        mean[j] = 0.0;

        int i;
        for (i = 0; i < _PB_N; i++)
        {
            mean[j] += data[i * M + j];
        }
        mean[j] /= (DATA_TYPE)FLOAT_N;
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* data_gpu,
    DATA_TYPE* mean_gpu,
    DATA_TYPE (*data)[N],
    DATA_TYPE* mean)
{
    checkCuda(
        cudaMemcpy(
            data_gpu,
            data,
            sizeof(DATA_TYPE) * M * N,
            cudaMemcpyHostToDevice),
        "copy data");

    checkCuda(
        cudaMemcpy(
            mean_gpu,
            mean,
            sizeof(DATA_TYPE) * M,
            cudaMemcpyHostToDevice),
        "copy mean");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_mean_once(
    int m,
    int n,
    DATA_TYPE* mean_gpu,
    DATA_TYPE* data_gpu,
    dim3 grid,
    dim3 block)
{
    mean_kernel<<<grid, block>>>(m, n, mean_gpu, data_gpu);
    checkCuda(cudaGetLastError(), "launch mean_kernel");
}

void covarianceCudaMeanKernelOnly(
    int m,
    int n,
    DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
    DATA_TYPE POLYBENCH_1D(mean, M, m),
    DATA_TYPE POLYBENCH_1D(mean_outputFromGpu, M, m))
{
    DATA_TYPE* data_gpu = NULL;
    DATA_TYPE* mean_gpu = NULL;

    checkCuda(cudaMalloc((void**)&data_gpu, sizeof(DATA_TYPE) * M * N), "cudaMalloc data_gpu");
    checkCuda(cudaMalloc((void**)&mean_gpu, sizeof(DATA_TYPE) * M), "cudaMalloc mean_gpu");

    dim3 block(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
    dim3 grid(
        (size_t)ceil(((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)),
        1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(data_gpu, mean_gpu, data, mean);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_mean_once(m, n, mean_gpu, data_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(data_gpu, mean_gpu, data, mean);

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
            launch_mean_once(m, n, mean_gpu, data_gpu, grid, block);
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
            mean_outputFromGpu,
            mean_gpu,
            sizeof(DATA_TYPE) * M,
            cudaMemcpyDeviceToHost),
        "copy mean_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=mean_kernel\n");
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

    checkCuda(cudaFree(data_gpu), "cudaFree data_gpu");
    checkCuda(cudaFree(mean_gpu), "cudaFree mean_gpu");
}

int main()
{
    int m = M;
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
    POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
    POLYBENCH_1D_ARRAY_DECL(mean_outputFromGpu, DATA_TYPE, M, m);

    init_arrays(m, n, POLYBENCH_ARRAY(data));

    memset(POLYBENCH_ARRAY(mean), 0, sizeof(DATA_TYPE) * M);

    GPU_argv_init_measurement();

    covarianceCudaMeanKernelOnly(
        m,
        n,
        POLYBENCH_ARRAY(data),
        POLYBENCH_ARRAY(mean),
        POLYBENCH_ARRAY(mean_outputFromGpu));

    POLYBENCH_FREE_ARRAY(data);
    POLYBENCH_FREE_ARRAY(mean);
    POLYBENCH_FREE_ARRAY(mean_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
