/**
 * 04_correlation_corr_kernel_repeat.cu
 *
 * Isolated measurement for corr_kernel.
 *
 * Measures a single corr_kernel launch
 * repeated many times for energy/time measurement.
 *
 * Protocol:
 * 1) allocate once
 * 2) prepare/reset inputs
 * 3) copy/reset inputs to device
 * 4) warm up with repeated single launches
 * 5) reset inputs
 * 6) measure repeated single launches
 *    - CUDA events for runtime
 *    - NVML total energy for energy
 * 7) copy back once
 * 8) exit
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

#include "correlation.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#define FLOAT_N 3214212.01f
#define EPS 0.005f

void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            data[i][j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

static void compute_mean_on_host(
    int m,
    int n,
    DATA_TYPE (*data)[N],
    DATA_TYPE* mean)
{
    int i, j;

    for (j = 0; j < m; j++)
    {
        mean[j] = 0.0f;
        for (i = 0; i < n; i++)
        {
            mean[j] += data[i][j];
        }
        mean[j] /= (DATA_TYPE)FLOAT_N;
    }
}

static void compute_std_on_host(
    int m,
    int n,
    DATA_TYPE (*data)[N],
    DATA_TYPE* mean,
    DATA_TYPE* stddev)
{
    int i, j;

    for (j = 0; j < m; j++)
    {
        stddev[j] = 0.0f;

        for (i = 0; i < n; i++)
        {
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        }

        stddev[j] /= FLOAT_N;
        stddev[j] = sqrt(stddev[j]);
        if (stddev[j] <= EPS)
        {
            stddev[j] = 1.0f;
        }
    }
}

static void reduce_on_host(
    int m,
    int n,
    DATA_TYPE (*data)[N],
    DATA_TYPE* mean,
    DATA_TYPE* stddev)
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            data[i][j] -= mean[j];
            data[i][j] /= (sqrt(FLOAT_N) * stddev[j]);
        }
    }
}

static void prepare_corr_inputs_on_host(
    int m,
    int n,
    DATA_TYPE (*data)[N],
    DATA_TYPE (*data_normalized)[N],
    DATA_TYPE* mean,
    DATA_TYPE* stddev,
    DATA_TYPE (*symmat)[M])
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++)
        {
            data_normalized[i][j] = data[i][j];
        }
    }

    compute_mean_on_host(m, n, data_normalized, mean);
    compute_std_on_host(m, n, data_normalized, mean, stddev);
    reduce_on_host(m, n, data_normalized, mean, stddev);

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++)
        {
            symmat[i][j] = 0.0f;
        }
    }
}

__global__ void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data)
{
    int j1 = blockIdx.x * blockDim.x + threadIdx.x;

    int i, j2;
    if (j1 < (_PB_M - 1))
    {
        symmat[j1 * M + j1] = 1.0;

        for (j2 = (j1 + 1); j2 < _PB_M; j2++)
        {
            symmat[j1 * M + j2] = 0.0;

            for (i = 0; i < _PB_N; i++)
            {
                symmat[j1 * M + j2] += data[i * M + j1] * data[i * M + j2];
            }
            symmat[j2 * M + j1] = symmat[j1 * M + j2];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* data_gpu,
    DATA_TYPE* symmat_gpu,
    DATA_TYPE (*data_normalized)[N],
    DATA_TYPE (*symmat)[M])
{
    checkCuda(
        cudaMemcpy(
            data_gpu,
            data_normalized,
            sizeof(DATA_TYPE) * M * N,
            cudaMemcpyHostToDevice),
        "copy data_normalized");

    checkCuda(
        cudaMemcpy(
            symmat_gpu,
            symmat,
            sizeof(DATA_TYPE) * M * M,
            cudaMemcpyHostToDevice),
        "copy symmat");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_corr_once(
    int m,
    int n,
    DATA_TYPE* symmat_gpu,
    DATA_TYPE* data_gpu,
    dim3 grid,
    dim3 block)
{
    corr_kernel<<<grid, block>>>(m, n, symmat_gpu, data_gpu);
    checkCuda(cudaGetLastError(), "launch corr_kernel");
}

void correlationCudaCorrKernelOnly(
    int m,
    int n,
    DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
    DATA_TYPE POLYBENCH_1D(mean, M, m),
    DATA_TYPE POLYBENCH_1D(stddev, M, m),
    DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m),
    DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, M, m, m))
{
    DATA_TYPE* data_gpu = NULL;
    DATA_TYPE* symmat_gpu = NULL;

    checkCuda(cudaMalloc((void**)&data_gpu, sizeof(DATA_TYPE) * M * N), "cudaMalloc data_gpu");
    checkCuda(cudaMalloc((void**)&symmat_gpu, sizeof(DATA_TYPE) * M * M), "cudaMalloc symmat_gpu");

    dim3 block(DIM_THREAD_BLOCK_KERNEL_4_X, DIM_THREAD_BLOCK_KERNEL_4_Y);
    dim3 grid(
        (size_t)ceil(((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)),
        1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    POLYBENCH_2D_ARRAY_DECL(data_normalized, DATA_TYPE, M, N, m, n);

    prepare_corr_inputs_on_host(m, n, data, POLYBENCH_ARRAY(data_normalized), mean, stddev, symmat);

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(data_gpu, symmat_gpu, POLYBENCH_ARRAY(data_normalized), symmat);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_corr_once(m, n, symmat_gpu, data_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    prepare_corr_inputs_on_host(m, n, data, POLYBENCH_ARRAY(data_normalized), mean, stddev, symmat);
    copy_inputs_to_device(data_gpu, symmat_gpu, POLYBENCH_ARRAY(data_normalized), symmat);

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
            launch_corr_once(m, n, symmat_gpu, data_gpu, grid, block);
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

    DATA_TYPE value_last_diag = 1.0f;
    checkCuda(
        cudaMemcpy(
            &(symmat_gpu[(M - 1) * M + (M - 1)]),
            &value_last_diag,
            sizeof(DATA_TYPE),
            cudaMemcpyHostToDevice),
        "set last diagonal");

    checkCuda(
        cudaMemcpy(
            symmat_outputFromGpu,
            symmat_gpu,
            sizeof(DATA_TYPE) * M * M,
            cudaMemcpyDeviceToHost),
        "copy symmat_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=corr_kernel\n");
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
    checkCuda(cudaFree(symmat_gpu), "cudaFree symmat_gpu");

    POLYBENCH_FREE_ARRAY(data_normalized);
}

int main()
{
    int m = M;
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
    POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
    POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);
    POLYBENCH_2D_ARRAY_DECL(symmat, DATA_TYPE, M, M, m, m);
    POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu, DATA_TYPE, M, M, m, m);

    init_arrays(m, n, POLYBENCH_ARRAY(data));

    GPU_argv_init_measurement();

    correlationCudaCorrKernelOnly(
        m,
        n,
        POLYBENCH_ARRAY(data),
        POLYBENCH_ARRAY(mean),
        POLYBENCH_ARRAY(stddev),
        POLYBENCH_ARRAY(symmat),
        POLYBENCH_ARRAY(symmat_outputFromGpu));

    POLYBENCH_FREE_ARRAY(data);
    POLYBENCH_FREE_ARRAY(mean);
    POLYBENCH_FREE_ARRAY(stddev);
    POLYBENCH_FREE_ARRAY(symmat);
    POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
