/**
 * 03_gemver_kernel3_repeat.cu
 *
 * Isolated measurement for gemver_kernel3.
 *
 * Measures a single gemver_kernel3 launch
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

#include "gemver.cuh"
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
    int n,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
    DATA_TYPE POLYBENCH_1D(u1, N, n),
    DATA_TYPE POLYBENCH_1D(v1, N, n),
    DATA_TYPE POLYBENCH_1D(u2, N, n),
    DATA_TYPE POLYBENCH_1D(v2, N, n),
    DATA_TYPE POLYBENCH_1D(w, N, n),
    DATA_TYPE POLYBENCH_1D(x, N, n),
    DATA_TYPE POLYBENCH_1D(y, N, n),
    DATA_TYPE POLYBENCH_1D(z, N, n))
{
    int i, j;

    *alpha = 43532;
    *beta = 12313;

    for (i = 0; i < N; i++)
    {
        u1[i] = i;
        u2[i] = (i + 1) / N / 2.0;
        v1[i] = (i + 1) / N / 4.0;
        v2[i] = (i + 1) / N / 6.0;
        y[i] = (i + 1) / N / 8.0;
        z[i] = (i + 1) / N / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;

        for (j = 0; j < N; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
        }
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            x[i] = x[i] + (*beta) * A[j][i] * y[j];
        }
        x[i] = x[i] + z[i];
    }
}

__global__ void gemver_kernel3(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *a,
    DATA_TYPE *x,
    DATA_TYPE *w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= 0) && (i < _PB_N))
    {
        int j;
        for (j = 0; j < _PB_N; j++)
        {
            w[i] += alpha * a[i * N + j] * x[j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE *A_gpu,
    DATA_TYPE *x_gpu,
    DATA_TYPE *w_gpu,
    DATA_TYPE (*A)[N],
    DATA_TYPE *x,
    DATA_TYPE *w)
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            x_gpu,
            x,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy x");

    checkCuda(
        cudaMemcpy(
            w_gpu,
            w,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy w");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel3_once(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *A_gpu,
    DATA_TYPE *x_gpu,
    DATA_TYPE *w_gpu,
    dim3 grid,
    dim3 block)
{
    gemver_kernel3<<<grid, block>>>(n, alpha, beta, A_gpu, x_gpu, w_gpu);
    checkCuda(cudaGetLastError(), "launch gemver_kernel3");
}

void gemverCudaKernel3Only(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
    DATA_TYPE POLYBENCH_1D(x, N, n),
    DATA_TYPE POLYBENCH_1D(w, N, n),
    DATA_TYPE POLYBENCH_1D(w_outputFromGpu, N, n))
{
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *x_gpu = NULL;
    DATA_TYPE *w_gpu = NULL;

    checkCuda(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc x_gpu");
    checkCuda(cudaMalloc((void **)&w_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc w_gpu");

    dim3 block(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
    dim3 grid(
        (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)),
        1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, x_gpu, w_gpu, A, x, w);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel3_once(n, alpha, beta, A_gpu, x_gpu, w_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, x_gpu, w_gpu, A, x, w);

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
            launch_kernel3_once(n, alpha, beta, A_gpu, x_gpu, w_gpu, grid, block);
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
            w_outputFromGpu,
            w_gpu,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyDeviceToHost),
        "copy w_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=gemver_kernel3\n");
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
    checkCuda(cudaFree(x_gpu), "cudaFree x_gpu");
    checkCuda(cudaFree(w_gpu), "cudaFree w_gpu");
}

int main()
{
    int n = N;

    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu, DATA_TYPE, N, n);

    init_arrays(
        n,
        &alpha,
        &beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(u1),
        POLYBENCH_ARRAY(v1),
        POLYBENCH_ARRAY(u2),
        POLYBENCH_ARRAY(v2),
        POLYBENCH_ARRAY(w),
        POLYBENCH_ARRAY(x),
        POLYBENCH_ARRAY(y),
        POLYBENCH_ARRAY(z));

    GPU_argv_init_measurement();

    gemverCudaKernel3Only(
        n,
        alpha,
        beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(x),
        POLYBENCH_ARRAY(w),
        POLYBENCH_ARRAY(w_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(u1);
    POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2);
    POLYBENCH_FREE_ARRAY(v2);
    POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(z);
    POLYBENCH_FREE_ARRAY(w_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
