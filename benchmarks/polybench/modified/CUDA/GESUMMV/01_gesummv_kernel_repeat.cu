/**
 * 01_gesummv_kernel_repeat.cu
 *
 * Isolated measurement for gesummv_kernel.
 *
 * Measures a single gesummv_kernel launch
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

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#define POLYBENCH_TIME 1

#include "gesummv.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

void init(
    int n,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
    DATA_TYPE POLYBENCH_1D(x,N,n),
    DATA_TYPE POLYBENCH_1D(tmp,N,n),
    DATA_TYPE POLYBENCH_1D(y,N,n))
{
    int i, j;

    *alpha = 43532;
    *beta = 12313;

    for (i = 0; i < n; i++)
    {
        x[i] = ((DATA_TYPE)i) / N;
        tmp[i] = 0;
        y[i] = 0;

        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
            B[i][j] = ((DATA_TYPE)i * j) / n;
        }
    }
}

__global__ void gesummv_kernel(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE* A,
    DATA_TYPE* B,
    DATA_TYPE* tmp,
    DATA_TYPE* x,
    DATA_TYPE* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for (j = 0; j < _PB_N; j++)
        {
            tmp[i] += A[i * N + j] * x[j];
            y[i] += B[i * N + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    DATA_TYPE* tmp_gpu,
    DATA_TYPE* x_gpu,
    DATA_TYPE* y_gpu,
    DATA_TYPE (*A)[N],
    DATA_TYPE (*B)[N],
    DATA_TYPE* tmp,
    DATA_TYPE* x,
    DATA_TYPE* y)
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
            B_gpu,
            B,
            sizeof(DATA_TYPE) * N * N,
            cudaMemcpyHostToDevice),
        "copy B");

    checkCuda(
        cudaMemcpy(
            tmp_gpu,
            tmp,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy tmp");

    checkCuda(
        cudaMemcpy(
            x_gpu,
            x,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy x");

    checkCuda(
        cudaMemcpy(
            y_gpu,
            y,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyHostToDevice),
        "copy y");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel_once(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    DATA_TYPE* tmp_gpu,
    DATA_TYPE* x_gpu,
    DATA_TYPE* y_gpu,
    dim3 grid,
    dim3 block)
{
    gesummv_kernel<<<grid, block>>>(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu);
    checkCuda(cudaGetLastError(), "launch gesummv_kernel");
}

void gesummvCudaKernelOnly(
    int n,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
    DATA_TYPE POLYBENCH_1D(tmp,N,n),
    DATA_TYPE POLYBENCH_1D(x,N,n),
    DATA_TYPE POLYBENCH_1D(y,N,n),
    DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *B_gpu = NULL;
    DATA_TYPE *tmp_gpu = NULL;
    DATA_TYPE *x_gpu = NULL;
    DATA_TYPE *y_gpu = NULL;

    checkCuda(cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void**)&B_gpu, sizeof(DATA_TYPE) * N * N), "cudaMalloc B_gpu");
    checkCuda(cudaMalloc((void**)&tmp_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc tmp_gpu");
    checkCuda(cudaMalloc((void**)&x_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc x_gpu");
    checkCuda(cudaMalloc((void**)&y_gpu, sizeof(DATA_TYPE) * N), "cudaMalloc y_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x)), 1);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu, A, B, tmp, x, y);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel_once(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu, A, B, tmp, x, y);

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
            launch_kernel_once(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu, grid, block);
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
            y_outputFromGpu,
            y_gpu,
            sizeof(DATA_TYPE) * N,
            cudaMemcpyDeviceToHost),
        "copy y_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=gesummv_kernel\n");
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
    checkCuda(cudaFree(tmp_gpu), "cudaFree tmp_gpu");
    checkCuda(cudaFree(x_gpu), "cudaFree x_gpu");
    checkCuda(cudaFree(y_gpu), "cudaFree y_gpu");
}

int main()
{
    int n = N;

    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu, DATA_TYPE, N, n);

    init(
        n,
        &alpha,
        &beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(x),
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(y));

    GPU_argv_init_measurement();

    gesummvCudaKernelOnly(
        n,
        alpha,
        beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(x),
        POLYBENCH_ARRAY(y),
        POLYBENCH_ARRAY(y_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(y_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
