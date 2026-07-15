/**
 * 01_gemm_kernel_repeat.cu
 *
 * Isolated measurement for gemm_kernel.
 *
 * Measures a single gemm_kernel launch
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

#include "gemm.cuh"
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
    int ni,
    int nj,
    int nk,
    DATA_TYPE* alpha,
    DATA_TYPE* beta,
    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
    DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j;

    *alpha = 32412;
    *beta = 2123;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }
}

__global__ void gemm_kernel(
    int ni,
    int nj,
    int nk,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *a,
    DATA_TYPE *b,
    DATA_TYPE *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NJ))
    {
        c[i * NJ + j] *= beta;

        int k;
        for (k = 0; k < _PB_NK; k++)
        {
            c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ + j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    DATA_TYPE* C_gpu,
    DATA_TYPE (*A)[NK],
    DATA_TYPE (*B)[NJ],
    DATA_TYPE (*C)[NJ])
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * NI * NK,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            B_gpu,
            B,
            sizeof(DATA_TYPE) * NK * NJ,
            cudaMemcpyHostToDevice),
        "copy B");

    checkCuda(
        cudaMemcpy(
            C_gpu,
            C,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy C");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel_once(
    int ni,
    int nj,
    int nk,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE* A_gpu,
    DATA_TYPE* B_gpu,
    DATA_TYPE* C_gpu,
    dim3 grid,
    dim3 block)
{
    gemm_kernel<<<grid, block>>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
    checkCuda(cudaGetLastError(), "launch gemm_kernel");
}

void gemmCudaKernelOnly(
    int ni,
    int nj,
    int nk,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
    DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    DATA_TYPE *A_gpu = NULL;
    DATA_TYPE *B_gpu = NULL;
    DATA_TYPE *C_gpu = NULL;

    checkCuda(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ), "cudaMalloc B_gpu");
    checkCuda(cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ), "cudaMalloc C_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NJ) / ((float)block.x)),
        (size_t)ceil(((float)NI) / ((float)block.y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, B_gpu, C_gpu, A, B, C);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel_once(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, B_gpu, C_gpu, A, B, C);

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
            launch_kernel_once(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu, grid, block);
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
            C_outputFromGpu,
            C_gpu,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyDeviceToHost),
        "copy C_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=gemm_kernel\n");
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
    checkCuda(cudaFree(C_gpu), "cudaFree C_gpu");
}

int main()
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init_measurement();

    gemmCudaKernelOnly(
        ni,
        nj,
        nk,
        alpha,
        beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(C_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
