/**
 * 02_2mm_kernel2_repeat.cu
 *
 * Isolated measurement for mm2_kernel2.
 *
 * Measures a single mm2_kernel2 launch
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

#include "2mm.cuh"
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
    int ni,
    int nj,
    int nk,
    int nl,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
    DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
    DATA_TYPE POLYBENCH_2D(tmp, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj),
    DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
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
            B[i][j] = ((DATA_TYPE)i * (j + 1)) / NJ;
        }
    }

    for (i = 0; i < nl; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE)i * (j + 3)) / NL;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE)i * (j + 2)) / NK;
        }
    }

    int k;
    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            tmp[i][j] = 0;
            for (k = 0; k < nk; k++)
            {
                tmp[i][j] += (*alpha) * A[i][k] * B[k][j];
            }
        }
    }
}

__global__ void mm2_kernel2(
    int ni,
    int nj,
    int nk,
    int nl,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *tmp,
    DATA_TYPE *C,
    DATA_TYPE *D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NL))
    {
        D[i * NL + j] *= beta;
        int k;
        for (k = 0; k < _PB_NJ; k++)
        {
            D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE *tmp_gpu,
    DATA_TYPE *C_gpu,
    DATA_TYPE *D_gpu,
    DATA_TYPE (*tmp)[NJ],
    DATA_TYPE (*C)[NJ],
    DATA_TYPE (*D)[NL])
{
    checkCuda(
        cudaMemcpy(
            tmp_gpu,
            tmp,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy tmp");

    checkCuda(
        cudaMemcpy(
            C_gpu,
            C,
            sizeof(DATA_TYPE) * NL * NJ,
            cudaMemcpyHostToDevice),
        "copy C");

    checkCuda(
        cudaMemcpy(
            D_gpu,
            D,
            sizeof(DATA_TYPE) * NI * NL,
            cudaMemcpyHostToDevice),
        "copy D");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel2_once(
    int ni,
    int nj,
    int nk,
    int nl,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *tmp_gpu,
    DATA_TYPE *C_gpu,
    DATA_TYPE *D_gpu,
    dim3 grid,
    dim3 block)
{
    mm2_kernel2<<<grid, block>>>(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu);
    checkCuda(cudaGetLastError(), "launch mm2_kernel2");
}

void mm2CudaKernel2Only(
    int ni,
    int nj,
    int nk,
    int nl,
    DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE POLYBENCH_2D(tmp, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj),
    DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl),
    DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
    DATA_TYPE *tmp_gpu = NULL;
    DATA_TYPE *C_gpu = NULL;
    DATA_TYPE *D_gpu = NULL;

    checkCuda(cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NI * NJ), "cudaMalloc tmp_gpu");
    checkCuda(cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NL * NJ), "cudaMalloc C_gpu");
    checkCuda(cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NI * NL), "cudaMalloc D_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NL) / ((float)block.x)),
        (size_t)ceil(((float)NI) / ((float)block.y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(tmp_gpu, C_gpu, D_gpu, tmp, C, D);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel2_once(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(tmp_gpu, C_gpu, D_gpu, tmp, C, D);

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
            launch_kernel2_once(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu, D_gpu, grid, block);
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
            D_outputFromGpu,
            D_gpu,
            sizeof(DATA_TYPE) * NI * NL,
            cudaMemcpyDeviceToHost),
        "copy D_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=mm2_kernel2\n");
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

    checkCuda(cudaFree(tmp_gpu), "cudaFree tmp_gpu");
    checkCuda(cudaFree(C_gpu), "cudaFree C_gpu");
    checkCuda(cudaFree(D_gpu), "cudaFree D_gpu");
}

int main()
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;

    DATA_TYPE alpha;
    DATA_TYPE beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NL, NJ, nl, nj);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);
    POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

    init_array(
        ni,
        nj,
        nk,
        nl,
        &alpha,
        &beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(D));

    GPU_argv_init_measurement();

    mm2CudaKernel2Only(
        ni,
        nj,
        nk,
        nl,
        alpha,
        beta,
        POLYBENCH_ARRAY(tmp),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(D),
        POLYBENCH_ARRAY(D_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(D_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
