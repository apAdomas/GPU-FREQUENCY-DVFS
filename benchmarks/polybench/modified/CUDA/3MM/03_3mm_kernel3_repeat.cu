/**
 * 03_3mm_kernel3_repeat.cu
 *
 * Isolated measurement for mm3_kernel3.
 *
 * Measures a single mm3_kernel3 launch
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

#include "3mm.cuh"
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
    int nm,
    DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
    DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
    DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
    DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nm; j++)
        {
            C[i][j] = ((DATA_TYPE)i * (j + 3)) / nl;
        }
    }

    for (i = 0; i < nm; i++)
    {
        for (j = 0; j < nl; j++)
        {
            D[i][j] = ((DATA_TYPE)i * (j + 2)) / nk;
        }
    }
}

static void compute_E_on_host(
    int ni,
    int nj,
    int nk,
    DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
    DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
    DATA_TYPE POLYBENCH_2D(E, NI, NJ, ni, nj))
{
    int i, j, k;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            E[i][j] = 0;
            for (k = 0; k < nk; ++k)
            {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

static void compute_F_on_host(
    int nj,
    int nl,
    int nm,
    DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
    DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl),
    DATA_TYPE POLYBENCH_2D(F, NJ, NL, nj, nl))
{
    int i, j, k;

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nl; j++)
        {
            F[i][j] = 0;
            for (k = 0; k < nm; ++k)
            {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }
}

__global__ void mm3_kernel3(
    int ni,
    int nj,
    int nk,
    int nl,
    int nm,
    DATA_TYPE *E,
    DATA_TYPE *F,
    DATA_TYPE *G)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < _PB_NI) && (j < _PB_NL))
    {
        G[i * NL + j] = 0;
        int k;
        for (k = 0; k < _PB_NJ; k++)
        {
            G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
        }
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* E_gpu,
    DATA_TYPE* F_gpu,
    DATA_TYPE* G_gpu,
    DATA_TYPE (*E)[NJ],
    DATA_TYPE (*F)[NL],
    DATA_TYPE (*G)[NL])
{
    checkCuda(
        cudaMemcpy(
            E_gpu,
            E,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy E");

    checkCuda(
        cudaMemcpy(
            F_gpu,
            F,
            sizeof(DATA_TYPE) * NJ * NL,
            cudaMemcpyHostToDevice),
        "copy F");

    checkCuda(
        cudaMemcpy(
            G_gpu,
            G,
            sizeof(DATA_TYPE) * NI * NL,
            cudaMemcpyHostToDevice),
        "copy G");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel3_once(
    int ni,
    int nj,
    int nk,
    int nl,
    int nm,
    DATA_TYPE* E_gpu,
    DATA_TYPE* F_gpu,
    DATA_TYPE* G_gpu,
    dim3 grid,
    dim3 block)
{
    mm3_kernel3<<<grid, block>>>(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu);
    checkCuda(cudaGetLastError(), "launch mm3_kernel3");
}

void mm3CudaKernel3Only(
    int ni,
    int nj,
    int nk,
    int nl,
    int nm,
    DATA_TYPE POLYBENCH_2D(E, NI, NJ, ni, nj),
    DATA_TYPE POLYBENCH_2D(F, NJ, NL, nj, nl),
    DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl),
    DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
    DATA_TYPE *E_gpu = NULL;
    DATA_TYPE *F_gpu = NULL;
    DATA_TYPE *G_gpu = NULL;

    checkCuda(cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ), "cudaMalloc E_gpu");
    checkCuda(cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL), "cudaMalloc F_gpu");
    checkCuda(cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL), "cudaMalloc G_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X)),
        (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(E_gpu, F_gpu, G_gpu, E, F, G);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel3_once(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(E_gpu, F_gpu, G_gpu, E, F, G);

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
            launch_kernel3_once(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu, grid, block);
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
            G_outputFromGpu,
            G_gpu,
            sizeof(DATA_TYPE) * NI * NL,
            cudaMemcpyDeviceToHost),
        "copy G_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=mm3_kernel3\n");
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

    checkCuda(cudaFree(E_gpu), "cudaFree E_gpu");
    checkCuda(cudaFree(F_gpu), "cudaFree F_gpu");
    checkCuda(cudaFree(G_gpu), "cudaFree G_gpu");
}

int main()
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL;
    int nm = NM;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
    POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
    POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
    POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

    init_array(
        ni,
        nj,
        nk,
        nl,
        nm,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(D));

    compute_E_on_host(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(E));
    compute_F_on_host(nj, nl, nm, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(F));

    memset(POLYBENCH_ARRAY(G), 0, sizeof(DATA_TYPE) * NI * NL);

    GPU_argv_init_measurement();

    mm3CudaKernel3Only(
        ni,
        nj,
        nk,
        nl,
        nm,
        POLYBENCH_ARRAY(E),
        POLYBENCH_ARRAY(F),
        POLYBENCH_ARRAY(G),
        POLYBENCH_ARRAY(G_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(E);
    POLYBENCH_FREE_ARRAY(F);
    POLYBENCH_FREE_ARRAY(G);
    POLYBENCH_FREE_ARRAY(G_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
