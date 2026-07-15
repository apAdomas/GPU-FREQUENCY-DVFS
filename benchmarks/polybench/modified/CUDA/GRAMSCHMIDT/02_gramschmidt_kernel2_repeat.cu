/**
 * 02_gramschmidt_kernel2_repeat.cu
 *
 * Isolated measurement for gramschmidt_kernel2.
 *
 * Measures a single gramschmidt_kernel2 launch (at fixed k = FIXED_K)
 * repeated many times for energy/time measurement.
 *
 * Protocol:
 * 1) allocate once
 * 2) copy/reset inputs
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

#include "gramschmidt.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

#ifndef FIXED_K
#define FIXED_K (NJ / 2)
#endif

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

void init_array(
    int ni,
    int nj,
    DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
    DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
    int i, j;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / ni;
            Q[i][j] = ((DATA_TYPE)i * (j + 1)) / nj;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nj; j++)
        {
            R[i][j] = ((DATA_TYPE)i * (j + 2)) / nj;
        }
    }
}

__global__ void gramschmidt_kernel2(
    int ni,
    int nj,
    DATA_TYPE *a,
    DATA_TYPE *r,
    DATA_TYPE *q,
    int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_NI)
    {
        q[i * NJ + k] =
            a[i * NJ + k] / r[k * NJ + k];
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* R_gpu,
    DATA_TYPE* Q_gpu,
    DATA_TYPE (*A)[NJ],
    DATA_TYPE (*R)[NJ],
    DATA_TYPE (*Q)[NJ])
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            R_gpu,
            R,
            sizeof(DATA_TYPE) * NJ * NJ,
            cudaMemcpyHostToDevice),
        "copy R");

    checkCuda(
        cudaMemcpy(
            Q_gpu,
            Q,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyHostToDevice),
        "copy Q");

    checkCuda(
        cudaDeviceSynchronize(),
        "sync after copy");
}

static void launch_kernel2_once(
    int ni,
    int nj,
    DATA_TYPE* A_gpu,
    DATA_TYPE* R_gpu,
    DATA_TYPE* Q_gpu,
    dim3 grid,
    dim3 block)
{
    gramschmidt_kernel2<<<grid, block>>>(
        ni,
        nj,
        A_gpu,
        R_gpu,
        Q_gpu,
        FIXED_K);

    checkCuda(cudaGetLastError(), "launch kernel2");
}

void gramschmidtCudaKernel2Only(
    int ni,
    int nj,
    DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
    DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(Q_outputFromGpu,NI,NJ,ni,nj))
{
    DATA_TYPE* A_gpu = NULL;
    DATA_TYPE* R_gpu = NULL;
    DATA_TYPE* Q_gpu = NULL;

    checkCuda(
        cudaMalloc((void**)&A_gpu,
                   sizeof(DATA_TYPE) * NI * NJ),
        "cudaMalloc A_gpu");

    checkCuda(
        cudaMalloc((void**)&R_gpu,
                   sizeof(DATA_TYPE) * NJ * NJ),
        "cudaMalloc R_gpu");

    checkCuda(
        cudaMalloc((void**)&Q_gpu,
                   sizeof(DATA_TYPE) * NI * NJ),
        "cudaMalloc Q_gpu");

    dim3 block(DIM_THREAD_BLOCK_X);

    dim3 grid(
        (size_t)ceil(
            ((float)NI) /
            ((float)DIM_THREAD_BLOCK_X)),
        1);

    nvmlDevice_t nvml_device;

    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");

    checkNvml(
        nvmlDeviceGetHandleByIndex(
            GPU_DEVICE,
            &nvml_device),
        "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(
        A_gpu,
        R_gpu,
        Q_gpu,
        A,
        R,
        Q);

    int warmup_launches = 0;

    {
        double warmup_start = now_seconds();

        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel2_once(
                ni,
                nj,
                A_gpu,
                R_gpu,
                Q_gpu,
                grid,
                block);

            checkCuda(
                cudaDeviceSynchronize(),
                "sync after warmup launch");

            warmup_launches++;
        }
    }

    copy_inputs_to_device(
        A_gpu,
        R_gpu,
        Q_gpu,
        A,
        R,
        Q);

    cudaEvent_t measure_start;
    cudaEvent_t measure_stop;

    checkCuda(
        cudaEventCreate(&measure_start),
        "cudaEventCreate measure_start");

    checkCuda(
        cudaEventCreate(&measure_stop),
        "cudaEventCreate measure_stop");

    int measured_launches = 0;
    float measured_cuda_ms = 0.0f;

    checkCuda(
        cudaDeviceSynchronize(),
        "sync before measure");

    checkNvml(
        nvmlDeviceGetTotalEnergyConsumption(
            nvml_device,
            &energy_start_mj),
        "energy_start");

    checkCuda(
        cudaEventRecord(measure_start),
        "record measure_start");

#ifdef NCU_PROFILE
    cudaProfilerStart();
#endif

    {
        double measure_wall_start = now_seconds();

        while ((now_seconds() - measure_wall_start) < MEASURE_SECONDS)
        {
            launch_kernel2_once(
                ni,
                nj,
                A_gpu,
                R_gpu,
                Q_gpu,
                grid,
                block);

            checkCuda(
                cudaDeviceSynchronize(),
                "sync after measured launch");

            measured_launches++;
        }
    }

#ifdef NCU_PROFILE
    cudaProfilerStop();
#endif

    checkCuda(
        cudaEventRecord(measure_stop),
        "record measure_stop");

    checkCuda(
        cudaEventSynchronize(measure_stop),
        "sync measure_stop");

    checkCuda(
        cudaDeviceSynchronize(),
        "final sync");

    checkNvml(
        nvmlDeviceGetTotalEnergyConsumption(
            nvml_device,
            &energy_end_mj),
        "energy_end");

    checkCuda(
        cudaEventElapsedTime(
            &measured_cuda_ms,
            measure_start,
            measure_stop),
        "elapsed measure");

    checkCuda(cudaDeviceSynchronize(), "final sync before energy_end");

    checkCuda(
        cudaMemcpy(
            Q_outputFromGpu,
            Q_gpu,
            sizeof(DATA_TYPE) * NI * NJ,
            cudaMemcpyDeviceToHost),
        "copy Q_outputFromGpu");

    {
        double measured_cuda_s =
            (double)measured_cuda_ms / 1000.0;

        unsigned long long measured_energy_mj =
            energy_end_mj - energy_start_mj;

        double measured_energy_j =
            (double)measured_energy_mj / 1000.0;

        double avg_power_w =
            (measured_cuda_s > 0.0)
            ? (measured_energy_j / measured_cuda_s)
            : 0.0;

        printf("RESULT kernel=gramschmidt_kernel2\n");

        printf("RESULT warmup_seconds_target=%.3f\n",
               (double)WARMUP_SECONDS);

        printf("RESULT measure_seconds_target=%.3f\n",
               (double)MEASURE_SECONDS);

        printf("RESULT fixed_k=%d\n", FIXED_K);

        printf("RESULT warmup_launches=%d\n",
               warmup_launches);

        printf("RESULT measured_launches=%d\n",
               measured_launches);

        printf("RESULT measured_cuda_time_ms=%.3f\n",
               measured_cuda_ms);

        printf("RESULT measured_cuda_time_s=%.6f\n",
               measured_cuda_s);

        printf("RESULT measured_energy_mj=%llu\n",
               measured_energy_mj);

        printf("RESULT measured_energy_j=%.6f\n",
               measured_energy_j);

        printf("RESULT average_power_w=%.6f\n",
               avg_power_w);
    }

    checkCuda(
        cudaEventDestroy(measure_start),
        "destroy measure_start");

    checkCuda(
        cudaEventDestroy(measure_stop),
        "destroy measure_stop");

    checkNvml(
        nvmlShutdown(),
        "nvmlShutdown");

    checkCuda(cudaFree(A_gpu), "cudaFree A_gpu");
    checkCuda(cudaFree(R_gpu), "cudaFree R_gpu");
    checkCuda(cudaFree(Q_gpu), "cudaFree Q_gpu");
}

int main()
{
    int ni = NI;
    int nj = NJ;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, NJ, NJ, nj, nj);
    POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, NI, NJ, ni, nj);

    POLYBENCH_2D_ARRAY_DECL(
        Q_outputFromGpu,
        DATA_TYPE,
        NI,
        NJ,
        ni,
        nj);

    init_array(
        ni,
        nj,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(R),
        POLYBENCH_ARRAY(Q));

    GPU_argv_init_measurement();

    gramschmidtCudaKernel2Only(
        ni,
        nj,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(R),
        POLYBENCH_ARRAY(Q),
        POLYBENCH_ARRAY(Q_outputFromGpu));

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(R);
    POLYBENCH_FREE_ARRAY(Q);
    POLYBENCH_FREE_ARRAY(Q_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"