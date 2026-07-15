/**
 * 02_doitgen_kernel2_repeat.cu
 *
 * Isolated measurement for doitgen_kernel2.
 *
 * Measures a single doitgen_kernel2 launch (at fixed r = FIXED_R)
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
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#include "../../common/polybenchUtilFuncts.h"
#include "../scripts/measurement_common.h"

/* Problem size. */
#define NR 128
#define NQ 128
#define NP 128

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#define GPU_DEVICE 0

typedef float DATA_TYPE;

#ifndef FIXED_R
#define FIXED_R (NR / 2)
#endif

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

void init_array(DATA_TYPE *A, DATA_TYPE *C4)
{
    for (int i = 0; i < NR; i++)
    {
        for (int j = 0; j < NQ; j++)
        {
            for (int k = 0; k < NP; k++)
            {
                A[i * (NQ * NP) + j * NP + k] =
                    ((DATA_TYPE)i * j + k) / NP;
            }
        }
    }

    for (int i = 0; i < NP; i++)
    {
        for (int j = 0; j < NP; j++)
        {
            C4[i * NP + j] = ((DATA_TYPE)i * j) / NP;
        }
    }
}

__global__ void doitgen_kernel2(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    (void)C4;

    if ((p < NP) && (q < NQ))
    {
        A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* A_gpu,
    DATA_TYPE* C4_gpu,
    DATA_TYPE* sum_gpu,
    DATA_TYPE* A,
    DATA_TYPE* C4,
    DATA_TYPE* sum)
{
    checkCuda(
        cudaMemcpy(
            A_gpu,
            A,
            sizeof(DATA_TYPE) * NR * NQ * NP,
            cudaMemcpyHostToDevice),
        "copy A");

    checkCuda(
        cudaMemcpy(
            C4_gpu,
            C4,
            sizeof(DATA_TYPE) * NP * NP,
            cudaMemcpyHostToDevice),
        "copy C4");

    checkCuda(
        cudaMemcpy(
            sum_gpu,
            sum,
            sizeof(DATA_TYPE) * NR * NQ * NP,
            cudaMemcpyHostToDevice),
        "copy sum");

    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_kernel2_once(
    DATA_TYPE* sum_gpu,
    DATA_TYPE* A_gpu,
    DATA_TYPE* C4_gpu,
    dim3 grid,
    dim3 block)
{
    doitgen_kernel2<<<grid, block>>>(sum_gpu, A_gpu, C4_gpu, FIXED_R);
    checkCuda(cudaGetLastError(), "launch kernel2");
}

void doitgenCudaKernel2Only(
    DATA_TYPE* A,
    DATA_TYPE* C4,
    DATA_TYPE* sum,
    DATA_TYPE* A_outputFromGpu)
{
    DATA_TYPE* A_gpu = NULL;
    DATA_TYPE* C4_gpu = NULL;
    DATA_TYPE* sum_gpu = NULL;

    checkCuda(cudaMalloc((void**)&A_gpu, sizeof(DATA_TYPE) * NR * NQ * NP), "cudaMalloc A_gpu");
    checkCuda(cudaMalloc((void**)&C4_gpu, sizeof(DATA_TYPE) * NP * NP), "cudaMalloc C4_gpu");
    checkCuda(cudaMalloc((void**)&sum_gpu, sizeof(DATA_TYPE) * NR * NQ * NP), "cudaMalloc sum_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (unsigned int)ceil(((float)NP) / ((float)block.x)),
        (unsigned int)ceil(((float)NQ) / ((float)block.y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(A_gpu, C4_gpu, sum_gpu, A, C4, sum);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
        {
            launch_kernel2_once(sum_gpu, A_gpu, C4_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(A_gpu, C4_gpu, sum_gpu, A, C4, sum);

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
            launch_kernel2_once(sum_gpu, A_gpu, C4_gpu, grid, block);
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
            A_outputFromGpu,
            A_gpu,
            sizeof(DATA_TYPE) * NR * NQ * NP,
            cudaMemcpyDeviceToHost),
        "copy A_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0)
            ? (measured_energy_j / measured_cuda_s)
            : 0.0;

        printf("RESULT kernel=doitgen_kernel2\n");
        printf("RESULT warmup_seconds_target=%.3f\n", (double)WARMUP_SECONDS);
        printf("RESULT measure_seconds_target=%.3f\n", (double)MEASURE_SECONDS);
        printf("RESULT fixed_r=%d\n", FIXED_R);
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
    checkCuda(cudaFree(C4_gpu), "cudaFree C4_gpu");
    checkCuda(cudaFree(sum_gpu), "cudaFree sum_gpu");
}

int main()
{
    DATA_TYPE* A = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
    DATA_TYPE* C4 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE));
    DATA_TYPE* sum = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));
    DATA_TYPE* A_outputFromGpu = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE));

    if (!A || !C4 || !sum || !A_outputFromGpu)
    {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    init_array(A, C4);

    for (int r = 0; r < NR; r++)
    {
        for (int q = 0; q < NQ; q++)
        {
            for (int p = 0; p < NP; p++)
            {
                sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)(r + q + p) / NP;
            }
        }
    }

    GPU_argv_init_measurement();

    doitgenCudaKernel2Only(
        A,
        C4,
        sum,
        A_outputFromGpu);

    free(A);
    free(C4);
    free(sum);
    free(A_outputFromGpu);

    return 0;
}
