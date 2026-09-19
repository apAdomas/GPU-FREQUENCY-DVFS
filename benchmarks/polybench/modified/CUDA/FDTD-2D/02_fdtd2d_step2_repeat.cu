/**
 * 02_fdtd2d_step2_repeat.cu
 *
 * Isolated measurement for fdtd_step2_kernel.
 *
 * Protocol inside this executable:
 * 1) allocate once
 * 2) copy/reset inputs to device
 * 3) warm up isolated step2 region for WARMUP_SECONDS
 * 4) reset inputs again
 * 5) measure isolated step2 region for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 6) copy back once
 * 7) exit
 *
 * Thermal warmup and clock lock are done by the launcher, not this exe.
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

#include "fdtd2d.cuh"
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
    int tmax,
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
    DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int i, j;

    for (i = 0; i < tmax; i++) {
        _fict_[i] = (DATA_TYPE)i;
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            ex[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
            ey[i][j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
            hz[i][j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
        }
    }
}

__global__ void fdtd_step2_kernel(
    int nx,
    int ny,
    DATA_TYPE* ex,
    DATA_TYPE* ey,
    DATA_TYPE* hz,
    int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    (void)nx;
    (void)ny;
    (void)ey;
    (void)t;

    if ((i < _PB_NX) && (j < _PB_NY) && (j > 0)) {
        ex[i * NY + j] = ex[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
    }
}

static void copy_inputs_to_device(
    DATA_TYPE* ex_gpu,
    DATA_TYPE* ey_gpu,
    DATA_TYPE* hz_gpu,
    DATA_TYPE (*ex)[NY],
    DATA_TYPE (*ey)[NY],
    DATA_TYPE (*hz)[NY])
{
    checkCuda(cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy ex");
    checkCuda(cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy ey");
    checkCuda(cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy hz");
    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void launch_step2_once(
    int nx,
    int ny,
    DATA_TYPE* ex_gpu,
    DATA_TYPE* ey_gpu,
    DATA_TYPE* hz_gpu,
    dim3 grid,
    dim3 block)
{
    fdtd_step2_kernel<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu, 0);
    checkCuda(cudaGetLastError(), "launch step2");
}

void fdtdCudaStep2Only(
    int tmax,
    int nx,
    int ny,
    DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
    DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny),
    DATA_TYPE POLYBENCH_2D(ex_outputFromGpu, NX, NY, nx, ny))
{
    DATA_TYPE* ex_gpu = NULL;
    DATA_TYPE* ey_gpu = NULL;
    DATA_TYPE* hz_gpu = NULL;

    checkCuda(cudaMalloc((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ex_gpu");
    checkCuda(cudaMalloc((void**)&ey_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ey_gpu");
    checkCuda(cudaMalloc((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc hz_gpu");

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid(
        (size_t)ceil(((float)NY) / ((float)block.x)),
        (size_t)ceil(((float)NX) / ((float)block.y)));

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    copy_inputs_to_device(
        ex_gpu, ey_gpu, hz_gpu,
        ex, ey, hz);

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            launch_step2_once(nx, ny, ex_gpu, ey_gpu, hz_gpu, grid, block);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

    copy_inputs_to_device(
        ex_gpu, ey_gpu, hz_gpu,
        ex, ey, hz);

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
        while ((now_seconds() - measure_wall_start) < MEASURE_SECONDS) {
            launch_step2_once(nx, ny, ex_gpu, ey_gpu, hz_gpu, grid, block);
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
    checkCuda(cudaMemcpy(ex_outputFromGpu, ex_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost),
              "copy ex_outputFromGpu");

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=step2\n");
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

    checkCuda(cudaFree(ex_gpu), "cudaFree ex_gpu");
    checkCuda(cudaFree(ey_gpu), "cudaFree ey_gpu");
    checkCuda(cudaFree(hz_gpu), "cudaFree hz_gpu");
}

int main()
{
    int tmax = TMAX;
    int nx = NX;
    int ny = NY;

    POLYBENCH_1D_ARRAY_DECL(_fict_, DATA_TYPE, TMAX, TMAX);
    POLYBENCH_2D_ARRAY_DECL(ex, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(ey, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(hz, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(ex_outputFromGpu, DATA_TYPE, NX, NY, nx, ny);

    init_arrays(
        tmax, nx, ny,
        POLYBENCH_ARRAY(_fict_),
        POLYBENCH_ARRAY(ex),
        POLYBENCH_ARRAY(ey),
        POLYBENCH_ARRAY(hz));

    GPU_argv_init_measurement();

    fdtdCudaStep2Only(
        tmax, nx, ny,
        POLYBENCH_ARRAY(_fict_),
        POLYBENCH_ARRAY(ex),
        POLYBENCH_ARRAY(ey),
        POLYBENCH_ARRAY(hz),
        POLYBENCH_ARRAY(ex_outputFromGpu));

    POLYBENCH_FREE_ARRAY(_fict_);
    POLYBENCH_FREE_ARRAY(ex);
    POLYBENCH_FREE_ARRAY(ey);
    POLYBENCH_FREE_ARRAY(hz);
    POLYBENCH_FREE_ARRAY(ex_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"