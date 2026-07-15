/**
 * 01_lbm_streamcollide_repeat.cu
 *
 * Isolated measurement for the Parboil lbm stream-collide kernel
 * (performStreamCollide_kernel).
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) build the LDC grids on the host once (init + obstacle file + special
 *    cells), exactly like MAIN_initialize() in main.cc
 * 2) allocate device grids once and copy the initialized grids over
 * 3) warm up isolated stream-collide launches for WARMUP_SECONDS
 * 4) measure isolated stream-collide launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * lbm has a single kernel. We launch src -> dst repeatedly WITHOUT swapping
 * the grids, so every measured launch reads the same initialized source grid
 * and performs identical work (stable, representative per-launch timing).
 *
 * lbm.cu is included directly to reuse the original grid setup helpers and
 * the CUDA_LBM_performStreamCollide() launch wrapper, so the measured kernel
 * stays in sync with upstream. lbm.cu pulls in parboil.h (via main.h) for
 * declarations only; no pb_* functions are called, so no parboil library is
 * needed at link time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#include "measurement_common.h"

// lbm.cu uses the long-removed cudaThreadSynchronize() in a helper we do not
// call; map it to its modern equivalent so the upstream file still compiles.
#define cudaThreadSynchronize cudaDeviceSynchronize

#include "lbm.cu"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef LBM_INPUT
#define LBM_INPUT "/home/s3209105/parboil_2.5/datasets/lbm/short/input/120_120_150_ldc.of"
#endif

int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : LBM_INPUT;

    GPU_argv_init_measurement();

    // --- One-time host grid setup (mirrors MAIN_initialize in main.cc) ---
    LBM_Grid TEMP_srcGrid, TEMP_dstGrid;
    LBM_allocateGrid((float **)&TEMP_srcGrid);
    LBM_allocateGrid((float **)&TEMP_dstGrid);
    LBM_initializeGrid(TEMP_srcGrid);
    LBM_initializeGrid(TEMP_dstGrid);

    LBM_loadObstacleFile(TEMP_srcGrid, input_path);
    LBM_loadObstacleFile(TEMP_dstGrid, input_path);

    LBM_initializeSpecialCellsForLDC(TEMP_srcGrid);
    LBM_initializeSpecialCellsForLDC(TEMP_dstGrid);

    // --- Device grids ---
    LBM_Grid CUDA_srcGrid, CUDA_dstGrid;
    CUDA_LBM_allocateGrid((float **)&CUDA_srcGrid);
    CUDA_LBM_allocateGrid((float **)&CUDA_dstGrid);
    CUDA_LBM_initializeGrid((float **)&CUDA_srcGrid, (float **)&TEMP_srcGrid);
    CUDA_LBM_initializeGrid((float **)&CUDA_dstGrid, (float **)&TEMP_dstGrid);

    LBM_freeGrid((float **)&TEMP_srcGrid);
    LBM_freeGrid((float **)&TEMP_dstGrid);

    checkCuda(cudaDeviceSynchronize(), "sync after grid setup");

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            CUDA_LBM_performStreamCollide(CUDA_srcGrid, CUDA_dstGrid);
            checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
            warmup_launches++;
        }
    }

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
            CUDA_LBM_performStreamCollide(CUDA_srcGrid, CUDA_dstGrid);
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

    {
        double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
        unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
        double measured_energy_j = (double)measured_energy_mj / 1000.0;
        double avg_power_w = (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;

        printf("RESULT kernel=parboil_lbm_streamcollide\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT size_x=%d\n", SIZE_X);
        printf("RESULT size_y=%d\n", SIZE_Y);
        printf("RESULT size_z=%d\n", SIZE_Z);
        printf("RESULT total_cells=%d\n", (int)TOTAL_CELLS);
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

    CUDA_LBM_freeGrid((float **)&CUDA_srcGrid);
    CUDA_LBM_freeGrid((float **)&CUDA_dstGrid);

    return 0;
}
