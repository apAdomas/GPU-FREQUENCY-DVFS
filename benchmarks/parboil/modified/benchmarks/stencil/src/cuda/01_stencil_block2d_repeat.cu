/**
 * 01_stencil_block2d_repeat.cu
 *
 * Isolated measurement for the Parboil stencil kernel
 * (block2D_hybrid_coarsen_x), a 7-point Jacobi stencil with x-coarsening.
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real input grid (raw float volume nx*ny*nz) once
 * 2) allocate device buffers once, copy A0, and seed Anext from A0 (so the
 *    untouched boundary cells are valid), exactly like main.cu
 * 3) warm up isolated stencil launches for WARMUP_SECONDS
 * 4) measure isolated stencil launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * stencil has a single kernel. We launch A0 -> Anext repeatedly WITHOUT
 * swapping the buffers, so every measured launch reads the same input grid and
 * performs identical work (stable, representative per-launch timing).
 *
 * kernels.cu is included directly to reuse the original kernel; it pulls in
 * common.h (Index3D). Neither depends on parboil.
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

#include "kernels.cu"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef STENCIL_INPUT
#define STENCIL_INPUT "/home/s3209105/parboil_2.5/datasets/stencil/small/input/128x128x32.bin"
#endif

#ifndef STENCIL_NX
#define STENCIL_NX 128
#endif
#ifndef STENCIL_NY
#define STENCIL_NY 128
#endif
#ifndef STENCIL_NZ
#define STENCIL_NZ 32
#endif

int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : STENCIL_INPUT;
    int nx = (argc > 2) ? atoi(argv[2]) : STENCIL_NX;
    int ny = (argc > 3) ? atoi(argv[3]) : STENCIL_NY;
    int nz = (argc > 4) ? atoi(argv[4]) : STENCIL_NZ;

    if (nx < 1 || ny < 1 || nz < 1) {
        fprintf(stderr, "invalid grid dimensions %d %d %d\n", nx, ny, nz);
        exit(EXIT_FAILURE);
    }

    GPU_argv_init_measurement();

    int size = nx * ny * nz;

    float *h_A0 = (float *)malloc(sizeof(float) * size);

    FILE *fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open stencil input %s\n", input_path);
        exit(EXIT_FAILURE);
    }
    size_t got = fread(h_A0, sizeof(float), size, fp);
    fclose(fp);
    if (got != (size_t)size) {
        fprintf(stderr, "stencil input has %zu floats, expected %d\n", got, size);
        exit(EXIT_FAILURE);
    }

    float c0 = 1.0f / 6.0f;
    float c1 = 1.0f / 6.0f / 6.0f;

    float *d_A0 = NULL;
    float *d_Anext = NULL;

    checkCuda(cudaMalloc((void **)&d_A0, size * sizeof(float)), "cudaMalloc d_A0");
    checkCuda(cudaMalloc((void **)&d_Anext, size * sizeof(float)), "cudaMalloc d_Anext");

    checkCuda(cudaMemcpy(d_A0, h_A0, size * sizeof(float), cudaMemcpyHostToDevice), "copy A0");
    checkCuda(cudaMemcpy(d_Anext, d_A0, size * sizeof(float), cudaMemcpyDeviceToDevice), "seed Anext");
    checkCuda(cudaDeviceSynchronize(), "sync after setup");

    int tx = 32;
    int ty = 4;
    dim3 block(tx, ty, 1);
    dim3 grid((nx + tx * 2 - 1) / (tx * 2), (ny + ty - 1) / ty, 1);
    int sh_size = tx * 2 * ty * sizeof(float);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            block2D_hybrid_coarsen_x<<<grid, block, sh_size>>>(c0, c1, d_A0, d_Anext, nx, ny, nz);
            checkCuda(cudaGetLastError(), "launch stencil (warmup)");
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
            block2D_hybrid_coarsen_x<<<grid, block, sh_size>>>(c0, c1, d_A0, d_Anext, nx, ny, nz);
            checkCuda(cudaGetLastError(), "launch stencil (measured)");
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

        printf("RESULT kernel=parboil_stencil_block2d\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT nx=%d\n", nx);
        printf("RESULT ny=%d\n", ny);
        printf("RESULT nz=%d\n", nz);
        printf("RESULT grid_points=%d\n", size);
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

    checkCuda(cudaFree(d_A0), "cudaFree d_A0");
    checkCuda(cudaFree(d_Anext), "cudaFree d_Anext");
    free(h_A0);

    return 0;
}
