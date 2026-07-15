/**
 * 01_sgemm_mysgemmnt_repeat.cu
 *
 * Isolated measurement for the Parboil sgemm kernel (mysgemmNT), the
 * register-tiled dense matrix-matrix multiply C = alpha*A*B^T + beta*C.
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real A and B^T matrices once (column-major)
 * 2) allocate device buffers once and copy A / B^T to device
 * 3) warm up isolated sgemm launches for WARMUP_SECONDS
 * 4) measure isolated sgemm launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * sgemm has a single kernel. We call regtileSgemm() (one mysgemmNT launch)
 * repeatedly with beta = 0, so C is fully overwritten every launch and each
 * launch performs identical work (stable, representative per-launch timing).
 *
 * sgemm_kernel.cu and io.cc are included directly to reuse the original kernel,
 * its launch wrapper and the matrix reader. Neither depends on parboil.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#include "measurement_common.h"

#include "sgemm_kernel.cu"

// Self-contained column-major matrix reader (avoids io.cc, whose
// readColMajorMatrixFile() is missing its return statement -> UB under -O3).
static void load_col_major_matrix(const char *fn, int &nr_row, int &nr_col,
                                  std::vector<float> &v)
{
    std::ifstream f(fn);
    if (!f.good()) {
        fprintf(stderr, "cannot open matrix file %s\n", fn);
        exit(EXIT_FAILURE);
    }
    f >> nr_row >> nr_col;
    float data;
    while (f >> data) {
        v.push_back(data);
    }
}

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef SGEMM_INPUT_A
#define SGEMM_INPUT_A "/home/s3209105/parboil_2.5/datasets/sgemm/small/input/matrix1.txt"
#endif

#ifndef SGEMM_INPUT_BT
#define SGEMM_INPUT_BT "/home/s3209105/parboil_2.5/datasets/sgemm/small/input/matrix2t.txt"
#endif

int main(int argc, char **argv)
{
    const char *input_a = (argc > 1) ? argv[1] : SGEMM_INPUT_A;
    const char *input_bt = (argc > 2) ? argv[2] : SGEMM_INPUT_BT;

    GPU_argv_init_measurement();

    int matArow, matAcol;
    int matBrow, matBcol;
    std::vector<float> matA, matBT;

    load_col_major_matrix(input_a, matArow, matAcol, matA);
    // B is stored transposed: file dims map to (matBcol, matBrow).
    load_col_major_matrix(input_bt, matBcol, matBrow, matBT);

    size_t A_sz = (size_t)matArow * matAcol * sizeof(float);
    size_t B_sz = (size_t)matBrow * matBcol * sizeof(float);
    size_t C_sz = (size_t)matArow * matBcol * sizeof(float);

    float *dA = NULL;
    float *dB = NULL;
    float *dC = NULL;

    checkCuda(cudaMalloc((void **)&dA, A_sz), "cudaMalloc dA");
    checkCuda(cudaMalloc((void **)&dB, B_sz), "cudaMalloc dB");
    checkCuda(cudaMalloc((void **)&dC, C_sz), "cudaMalloc dC");

    checkCuda(cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice), "copy B^T");
    checkCuda(cudaMemset(dC, 0, C_sz), "memset dC");
    checkCuda(cudaDeviceSynchronize(), "sync after setup");

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
                         dA, matArow, dB, matBcol, 0.0f, dC, matArow);
            checkCuda(cudaGetLastError(), "launch sgemm (warmup)");
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
            regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f,
                         dA, matArow, dB, matBcol, 0.0f, dC, matArow);
            checkCuda(cudaGetLastError(), "launch sgemm (measured)");
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

        printf("RESULT kernel=parboil_sgemm_mysgemmnt\n");
        printf("RESULT input_a=%s\n", input_a);
        printf("RESULT input_bt=%s\n", input_bt);
        printf("RESULT mat_a_row=%d\n", matArow);
        printf("RESULT mat_a_col=%d\n", matAcol);
        printf("RESULT mat_b_row=%d\n", matBrow);
        printf("RESULT mat_b_col=%d\n", matBcol);
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

    checkCuda(cudaFree(dA), "cudaFree dA");
    checkCuda(cudaFree(dB), "cudaFree dB");
    checkCuda(cudaFree(dC), "cudaFree dC");

    return 0;
}
