/**
 * 02_mriq_computeq_repeat.cu
 *
 * Isolated measurement for the Parboil mri-q ComputeQ kernel
 * (ComputeQ_GPU). This is the heavy compute kernel of mri-q.
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real mri-q input once
 * 2) run the upstream PhiMag stage ONCE and build the kValues array, exactly
 *    like main.cu, so ComputeQ has valid inputs
 * 3) allocate device buffers once, copy x/y/z, and load one k-space tile into
 *    constant memory (ck)
 * 4) warm up isolated ComputeQ launches for WARMUP_SECONDS
 * 5) measure isolated ComputeQ launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 6) exit
 *
 * The full benchmark issues ceil(numK/1024) ComputeQ launches, each preceded
 * by a constant-memory copy of the next k-tile. For isolated per-kernel
 * measurement we fix one tile (kGlobalIndex = 0) and repeat a single launch,
 * which performs identical work every iteration (the cos/sin K-loop is fixed).
 *
 * computeQ.cu and file.cc are included directly to reuse the original kernel,
 * its launch wrapper and the input reader. Neither depends on parboil.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvml.h>

#include "measurement_common.h"

#include "computeQ.cu"
#include "file.cc"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef MRIQ_INPUT
#define MRIQ_INPUT "/home/s3209105/parboil_2.5/datasets/mri-q/small/input/32_32_32_dataset.bin"
#endif

int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : MRIQ_INPUT;

    GPU_argv_init_measurement();

    int numK, numX;
    float *kx, *ky, *kz;
    float *x, *y, *z;
    float *phiR, *phiI;

    inputData((char *)input_path, &numK, &numX,
              &kx, &ky, &kz, &x, &y, &z, &phiR, &phiI);

    float *phiMag = NULL;
    float *Qr = NULL;
    float *Qi = NULL;
    createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

    // --- One-time setup: PhiMag stage on the GPU (mirrors main.cu section 1) ---
    {
        float *phiR_d = NULL;
        float *phiI_d = NULL;
        float *phiMag_d = NULL;

        checkCuda(cudaMalloc((void **)&phiR_d, numK * sizeof(float)), "cudaMalloc phiR_d");
        checkCuda(cudaMalloc((void **)&phiI_d, numK * sizeof(float)), "cudaMalloc phiI_d");
        checkCuda(cudaMalloc((void **)&phiMag_d, numK * sizeof(float)), "cudaMalloc phiMag_d");

        checkCuda(cudaMemcpy(phiR_d, phiR, numK * sizeof(float), cudaMemcpyHostToDevice), "copy phiR");
        checkCuda(cudaMemcpy(phiI_d, phiI, numK * sizeof(float), cudaMemcpyHostToDevice), "copy phiI");

        computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);
        checkCuda(cudaGetLastError(), "launch computePhiMag (setup)");
        checkCuda(cudaDeviceSynchronize(), "sync after phiMag");

        checkCuda(cudaMemcpy(phiMag, phiMag_d, numK * sizeof(float), cudaMemcpyDeviceToHost),
                  "copy phiMag back");

        checkCuda(cudaFree(phiR_d), "cudaFree phiR_d");
        checkCuda(cudaFree(phiI_d), "cudaFree phiI_d");
        checkCuda(cudaFree(phiMag_d), "cudaFree phiMag_d");
    }

    struct kValues *kVals = (struct kValues *)calloc(numK, sizeof(struct kValues));
    for (int k = 0; k < numK; k++) {
        kVals[k].Kx = kx[k];
        kVals[k].Ky = ky[k];
        kVals[k].Kz = kz[k];
        kVals[k].PhiMag = phiMag[k];
    }
    free(phiMag);

    // --- Device buffers for the Q stage (mirrors main.cu section 2) ---
    float *x_d = NULL;
    float *y_d = NULL;
    float *z_d = NULL;
    float *Qr_d = NULL;
    float *Qi_d = NULL;

    checkCuda(cudaMalloc((void **)&x_d, numX * sizeof(float)), "cudaMalloc x_d");
    checkCuda(cudaMalloc((void **)&y_d, numX * sizeof(float)), "cudaMalloc y_d");
    checkCuda(cudaMalloc((void **)&z_d, numX * sizeof(float)), "cudaMalloc z_d");
    checkCuda(cudaMemcpy(x_d, x, numX * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    checkCuda(cudaMemcpy(y_d, y, numX * sizeof(float), cudaMemcpyHostToDevice), "copy y");
    checkCuda(cudaMemcpy(z_d, z, numX * sizeof(float), cudaMemcpyHostToDevice), "copy z");

    checkCuda(cudaMalloc((void **)&Qr_d, numX * sizeof(float)), "cudaMalloc Qr_d");
    checkCuda(cudaMalloc((void **)&Qi_d, numX * sizeof(float)), "cudaMalloc Qi_d");
    checkCuda(cudaMemset(Qr_d, 0, numX * sizeof(float)), "memset Qr_d");
    checkCuda(cudaMemset(Qi_d, 0, numX * sizeof(float)), "memset Qi_d");

    // Load one k-space tile into constant memory (the first grid tile).
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK);
    checkCuda(cudaMemcpyToSymbol(ck, kVals, numElems * sizeof(struct kValues), 0),
              "copy ck constant tile");

    int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
    if (numX % KERNEL_Q_THREADS_PER_BLOCK)
        QBlocks++;
    dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
    dim3 DimQGrid(QBlocks, 1);

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
            ComputeQ_GPU<<<DimQGrid, DimQBlock>>>(numK, 0, x_d, y_d, z_d, Qr_d, Qi_d);
            checkCuda(cudaGetLastError(), "launch ComputeQ (warmup)");
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
            ComputeQ_GPU<<<DimQGrid, DimQBlock>>>(numK, 0, x_d, y_d, z_d, Qr_d, Qi_d);
            checkCuda(cudaGetLastError(), "launch ComputeQ (measured)");
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

        printf("RESULT kernel=parboil_mriq_computeq\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT num_k=%d\n", numK);
        printf("RESULT num_x=%d\n", numX);
        printf("RESULT k_elems_per_tile=%d\n", numElems);
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

    checkCuda(cudaFree(x_d), "cudaFree x_d");
    checkCuda(cudaFree(y_d), "cudaFree y_d");
    checkCuda(cudaFree(z_d), "cudaFree z_d");
    checkCuda(cudaFree(Qr_d), "cudaFree Qr_d");
    checkCuda(cudaFree(Qi_d), "cudaFree Qi_d");

    free(kx); free(ky); free(kz);
    free(x); free(y); free(z);
    free(phiR); free(phiI);
    free(kVals);
    free(Qr); free(Qi);

    return 0;
}
