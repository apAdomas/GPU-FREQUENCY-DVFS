/**
 * 01_mriq_computephimag_repeat.cu
 *
 * Isolated measurement for the Parboil mri-q ComputePhiMag kernel
 * (ComputePhiMag_GPU).
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real mri-q input once
 * 2) allocate device buffers once and copy phiR / phiI to device
 * 3) warm up isolated ComputePhiMag launches for WARMUP_SECONDS
 * 4) measure isolated ComputePhiMag launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * computeQ.cu and file.cc are included directly to reuse the original kernel,
 * its launch wrapper and the input reader, so the measured kernel stays in
 * sync with upstream. Neither file depends on parboil.
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

    float *phiR_d = NULL;
    float *phiI_d = NULL;
    float *phiMag_d = NULL;

    checkCuda(cudaMalloc((void **)&phiR_d, numK * sizeof(float)), "cudaMalloc phiR_d");
    checkCuda(cudaMalloc((void **)&phiI_d, numK * sizeof(float)), "cudaMalloc phiI_d");
    checkCuda(cudaMalloc((void **)&phiMag_d, numK * sizeof(float)), "cudaMalloc phiMag_d");

    checkCuda(cudaMemcpy(phiR_d, phiR, numK * sizeof(float), cudaMemcpyHostToDevice), "copy phiR");
    checkCuda(cudaMemcpy(phiI_d, phiI, numK * sizeof(float), cudaMemcpyHostToDevice), "copy phiI");
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
            computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);
            checkCuda(cudaGetLastError(), "launch computePhiMag (warmup)");
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
            computePhiMag_GPU(numK, phiR_d, phiI_d, phiMag_d);
            checkCuda(cudaGetLastError(), "launch computePhiMag (measured)");
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

        printf("RESULT kernel=parboil_mriq_computephimag\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT num_k=%d\n", numK);
        printf("RESULT num_x=%d\n", numX);
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

    checkCuda(cudaFree(phiR_d), "cudaFree phiR_d");
    checkCuda(cudaFree(phiI_d), "cudaFree phiI_d");
    checkCuda(cudaFree(phiMag_d), "cudaFree phiMag_d");

    free(kx); free(ky); free(kz);
    free(x); free(y); free(z);
    free(phiR); free(phiI);

    return 0;
}
