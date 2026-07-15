/**
 * 01_spmv_jds_repeat.cu
 *
 * Isolated measurement for the Parboil spmv kernel (spmv_jds), sparse
 * matrix-vector multiply in JDS (jagged diagonal storage) format.
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real sparse matrix (.mtx -> JDS via coo_to_jds) and x vector
 * 2) allocate device buffers once, copy data, and load the JDS row pointers
 *    and non-zero counts into constant memory (exactly like main.cu)
 * 3) warm up isolated spmv launches for WARMUP_SECONDS
 * 4) measure isolated spmv launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * spmv has a single kernel. Each launch overwrites the output vector and reads
 * the same inputs, so repeated launches perform identical work (stable,
 * representative per-launch timing).
 *
 * jds_kernels.cu, file.cc and gpu_info.cc are included directly to reuse the
 * original kernel and helpers. spmv_jds.h is NOT included: it declares an
 * unused texture<float,1>, a feature removed in CUDA 12+, so we instead define
 * the two __constant__ arrays the kernel needs (identical to spmv_jds.h).
 * coo_to_jds / mmio are plain C and are linked in as separate objects.
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

// Constant memory used by spmv_jds (mirrors spmv_jds.h, which we cannot include
// because of its removed texture<float,1> declaration).
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];

#include "jds_kernels.cu"
#include "file.cc"
#include "gpu_info.cc"
#include "convert_dataset.h"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef SPMV_INPUT_MTX
#define SPMV_INPUT_MTX "/home/s3209105/parboil_2.5/datasets/spmv/small/input/1138_bus.mtx"
#endif

#ifndef SPMV_INPUT_VEC
#define SPMV_INPUT_VEC "/home/s3209105/parboil_2.5/datasets/spmv/small/input/vector.bin"
#endif

int main(int argc, char **argv)
{
    const char *input_mtx = (argc > 1) ? argv[1] : SPMV_INPUT_MTX;
    const char *input_vec_file = (argc > 2) ? argv[2] : SPMV_INPUT_VEC;

    GPU_argv_init_measurement();

    int len, depth, dim, pad = 32, nzcnt_len;
    int col_count;

    float *h_data;
    int *h_indices;
    int *h_ptr;
    int *h_perm;
    int *h_nzcnt;
    float *h_Ax_vector;
    float *h_x_vector;

    coo_to_jds(
        (char *)input_mtx,
        1,    // row padding
        pad,  // warp size
        1,    // pack size
        1,    // is mirrored?
        0,    // binary matrix
        1,    // debug level
        &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
        &col_count, &dim, &len, &nzcnt_len, &depth);

    if (depth > 5000 || nzcnt_len > 5000) {
        fprintf(stderr, "JDS dimensions exceed constant memory capacity (depth=%d nzcnt_len=%d)\n",
                depth, nzcnt_len);
        exit(EXIT_FAILURE);
    }

    h_Ax_vector = (float *)malloc(sizeof(float) * dim);
    h_x_vector = (float *)malloc(sizeof(float) * dim);
    input_vec((char *)input_vec_file, h_x_vector, dim);

    cudaDeviceProp deviceProp;
    checkCuda(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE), "cudaGetDeviceProperties");

    float *d_data, *d_x_vector, *d_Ax_vector;
    int *d_indices, *d_ptr, *d_perm, *d_nzcnt;

    checkCuda(cudaMalloc((void **)&d_data, len * sizeof(float)), "cudaMalloc d_data");
    checkCuda(cudaMalloc((void **)&d_indices, len * sizeof(int)), "cudaMalloc d_indices");
    checkCuda(cudaMalloc((void **)&d_ptr, depth * sizeof(int)), "cudaMalloc d_ptr");
    checkCuda(cudaMalloc((void **)&d_perm, dim * sizeof(int)), "cudaMalloc d_perm");
    checkCuda(cudaMalloc((void **)&d_nzcnt, nzcnt_len * sizeof(int)), "cudaMalloc d_nzcnt");
    checkCuda(cudaMalloc((void **)&d_x_vector, dim * sizeof(float)), "cudaMalloc d_x_vector");
    checkCuda(cudaMalloc((void **)&d_Ax_vector, dim * sizeof(float)), "cudaMalloc d_Ax_vector");
    checkCuda(cudaMemset(d_Ax_vector, 0, dim * sizeof(float)), "memset d_Ax_vector");

    checkCuda(cudaMemcpy(d_data, h_data, len * sizeof(float), cudaMemcpyHostToDevice), "copy data");
    checkCuda(cudaMemcpy(d_indices, h_indices, len * sizeof(int), cudaMemcpyHostToDevice), "copy indices");
    checkCuda(cudaMemcpy(d_perm, h_perm, dim * sizeof(int), cudaMemcpyHostToDevice), "copy perm");
    checkCuda(cudaMemcpy(d_x_vector, h_x_vector, dim * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    checkCuda(cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth * sizeof(int)), "copy jds_ptr_int");
    checkCuda(cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt, nzcnt_len * sizeof(int)), "copy sh_zcnt_int");

    unsigned int grid;
    unsigned int block;
    compute_active_thread(&block, &grid, nzcnt_len, pad,
                          deviceProp.major, deviceProp.minor,
                          deviceProp.warpSize, deviceProp.multiProcessorCount);

    checkCuda(cudaFuncSetCacheConfig(spmv_jds, cudaFuncCachePreferL1), "set cache config");
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
            spmv_jds<<<grid, block>>>(d_Ax_vector, d_data, d_indices, d_perm,
                                      d_x_vector, d_nzcnt, dim);
            checkCuda(cudaGetLastError(), "launch spmv (warmup)");
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
            spmv_jds<<<grid, block>>>(d_Ax_vector, d_data, d_indices, d_perm,
                                      d_x_vector, d_nzcnt, dim);
            checkCuda(cudaGetLastError(), "launch spmv (measured)");
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

        printf("RESULT kernel=parboil_spmv_jds\n");
        printf("RESULT input_mtx=%s\n", input_mtx);
        printf("RESULT input_vec=%s\n", input_vec_file);
        printf("RESULT dim=%d\n", dim);
        printf("RESULT nnz_len=%d\n", len);
        printf("RESULT jds_depth=%d\n", depth);
        printf("RESULT nzcnt_len=%d\n", nzcnt_len);
        printf("RESULT grid=%u\n", grid);
        printf("RESULT block=%u\n", block);
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

    checkCuda(cudaFree(d_data), "cudaFree d_data");
    checkCuda(cudaFree(d_indices), "cudaFree d_indices");
    checkCuda(cudaFree(d_ptr), "cudaFree d_ptr");
    checkCuda(cudaFree(d_perm), "cudaFree d_perm");
    checkCuda(cudaFree(d_nzcnt), "cudaFree d_nzcnt");
    checkCuda(cudaFree(d_x_vector), "cudaFree d_x_vector");
    checkCuda(cudaFree(d_Ax_vector), "cudaFree d_Ax_vector");

    free(h_data);
    free(h_indices);
    free(h_ptr);
    free(h_perm);
    free(h_nzcnt);
    free(h_Ax_vector);
    free(h_x_vector);

    return 0;
}
