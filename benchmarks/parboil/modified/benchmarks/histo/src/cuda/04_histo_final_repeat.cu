/**
 * 04_histo_final_repeat.cu
 *
 * Isolated measurement for the Parboil histo final kernel
 * (histo_final_kernel).
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real histo input image once
 * 2) allocate device buffers once and copy input to device
 * 3) run the upstream prescan + intermediates + main kernels ONCE to
 *    produce valid sub-histograms (the inputs of the final kernel)
 * 4) warm up isolated final launches for WARMUP_SECONDS
 * 5) measure isolated final launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 6) exit
 *
 * Only histo_final_kernel is inside the measured region.
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

#include "util.h"
#include "measurement_common.h"

#include "histo_prescan.cu"
#include "histo_intermediates.cu"
#include "histo_main.cu"
#include "histo_final.cu"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef HISTO_INPUT
#define HISTO_INPUT "/home/s3209105/parboil_2.5/datasets/histo/default/input/img.bin"
#endif

typedef struct {
    unsigned int img_width;
    unsigned int img_height;
    unsigned int histo_width;
    unsigned int histo_height;
    unsigned int even_width;
    unsigned int padded_height;
    unsigned int *img;
} HistoInput;

static void load_input(const char *path, HistoInput *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open histo input %s\n", path);
        exit(EXIT_FAILURE);
    }

    int r = 0;
    r += (int)fread(&h->img_width, sizeof(unsigned int), 1, f);
    r += (int)fread(&h->img_height, sizeof(unsigned int), 1, f);
    r += (int)fread(&h->histo_width, sizeof(unsigned int), 1, f);
    r += (int)fread(&h->histo_height, sizeof(unsigned int), 1, f);
    if (r != 4) {
        fprintf(stderr, "error reading histo header\n");
        exit(EXIT_FAILURE);
    }

    h->img = (unsigned int *)malloc((size_t)h->img_width * h->img_height * sizeof(unsigned int));
    size_t got = fread(h->img, sizeof(unsigned int), (size_t)h->img_width * h->img_height, f);
    fclose(f);

    if (got != (size_t)h->img_width * h->img_height) {
        fprintf(stderr, "error reading histo image data\n");
        exit(EXIT_FAILURE);
    }

    h->even_width = ((h->img_width + 1) / 2) * 2;
    h->padded_height = ((h->img_height + UNROLL - 1) / UNROLL) * UNROLL;
}

static void copy_input_to_device(unsigned int *d_input, const HistoInput *h)
{
    for (unsigned int y = 0; y < h->img_height; y++) {
        checkCuda(
            cudaMemcpy(
                &d_input[y * h->even_width],
                &h->img[y * h->img_width],
                h->img_width * sizeof(unsigned int),
                cudaMemcpyHostToDevice),
            "copy input row");
    }
    checkCuda(cudaDeviceSynchronize(), "sync after input copy");
}

static void reset_ranges(unsigned int *d_ranges)
{
    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    checkCuda(
        cudaMemcpy(d_ranges, ranges_h, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice),
        "reset ranges");
    checkCuda(cudaDeviceSynchronize(), "sync after reset ranges");
}

int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : HISTO_INPUT;

    HistoInput h;
    load_input(input_path, &h);

    GPU_argv_init_measurement();

    unsigned int *d_input = NULL;
    unsigned int *d_ranges = NULL;
    uchar4 *d_sm_mappings = NULL;
    unsigned int *d_global_subhisto = NULL;
    unsigned short *d_global_histo = NULL;
    unsigned int *d_global_overflow = NULL;
    unsigned char *d_final_histo = NULL;

    checkCuda(cudaMalloc((void **)&d_input,
                         (size_t)h.even_width * h.padded_height * sizeof(unsigned int)),
              "cudaMalloc d_input");
    checkCuda(cudaMalloc((void **)&d_ranges, 2 * sizeof(unsigned int)), "cudaMalloc d_ranges");
    checkCuda(cudaMalloc((void **)&d_sm_mappings,
                         (size_t)h.img_width * h.img_height * sizeof(uchar4)),
              "cudaMalloc d_sm_mappings");
    checkCuda(cudaMalloc((void **)&d_global_subhisto,
                         (size_t)BLOCK_X * h.img_width * h.histo_height * sizeof(unsigned int)),
              "cudaMalloc d_global_subhisto");
    checkCuda(cudaMalloc((void **)&d_global_histo,
                         (size_t)h.img_width * h.histo_height * sizeof(unsigned short)),
              "cudaMalloc d_global_histo");
    checkCuda(cudaMalloc((void **)&d_global_overflow,
                         (size_t)h.img_width * h.histo_height * sizeof(unsigned int)),
              "cudaMalloc d_global_overflow");
    checkCuda(cudaMalloc((void **)&d_final_histo,
                         (size_t)h.img_width * h.histo_height * sizeof(unsigned char)),
              "cudaMalloc d_final_histo");

    checkCuda(cudaMemset(d_final_histo, 0,
                         (size_t)h.img_width * h.histo_height * sizeof(unsigned char)),
              "memset final_histo");

    copy_input_to_device(d_input, &h);

    // --- One-time setup: produce sub-histograms via prescan + intermediates + main ---
    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    reset_ranges(d_ranges);

    histo_prescan_kernel<<<PRESCAN_BLOCKS_X, PRESCAN_THREADS>>>(
        (unsigned int *)d_input, (int)(h.img_height * h.img_width), d_ranges);
    checkCuda(cudaGetLastError(), "launch prescan (setup)");

    checkCuda(cudaMemcpy(ranges_h, d_ranges, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost),
              "copy ranges back");

    checkCuda(cudaMemset(d_global_subhisto, 0,
                         (size_t)h.img_width * h.histo_height * sizeof(unsigned int)),
              "memset subhisto");

    histo_intermediates_kernel<<<dim3((h.img_height + UNROLL - 1) / UNROLL),
                                 dim3((h.img_width + 1) / 2)>>>(
        (uint2 *)d_input, h.img_height, h.img_width, (h.img_width + 1) / 2, d_sm_mappings);
    checkCuda(cudaGetLastError(), "launch intermediates (setup)");

    histo_main_kernel<<<dim3(BLOCK_X, ranges_h[1] - ranges_h[0] + 1), dim3(THREADS)>>>(
        d_sm_mappings, h.img_height * h.img_width, ranges_h[0], ranges_h[1],
        h.histo_height, h.histo_width,
        d_global_subhisto, (unsigned int *)d_global_histo, d_global_overflow);
    checkCuda(cudaGetLastError(), "launch main (setup)");
    checkCuda(cudaDeviceSynchronize(), "sync after setup");

    dim3 final_grid(BLOCK_X * 3);
    dim3 final_block(512);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            histo_final_kernel<<<final_grid, final_block>>>(
                ranges_h[0], ranges_h[1], h.histo_height, h.histo_width,
                d_global_subhisto, (unsigned int *)d_global_histo, d_global_overflow,
                (unsigned int *)d_final_histo);
            checkCuda(cudaGetLastError(), "launch final (warmup)");
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
            histo_final_kernel<<<final_grid, final_block>>>(
                ranges_h[0], ranges_h[1], h.histo_height, h.histo_width,
                d_global_subhisto, (unsigned int *)d_global_histo, d_global_overflow,
                (unsigned int *)d_final_histo);
            checkCuda(cudaGetLastError(), "launch final (measured)");
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

        printf("RESULT kernel=parboil_histo_final\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT img_width=%u\n", h.img_width);
        printf("RESULT img_height=%u\n", h.img_height);
        printf("RESULT histo_width=%u\n", h.histo_width);
        printf("RESULT histo_height=%u\n", h.histo_height);
        printf("RESULT input_elements=%d\n", (int)(h.img_height * h.img_width));
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

    checkCuda(cudaFree(d_input), "cudaFree d_input");
    checkCuda(cudaFree(d_ranges), "cudaFree d_ranges");
    checkCuda(cudaFree(d_sm_mappings), "cudaFree d_sm_mappings");
    checkCuda(cudaFree(d_global_subhisto), "cudaFree d_global_subhisto");
    checkCuda(cudaFree(d_global_histo), "cudaFree d_global_histo");
    checkCuda(cudaFree(d_global_overflow), "cudaFree d_global_overflow");
    checkCuda(cudaFree(d_final_histo), "cudaFree d_final_histo");
    free(h.img);

    return 0;
}
