/**
 * 02_histo_intermediates_repeat.cu
 *
 * Isolated measurement for the Parboil histo intermediates kernel
 * (histo_intermediates_kernel).
 *
 * Protocol (matches the polybench/rodinia *_repeat.cu pattern):
 * 1) load the real histo input image once
 * 2) allocate device buffers once and copy input to device
 * 3) warm up isolated intermediates launches for WARMUP_SECONDS
 * 4) measure isolated intermediates launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * The intermediates kernel only reads the input image and writes
 * sm_mappings, so no upstream kernel is required to set it up.
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

#include "histo_intermediates.cu"

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

static void launch_intermediates_once(unsigned int *d_input, uchar4 *d_sm_mappings, const HistoInput *h)
{
    histo_intermediates_kernel<<<dim3((h->img_height + UNROLL - 1) / UNROLL),
                                 dim3((h->img_width + 1) / 2)>>>(
        (uint2 *)d_input,
        h->img_height,
        h->img_width,
        (h->img_width + 1) / 2,
        d_sm_mappings);
    checkCuda(cudaGetLastError(), "launch intermediates");
}

int main(int argc, char **argv)
{
    const char *input_path = (argc > 1) ? argv[1] : HISTO_INPUT;

    HistoInput h;
    load_input(input_path, &h);

    GPU_argv_init_measurement();

    int size = (int)(h.img_height * h.img_width);

    unsigned int *d_input = NULL;
    uchar4 *d_sm_mappings = NULL;

    checkCuda(cudaMalloc((void **)&d_input,
                         (size_t)h.even_width * h.padded_height * sizeof(unsigned int)),
              "cudaMalloc d_input");
    checkCuda(cudaMalloc((void **)&d_sm_mappings,
                         (size_t)h.img_width * h.img_height * sizeof(uchar4)),
              "cudaMalloc d_sm_mappings");

    copy_input_to_device(d_input, &h);

    nvmlDevice_t nvml_device;
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;

    checkNvml(nvmlInit(), "nvmlInit");
    checkNvml(nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device), "nvmlDeviceGetHandleByIndex");

    int warmup_launches = 0;
    {
        double warmup_start = now_seconds();
        while ((now_seconds() - warmup_start) < WARMUP_SECONDS) {
            launch_intermediates_once(d_input, d_sm_mappings, &h);
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
            launch_intermediates_once(d_input, d_sm_mappings, &h);
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

        printf("RESULT kernel=parboil_histo_intermediates\n");
        printf("RESULT input_file=%s\n", input_path);
        printf("RESULT img_width=%u\n", h.img_width);
        printf("RESULT img_height=%u\n", h.img_height);
        printf("RESULT histo_width=%u\n", h.histo_width);
        printf("RESULT histo_height=%u\n", h.histo_height);
        printf("RESULT input_elements=%d\n", size);
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
    checkCuda(cudaFree(d_sm_mappings), "cudaFree d_sm_mappings");
    free(h.img);

    return 0;
}
