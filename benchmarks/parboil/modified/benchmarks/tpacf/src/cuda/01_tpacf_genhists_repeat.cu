/**
 * 01_tpacf_genhists_repeat.cu
 *
 * Isolated measurement for the Parboil tpacf kernel (gen_hists), the two-point
 * angular correlation function histogram builder.
 *
 * Protocol (matches the polybench/rodinia/parboil *_repeat.cu pattern):
 * 1) read the real input once: 1 data file + NUM_SETS random files, each with
 *    NUM_ELEMENTS points, then split into contiguous x/y/z arrays (exactly as
 *    main.cu does)
 * 2) allocate device buffers once, copy the point data, and initialise the bin
 *    boundary constant (dev_binb) via initBinB
 * 3) warm up isolated TPACF (gen_hists) launches for WARMUP_SECONDS
 * 4) measure isolated launches for MEASURE_SECONDS
 *    - CUDA events for time
 *    - NVML total energy for energy
 * 5) exit
 *
 * tpacf has a single kernel. gen_hists fully overwrites the global histogram
 * buffer on every launch (it accumulates only into per-block shared memory that
 * is zeroed at kernel start), so every measured launch performs identical work.
 *
 * tpacf_kernel.cu is included directly to reuse the original kernel, the TPACF
 * launch wrapper, the dev_binb constant, the NUM_SETS/NUM_ELEMENTS globals, and
 * initBinB. model_io.cc is included to reuse readdatafile.
 *
 * tpacf_kernel.cu pulls in model.h -> parboil.h, and initBinB references
 * pb_SwitchToTimer purely for upstream timing. We do not link the parboil
 * runtime, so we provide a no-op pb_SwitchToTimer stub to satisfy that single
 * reference; the real cudaMemcpyToSymbol work inside initBinB still runs.
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

#include "model.h"
#include "tpacf_kernel.cu"
#include "model_io.cc"

#ifndef WARMUP_SECONDS
#define WARMUP_SECONDS 25.0
#endif

#ifndef MEASURE_SECONDS
#define MEASURE_SECONDS 5.0
#endif

#ifndef TPACF_INPUT_DIR
#define TPACF_INPUT_DIR "/home/s3209105/parboil_2.5/datasets/tpacf/small/input"
#endif

#ifndef TPACF_NPOINTS
#define TPACF_NPOINTS 487
#endif

#ifndef TPACF_RANDOM_COUNT
#define TPACF_RANDOM_COUNT 100
#endif

/* No-op stub: initBinB calls pb_SwitchToTimer only for upstream timing. We do
 * not link the parboil runtime, so resolve that one reference here. */
extern "C" void pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
    (void)timers;
    (void)timer;
}

int main(int argc, char **argv)
{
    const char *input_dir = (argc > 1) ? argv[1] : TPACF_INPUT_DIR;
    int npoints = (argc > 2) ? atoi(argv[2]) : TPACF_NPOINTS;
    int random_count = (argc > 3) ? atoi(argv[3]) : TPACF_RANDOM_COUNT;

    if (npoints < 1 || random_count < 1) {
        fprintf(stderr, "invalid npoints=%d random_count=%d\n", npoints, random_count);
        exit(EXIT_FAILURE);
    }

    GPU_argv_init_measurement();

    NUM_ELEMENTS = npoints;
    NUM_SETS = random_count;
    int num_elements = NUM_ELEMENTS;

    unsigned mem_size = (1 + NUM_SETS) * num_elements * sizeof(struct cartesian);
    unsigned f_mem_size = (1 + NUM_SETS) * num_elements * sizeof(REAL);

    struct cartesian *h_all_data = (struct cartesian *)malloc(mem_size);
    if (!h_all_data) {
        fprintf(stderr, "failed to allocate host point buffer\n");
        exit(EXIT_FAILURE);
    }

    char path[4096];
    struct cartesian *working = h_all_data;

    snprintf(path, sizeof(path), "%s/Datapnts.1", input_dir);
    if (readdatafile(path, working, num_elements) != num_elements) {
        fprintf(stderr, "failed to read %d points from %s\n", num_elements, path);
        exit(EXIT_FAILURE);
    }
    working += num_elements;

    for (int i = 0; i < NUM_SETS; i++) {
        snprintf(path, sizeof(path), "%s/Randompnts.%d", input_dir, i + 1);
        if (readdatafile(path, working, num_elements) != num_elements) {
            fprintf(stderr, "failed to read %d points from %s\n", num_elements, path);
            exit(EXIT_FAILURE);
        }
        working += num_elements;
    }

    REAL *h_x_data = (REAL *)malloc(3 * f_mem_size);
    REAL *h_y_data = h_x_data + NUM_ELEMENTS * (NUM_SETS + 1);
    REAL *h_z_data = h_y_data + NUM_ELEMENTS * (NUM_SETS + 1);
    for (int i = 0; i < (NUM_SETS + 1); ++i) {
        for (int j = 0; j < NUM_ELEMENTS; ++j) {
            h_x_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].x;
            h_y_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].y;
            h_z_data[i * NUM_ELEMENTS + j] = h_all_data[i * NUM_ELEMENTS + j].z;
        }
    }
    free(h_all_data);

    REAL *d_x_data;
    checkCuda(cudaMalloc((void **)&d_x_data, 3 * f_mem_size), "cudaMalloc d_x_data");
    REAL *d_y_data = d_x_data + NUM_ELEMENTS * (NUM_SETS + 1);
    REAL *d_z_data = d_y_data + NUM_ELEMENTS * (NUM_SETS + 1);

    hist_t *d_hists;
    checkCuda(cudaMalloc((void **)&d_hists, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t)),
              "cudaMalloc d_hists");

    /* initialise the bin boundary constant (dev_binb). The pb_SwitchToTimer
     * calls inside are no-ops via our stub; the cudaMemcpyToSymbol is real. */
    struct pb_TimerSet dummy_timers;
    initBinB(&dummy_timers);

    checkCuda(cudaMemcpy(d_x_data, h_x_data, 3 * f_mem_size, cudaMemcpyHostToDevice),
              "copy point data");
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
            TPACF(d_hists, d_x_data, d_y_data, d_z_data);
            checkCuda(cudaGetLastError(), "launch tpacf (warmup)");
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
            TPACF(d_hists, d_x_data, d_y_data, d_z_data);
            checkCuda(cudaGetLastError(), "launch tpacf (measured)");
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

        printf("RESULT kernel=parboil_tpacf_genhists\n");
        printf("RESULT input_dir=%s\n", input_dir);
        printf("RESULT npoints=%d\n", npoints);
        printf("RESULT random_count=%d\n", random_count);
        printf("RESULT num_bins=%d\n", NUM_BINS);
        printf("RESULT grid_blocks=%d\n", NUM_SETS * 2 + 1);
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

    checkCuda(cudaFree(d_x_data), "cudaFree d_x_data");
    checkCuda(cudaFree(d_hists), "cudaFree d_hists");
    free(h_x_data);

    return 0;
}
