/**
 * 02_particlefilter_sum_repeat.cu
 *
 * Isolated measurement for Rodinia Particle Filter sum_kernel.
 *
 * Usage:
 *   ./02_particlefilter_sum_repeat.exe <IszX> <IszY> <Nfr> <Nparticles>
 *
 * Example:
 *   ./02_particlefilter_sum_repeat.exe 128 128 10 100000
 *
 * IszX/IszY/Nfr are accepted for runner consistency, but this kernel only
 * depends on Nparticles and num_blocks.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <limits.h>
 #include <math.h>
 #include <string.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include "measurement_common.h"
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 const int threads_per_block = 512;
 
 __global__ void sum_kernel(double* partial_sums, int Nparticles)
 {
     int block_id = blockIdx.x;
     int i = blockDim.x * block_id + threadIdx.x;
 
     if (i == 0)
     {
         int x;
         double sum = 0.0;
         int num_blocks = ceil((double)Nparticles / (double)threads_per_block);
 
         for (x = 0; x < num_blocks; x++)
         {
             sum += partial_sums[x];
         }
 
         partial_sums[0] = sum;
     }
 }
 
 static void fatal(const char* msg)
 {
     fprintf(stderr, "ERROR: %s\n", msg);
     exit(EXIT_FAILURE);
 }
 
 template <typename T>
 T* alloc_device(size_t N, const char* name)
 {
     T* ptr = NULL;
     checkCuda(cudaMalloc((void**)&ptr, sizeof(T) * N), name);
     return ptr;
 }
 
 template <typename T>
 void dealloc_device(T* ptr, const char* name)
 {
     if (ptr != NULL)
     {
         checkCuda(cudaFree((void*)ptr), name);
     }
 }
 
 template <typename T>
 void upload_device(T* dst, const T* src, size_t N, const char* name)
 {
     checkCuda(
         cudaMemcpy(
             (void*)dst,
             (const void*)src,
             sizeof(T) * N,
             cudaMemcpyHostToDevice),
         name);
 }
 
 static void initialize_partial_sums(
     double* partial_sums,
     int num_blocks)
 {
     /*
      * Deterministic fake output from likelihood_kernel.
      * Positive values roughly simulate per-block weight sums.
      */
     for (int i = 0; i < num_blocks; i++)
     {
         partial_sums[i] = 1.0 + ((double)(i % 17) / 100.0);
     }
 }
 
 static void reset_device_state(
     double* d_partial_sums,
     const double* partial_sums,
     int num_blocks)
 {
     upload_device<double>(
         d_partial_sums,
         partial_sums,
         (size_t)num_blocks,
         "copy d_partial_sums");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_sum_once(
     double* d_partial_sums,
     int Nparticles,
     int num_blocks)
 {
     sum_kernel<<<num_blocks, threads_per_block>>>(
         d_partial_sums,
         Nparticles);
 
     checkCuda(cudaGetLastError(), "launch particlefilter_sum");
 }
 
 static void print_result(
     const char* kernel_name,
     int warmup_launches,
     int measured_launches,
     float measured_cuda_ms,
     unsigned long long energy_start_mj,
     unsigned long long energy_end_mj)
 {
     double measured_cuda_s = (double)measured_cuda_ms / 1000.0;
     unsigned long long measured_energy_mj = energy_end_mj - energy_start_mj;
     double measured_energy_j = (double)measured_energy_mj / 1000.0;
     double avg_power_w =
         (measured_cuda_s > 0.0) ? (measured_energy_j / measured_cuda_s) : 0.0;
 
     printf("RESULT kernel=%s\n", kernel_name);
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
 
 static void usage(const char* program)
 {
     fprintf(stderr, "Usage: %s <IszX> <IszY> <Nfr> <Nparticles>\n", program);
     fprintf(stderr, "Example: %s 128 128 10 100000\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 5)
     {
         usage(argv[0]);
     }
 
     int IszX = atoi(argv[1]);
     int IszY = atoi(argv[2]);
     int Nfr = atoi(argv[3]);
     int Nparticles = atoi(argv[4]);
 
     if (IszX <= 0 || IszY <= 0 || Nfr <= 1 || Nparticles <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     int num_blocks =
         (int)ceil((double)Nparticles / (double)threads_per_block);
 
     printf(
         "Loaded ParticleFilter config: kernel=sum IszX=%d IszY=%d Nfr=%d Nparticles=%d num_blocks=%d\n",
         IszX,
         IszY,
         Nfr,
         Nparticles,
         num_blocks);
 
     double* partial_sums =
         (double*)malloc(sizeof(double) * (size_t)num_blocks);
 
     if (partial_sums == NULL)
     {
         fatal("unable to allocate host partial_sums");
     }
 
     initialize_partial_sums(partial_sums, num_blocks);
 
     double* d_partial_sums =
         alloc_device<double>((size_t)num_blocks, "cudaMalloc d_partial_sums");
 
     reset_device_state(
         d_partial_sums,
         partial_sums,
         num_blocks);
 
     nvmlDevice_t nvml_device;
     unsigned long long energy_start_mj = 0;
     unsigned long long energy_end_mj = 0;
 
     checkNvml(nvmlInit(), "nvmlInit");
     checkNvml(
         nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device),
         "nvmlDeviceGetHandleByIndex");
 
     int warmup_launches = 0;
     {
         double warmup_start = now_seconds();
 
         while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
         {
             launch_sum_once(
                 d_partial_sums,
                 Nparticles,
                 num_blocks);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because sum_kernel overwrites partial_sums[0].
      */
     reset_device_state(
         d_partial_sums,
         partial_sums,
         num_blocks);
 
     checkCuda(cudaDeviceSynchronize(), "sync before measure");
 
     cudaEvent_t measure_start;
     cudaEvent_t measure_stop;
 
     checkCuda(cudaEventCreate(&measure_start), "cudaEventCreate measure_start");
     checkCuda(cudaEventCreate(&measure_stop), "cudaEventCreate measure_stop");
 
     int measured_launches = 0;
     float measured_cuda_ms = 0.0f;
 
     checkNvml(
         nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_start_mj),
         "energy_start");
 
     checkCuda(cudaEventRecord(measure_start), "record measure_start");

#ifdef NCU_PROFILE
    cudaProfilerStart();
#endif
 
     {
         double measure_wall_start = now_seconds();
 
         while ((now_seconds() - measure_wall_start) < MEASURE_SECONDS)
         {
             launch_sum_once(
                 d_partial_sums,
                 Nparticles,
                 num_blocks);
 
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
 
     checkNvml(
         nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_end_mj),
         "energy_end");
 
     checkCuda(
         cudaEventElapsedTime(&measured_cuda_ms, measure_start, measure_stop),
         "elapsed measure");
 
     print_result(
         "particlefilter_sum",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<double>(d_partial_sums, "cudaFree d_partial_sums");
 
     free(partial_sums);
 
     return EXIT_SUCCESS;
 }