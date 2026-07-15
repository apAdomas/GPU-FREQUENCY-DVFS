/**
 * 03_particlefilter_normalize_weights_repeat.cu
 *
 * Isolated measurement for Rodinia Particle Filter normalize_weights_kernel.
 *
 * Usage:
 *   ./03_particlefilter_normalize_weights_repeat.exe <IszX> <IszY> <Nfr> <Nparticles>
 *
 * Example:
 *   ./03_particlefilter_normalize_weights_repeat.exe 128 128 10 100000
 *
 * IszX/IszY/Nfr are accepted for runner consistency, but this kernel only
 * depends on Nparticles, weights, partial_sums, CDF, u, and seed.
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
 
 __device__ double d_randu(int* seed, int index)
 {
     int M = INT_MAX;
     int A = 1103515245;
     int C = 12345;
 
     int num = A * seed[index] + C;
     seed[index] = num % M;
 
     return fabs(seed[index] / ((double)M));
 }
 
 __device__ void cdfCalc(double* CDF, double* weights, int Nparticles)
 {
     CDF[0] = weights[0];
 
     for (int x = 1; x < Nparticles; x++)
     {
         CDF[x] = weights[x] + CDF[x - 1];
     }
 }
 
 /*
  * Target kernel from Rodinia particlefilter_float.
  */
 __global__ void normalize_weights_kernel(
     double* weights,
     int Nparticles,
     double* partial_sums,
     double* CDF,
     double* u,
     int* seed)
 {
     int block_id = blockIdx.x;
     int i = blockDim.x * block_id + threadIdx.x;
 
     __shared__ double u1;
     __shared__ double sumWeights;
 
     if (0 == threadIdx.x)
     {
         sumWeights = partial_sums[0];
     }
 
     __syncthreads();
 
     if (i < Nparticles)
     {
         weights[i] = weights[i] / sumWeights;
     }
 
     __syncthreads();
 
     if (i == 0)
     {
         cdfCalc(CDF, weights, Nparticles);
         u[0] = (1.0 / ((double)Nparticles)) * d_randu(seed, i);
     }
 
     __syncthreads();
 
     if (0 == threadIdx.x)
     {
         u1 = u[0];
     }
 
     __syncthreads();
 
     if (i < Nparticles)
     {
         u[i] = u1 + i / ((double)Nparticles);
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
 
 static void initialize_host_state(
     double* weights,
     double* partial_sums,
     double* CDF,
     double* u,
     int* seed,
     int Nparticles,
     int num_blocks)
 {
     /*
      * Simulate state after likelihood + sum:
      * - weights are positive unnormalized values
      * - partial_sums[0] contains total sum
      */
     double total = 0.0;
 
     for (int i = 0; i < Nparticles; i++)
     {
         weights[i] = 1.0 + ((double)(i % 31) / 100.0);
         CDF[i] = 0.0;
         u[i] = 0.0;
         seed[i] = 1337 + i * 17;
 
         total += weights[i];
     }
 
     for (int i = 0; i < num_blocks; i++)
     {
         partial_sums[i] = 0.0;
     }
 
     partial_sums[0] = total;
 }
 
 static void reset_device_state(
     double* d_weights,
     double* d_partial_sums,
     double* d_CDF,
     double* d_u,
     int* d_seed,
     const double* weights,
     const double* partial_sums,
     const double* CDF,
     const double* u,
     const int* seed,
     int Nparticles,
     int num_blocks)
 {
     upload_device<double>(
         d_weights,
         weights,
         (size_t)Nparticles,
         "copy d_weights");
 
     upload_device<double>(
         d_partial_sums,
         partial_sums,
         (size_t)num_blocks,
         "copy d_partial_sums");
 
     upload_device<double>(
         d_CDF,
         CDF,
         (size_t)Nparticles,
         "copy d_CDF");
 
     upload_device<double>(
         d_u,
         u,
         (size_t)Nparticles,
         "copy d_u");
 
     upload_device<int>(
         d_seed,
         seed,
         (size_t)Nparticles,
         "copy d_seed");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_normalize_once(
     double* d_weights,
     double* d_partial_sums,
     double* d_CDF,
     double* d_u,
     int* d_seed,
     int Nparticles,
     int num_blocks)
 {
     normalize_weights_kernel<<<num_blocks, threads_per_block>>>(
         d_weights,
         Nparticles,
         d_partial_sums,
         d_CDF,
         d_u,
         d_seed);
 
     checkCuda(cudaGetLastError(), "launch particlefilter_normalize_weights");
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
         "Loaded ParticleFilter config: kernel=normalize_weights IszX=%d IszY=%d Nfr=%d Nparticles=%d num_blocks=%d\n",
         IszX,
         IszY,
         Nfr,
         Nparticles,
         num_blocks);
 
     double* weights =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* partial_sums =
         (double*)malloc(sizeof(double) * (size_t)num_blocks);
 
     double* CDF =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* u =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     int* seed =
         (int*)malloc(sizeof(int) * (size_t)Nparticles);
 
     if (
         weights == NULL ||
         partial_sums == NULL ||
         CDF == NULL ||
         u == NULL ||
         seed == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_host_state(
         weights,
         partial_sums,
         CDF,
         u,
         seed,
         Nparticles,
         num_blocks);
 
     double* d_weights =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_weights");
 
     double* d_partial_sums =
         alloc_device<double>((size_t)num_blocks, "cudaMalloc d_partial_sums");
 
     double* d_CDF =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_CDF");
 
     double* d_u =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_u");
 
     int* d_seed =
         alloc_device<int>((size_t)Nparticles, "cudaMalloc d_seed");
 
     reset_device_state(
         d_weights,
         d_partial_sums,
         d_CDF,
         d_u,
         d_seed,
         weights,
         partial_sums,
         CDF,
         u,
         seed,
         Nparticles,
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
             launch_normalize_once(
                 d_weights,
                 d_partial_sums,
                 d_CDF,
                 d_u,
                 d_seed,
                 Nparticles,
                 num_blocks);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because normalize_weights_kernel mutates:
      * weights, CDF, u, and seed.
      */
     reset_device_state(
         d_weights,
         d_partial_sums,
         d_CDF,
         d_u,
         d_seed,
         weights,
         partial_sums,
         CDF,
         u,
         seed,
         Nparticles,
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
             launch_normalize_once(
                 d_weights,
                 d_partial_sums,
                 d_CDF,
                 d_u,
                 d_seed,
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
         "particlefilter_normalize_weights",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<double>(d_weights, "cudaFree d_weights");
     dealloc_device<double>(d_partial_sums, "cudaFree d_partial_sums");
     dealloc_device<double>(d_CDF, "cudaFree d_CDF");
     dealloc_device<double>(d_u, "cudaFree d_u");
     dealloc_device<int>(d_seed, "cudaFree d_seed");
 
     free(weights);
     free(partial_sums);
     free(CDF);
     free(u);
     free(seed);
 
     return EXIT_SUCCESS;
 }