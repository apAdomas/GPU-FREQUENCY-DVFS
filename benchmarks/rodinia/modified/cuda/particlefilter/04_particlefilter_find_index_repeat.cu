/**
 * 04_particlefilter_find_index_repeat.cu
 *
 * Isolated measurement for Rodinia Particle Filter find_index_kernel.
 *
 * Usage:
 *   ./04_particlefilter_find_index_repeat.exe <IszX> <IszY> <Nfr> <Nparticles>
 *
 * Example:
 *   ./04_particlefilter_find_index_repeat.exe 128 128 10 100000
 *
 * IszX/IszY/Nfr are accepted for runner consistency, but this kernel only
 * depends on Nparticles, arrayX, arrayY, CDF, u, xj, yj, and weights.
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
 
 /*
  * Target kernel from Rodinia particlefilter_float.
  */
 __global__ void find_index_kernel(
     double* arrayX,
     double* arrayY,
     double* CDF,
     double* u,
     double* xj,
     double* yj,
     double* weights,
     int Nparticles)
 {
     int block_id = blockIdx.x;
     int i = blockDim.x * block_id + threadIdx.x;
 
     if (i < Nparticles)
     {
         int index = -1;
 
         for (int x = 0; x < Nparticles; x++)
         {
             if (CDF[x] >= u[i])
             {
                 index = x;
                 break;
             }
         }
 
         if (index == -1)
         {
             index = Nparticles - 1;
         }
 
         xj[i] = arrayX[index];
         yj[i] = arrayY[index];
 
         /*
          * Original code leaves weights untouched here.
          */
         (void)weights;
     }
 
     __syncthreads();
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
     double* arrayX,
     double* arrayY,
     double* CDF,
     double* u,
     double* xj,
     double* yj,
     double* weights,
     int Nparticles)
 {
     /*
      * Simulate state after normalize_weights_kernel:
      * - CDF is monotonic increasing and ends near 1
      * - u contains sorted resampling thresholds
      * - arrayX/arrayY contain particle positions
      */
     for (int i = 0; i < Nparticles; i++)
     {
         double t = (double)i / (double)Nparticles;
 
         arrayX[i] = 64.0 + 0.01 * (double)(i % 1000);
         arrayY[i] = 64.0 - 0.01 * (double)(i % 1000);
 
         weights[i] = 1.0 / (double)Nparticles;
         CDF[i] = (double)(i + 1) / (double)Nparticles;
 
         u[i] = t + 0.5 / (double)Nparticles;
 
         if (u[i] > 1.0)
         {
             u[i] = 1.0;
         }
 
         xj[i] = 0.0;
         yj[i] = 0.0;
     }
 }
 
 static void reset_device_state(
     double* d_arrayX,
     double* d_arrayY,
     double* d_CDF,
     double* d_u,
     double* d_xj,
     double* d_yj,
     double* d_weights,
     const double* arrayX,
     const double* arrayY,
     const double* CDF,
     const double* u,
     const double* xj,
     const double* yj,
     const double* weights,
     int Nparticles)
 {
     upload_device<double>(
         d_arrayX,
         arrayX,
         (size_t)Nparticles,
         "copy d_arrayX");
 
     upload_device<double>(
         d_arrayY,
         arrayY,
         (size_t)Nparticles,
         "copy d_arrayY");
 
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
 
     upload_device<double>(
         d_xj,
         xj,
         (size_t)Nparticles,
         "copy d_xj");
 
     upload_device<double>(
         d_yj,
         yj,
         (size_t)Nparticles,
         "copy d_yj");
 
     upload_device<double>(
         d_weights,
         weights,
         (size_t)Nparticles,
         "copy d_weights");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_find_index_once(
     double* d_arrayX,
     double* d_arrayY,
     double* d_CDF,
     double* d_u,
     double* d_xj,
     double* d_yj,
     double* d_weights,
     int Nparticles,
     int num_blocks)
 {
     find_index_kernel<<<num_blocks, threads_per_block>>>(
         d_arrayX,
         d_arrayY,
         d_CDF,
         d_u,
         d_xj,
         d_yj,
         d_weights,
         Nparticles);
 
     checkCuda(cudaGetLastError(), "launch particlefilter_find_index");
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
         "Loaded ParticleFilter config: kernel=find_index IszX=%d IszY=%d Nfr=%d Nparticles=%d num_blocks=%d\n",
         IszX,
         IszY,
         Nfr,
         Nparticles,
         num_blocks);
 
     double* arrayX =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* arrayY =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* CDF =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* u =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* xj =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* yj =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* weights =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     if (
         arrayX == NULL ||
         arrayY == NULL ||
         CDF == NULL ||
         u == NULL ||
         xj == NULL ||
         yj == NULL ||
         weights == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_host_state(
         arrayX,
         arrayY,
         CDF,
         u,
         xj,
         yj,
         weights,
         Nparticles);
 
     double* d_arrayX =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_arrayX");
 
     double* d_arrayY =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_arrayY");
 
     double* d_CDF =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_CDF");
 
     double* d_u =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_u");
 
     double* d_xj =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_xj");
 
     double* d_yj =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_yj");
 
     double* d_weights =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_weights");
 
     reset_device_state(
         d_arrayX,
         d_arrayY,
         d_CDF,
         d_u,
         d_xj,
         d_yj,
         d_weights,
         arrayX,
         arrayY,
         CDF,
         u,
         xj,
         yj,
         weights,
         Nparticles);
 
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
             launch_find_index_once(
                 d_arrayX,
                 d_arrayY,
                 d_CDF,
                 d_u,
                 d_xj,
                 d_yj,
                 d_weights,
                 Nparticles,
                 num_blocks);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because find_index_kernel mutates xj and yj.
      */
     reset_device_state(
         d_arrayX,
         d_arrayY,
         d_CDF,
         d_u,
         d_xj,
         d_yj,
         d_weights,
         arrayX,
         arrayY,
         CDF,
         u,
         xj,
         yj,
         weights,
         Nparticles);
 
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
             launch_find_index_once(
                 d_arrayX,
                 d_arrayY,
                 d_CDF,
                 d_u,
                 d_xj,
                 d_yj,
                 d_weights,
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
         "particlefilter_find_index",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<double>(d_arrayX, "cudaFree d_arrayX");
     dealloc_device<double>(d_arrayY, "cudaFree d_arrayY");
     dealloc_device<double>(d_CDF, "cudaFree d_CDF");
     dealloc_device<double>(d_u, "cudaFree d_u");
     dealloc_device<double>(d_xj, "cudaFree d_xj");
     dealloc_device<double>(d_yj, "cudaFree d_yj");
     dealloc_device<double>(d_weights, "cudaFree d_weights");
 
     free(arrayX);
     free(arrayY);
     free(CDF);
     free(u);
     free(xj);
     free(yj);
     free(weights);
 
     return EXIT_SUCCESS;
 }