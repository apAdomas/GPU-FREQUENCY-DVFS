/**
 * 06_srad_compress_repeat.cu
 *
 * Isolated measurement for Rodinia SRAD v1 compress kernel.
 *
 * Measures:
 *   compress<<<blocks, threads>>>(Ne, d_I)
 *
 * Usage:
 *   ./06_srad_compress_repeat.exe <Nr> <Nc>
 *
 * Example:
 *   ./06_srad_compress_repeat.exe 2048 2048
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <math.h>
 
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
 
 #ifndef NUMBER_THREADS
 #define NUMBER_THREADS 512
 #endif
 
 typedef float fp;
 
 /*
  * Original SRAD v1 compress kernel.
  */
 __global__ void compress(
     long d_Ne,
     fp* d_I)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
     int ei = (bx * NUMBER_THREADS) + tx;
 
     if (ei < d_Ne)
     {
         d_I[ei] = logf(d_I[ei]) * 255.0f;
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
 
 static void initialize_image(
     fp* image,
     long Ne)
 {
     /*
      * Simulate image after SRAD update.
      * Must be positive because compress applies log().
      */
     for (long i = 0; i < Ne; i++)
     {
         image[i] = 1.0f + ((fp)(i % 255) / 255.0f);
     }
 }
 
 static void reset_device_state(
     fp* d_I,
     const fp* image,
     long Ne)
 {
     upload_device<fp>(
         d_I,
         image,
         (size_t)Ne,
         "copy d_I");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_compress_once(
     fp* d_I,
     long Ne,
     int blocks_x)
 {
     dim3 threads(NUMBER_THREADS);
     dim3 blocks(blocks_x);
 
     compress<<<blocks, threads>>>(Ne, d_I);
 
     checkCuda(cudaGetLastError(), "launch srad_compress");
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
     fprintf(stderr, "Usage: %s <Nr> <Nc>\n", program);
     fprintf(stderr, "Example: %s 2048 2048\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 3)
     {
         usage(argv[0]);
     }
 
     int Nr = atoi(argv[1]);
     int Nc = atoi(argv[2]);
 
     if (Nr <= 0 || Nc <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     long Ne = (long)Nr * (long)Nc;
 
     int blocks_x = (int)(Ne / NUMBER_THREADS);
     if (Ne % NUMBER_THREADS != 0)
     {
         blocks_x++;
     }
 
     printf(
         "Loaded SRAD v1 config: kernel=compress Nr=%d Nc=%d Ne=%ld blocks=%d threads=%d\n",
         Nr,
         Nc,
         Ne,
         blocks_x,
         NUMBER_THREADS);
 
     fp* image = (fp*)malloc(sizeof(fp) * (size_t)Ne);
 
     if (image == NULL)
     {
         fatal("unable to allocate host image");
     }
 
     initialize_image(image, Ne);
 
     fp* d_I =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_I");
 
     reset_device_state(
         d_I,
         image,
         Ne);
 
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
             launch_compress_once(
                 d_I,
                 Ne,
                 blocks_x);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because compress mutates d_I.
      */
     reset_device_state(
         d_I,
         image,
         Ne);
 
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
             launch_compress_once(
                 d_I,
                 Ne,
                 blocks_x);
 
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
         "srad_compress",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<fp>(d_I, "cudaFree d_I");
 
     free(image);
 
     return EXIT_SUCCESS;
 }