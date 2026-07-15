/**
 * 03_srad_reduce_repeat.cu
 *
 * Isolated measurement for Rodinia SRAD v1 reduce kernel.
 *
 * Measures one representative first-stage reduction:
 *   reduce<<<blocks, threads>>>(Ne, Ne, 1, d_sums, d_sums2)
 *
 * Usage:
 *   ./03_srad_reduce_repeat.exe <Nr> <Nc>
 *
 * Example:
 *   ./03_srad_reduce_repeat.exe 2048 2048
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
  * Original SRAD v1 reduce kernel.
  */
 __global__ void reduce(
     long d_Ne,
     int d_no,
     int d_mul,
     fp* d_sums,
     fp* d_sums2)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
     int ei = (bx * NUMBER_THREADS) + tx;
     int nf = NUMBER_THREADS - (gridDim.x * NUMBER_THREADS - d_no);
     int df = 0;
 
     __shared__ fp d_psum[NUMBER_THREADS];
     __shared__ fp d_psum2[NUMBER_THREADS];
 
     int i;
 
     if (ei < d_no)
     {
         d_psum[tx] = d_sums[ei * d_mul];
         d_psum2[tx] = d_sums2[ei * d_mul];
     }
 
     __syncthreads();
 
     if (nf == NUMBER_THREADS)
     {
         for (i = 2; i <= NUMBER_THREADS; i = 2 * i)
         {
             if ((tx + 1) % i == 0)
             {
                 d_psum[tx] = d_psum[tx] + d_psum[tx - i / 2];
                 d_psum2[tx] = d_psum2[tx] + d_psum2[tx - i / 2];
             }
 
             __syncthreads();
         }
 
         if (tx == (NUMBER_THREADS - 1))
         {
             d_sums[bx * d_mul * NUMBER_THREADS] = d_psum[tx];
             d_sums2[bx * d_mul * NUMBER_THREADS] = d_psum2[tx];
         }
     }
     else
     {
         if (bx != (gridDim.x - 1))
         {
             for (i = 2; i <= NUMBER_THREADS; i = 2 * i)
             {
                 if ((tx + 1) % i == 0)
                 {
                     d_psum[tx] = d_psum[tx] + d_psum[tx - i / 2];
                     d_psum2[tx] = d_psum2[tx] + d_psum2[tx - i / 2];
                 }
 
                 __syncthreads();
             }
 
             if (tx == (NUMBER_THREADS - 1))
             {
                 d_sums[bx * d_mul * NUMBER_THREADS] = d_psum[tx];
                 d_sums2[bx * d_mul * NUMBER_THREADS] = d_psum2[tx];
             }
         }
         else
         {
             for (i = 2; i <= NUMBER_THREADS; i = 2 * i)
             {
                 if (nf >= i)
                 {
                     df = i;
                 }
             }
 
             for (i = 2; i <= df; i = 2 * i)
             {
                 if ((tx + 1) % i == 0 && tx < df)
                 {
                     d_psum[tx] = d_psum[tx] + d_psum[tx - i / 2];
                     d_psum2[tx] = d_psum2[tx] + d_psum2[tx - i / 2];
                 }
 
                 __syncthreads();
             }
 
             if (tx == (df - 1))
             {
                 for (i = (bx * NUMBER_THREADS) + df;
                      i < (bx * NUMBER_THREADS) + nf;
                      i++)
                 {
                     d_psum[tx] = d_psum[tx] + d_sums[i];
                     d_psum2[tx] = d_psum2[tx] + d_sums2[i];
                 }
 
                 d_sums[bx * d_mul * NUMBER_THREADS] = d_psum[tx];
                 d_sums2[bx * d_mul * NUMBER_THREADS] = d_psum2[tx];
             }
         }
     }
 
     (void)d_Ne;
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
 
 static void initialize_sums(
     fp* sums,
     fp* sums2,
     long Ne)
 {
     /*
      * Simulate output from prepare.
      */
     for (long i = 0; i < Ne; i++)
     {
         fp value = expf((1.0f + (fp)(i % 255)) / 255.0f);
         sums[i] = value;
         sums2[i] = value * value;
     }
 }
 
 static void reset_device_state(
     fp* d_sums,
     fp* d_sums2,
     const fp* sums,
     const fp* sums2,
     long Ne)
 {
     upload_device<fp>(
         d_sums,
         sums,
         (size_t)Ne,
         "copy d_sums");
 
     upload_device<fp>(
         d_sums2,
         sums2,
         (size_t)Ne,
         "copy d_sums2");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_reduce_once(
     fp* d_sums,
     fp* d_sums2,
     long Ne,
     int blocks_x)
 {
     dim3 threads(NUMBER_THREADS);
     dim3 blocks(blocks_x);
 
     /*
      * Representative first-stage reduction from the original loop:
      * no = Ne, mul = 1.
      */
     reduce<<<blocks, threads>>>(
         Ne,
         (int)Ne,
         1,
         d_sums,
         d_sums2);
 
     checkCuda(cudaGetLastError(), "launch srad_reduce");
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
 
     if (Ne > INT32_MAX)
     {
         fatal("Ne too large for original SRAD reduce int d_no");
     }
 
     int blocks_x = (int)(Ne / NUMBER_THREADS);
     if (Ne % NUMBER_THREADS != 0)
     {
         blocks_x++;
     }
 
     printf(
         "Loaded SRAD v1 config: kernel=reduce Nr=%d Nc=%d Ne=%ld blocks=%d threads=%d reduce_no=%ld reduce_mul=1\n",
         Nr,
         Nc,
         Ne,
         blocks_x,
         NUMBER_THREADS,
         Ne);
 
     fp* sums = (fp*)malloc(sizeof(fp) * (size_t)Ne);
     fp* sums2 = (fp*)malloc(sizeof(fp) * (size_t)Ne);
 
     if (sums == NULL || sums2 == NULL)
     {
         fatal("unable to allocate host sums");
     }
 
     initialize_sums(sums, sums2, Ne);
 
     fp* d_sums =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_sums");
 
     fp* d_sums2 =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_sums2");
 
     reset_device_state(
         d_sums,
         d_sums2,
         sums,
         sums2,
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
             launch_reduce_once(
                 d_sums,
                 d_sums2,
                 Ne,
                 blocks_x);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because reduce mutates d_sums and d_sums2.
      */
     reset_device_state(
         d_sums,
         d_sums2,
         sums,
         sums2,
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
             launch_reduce_once(
                 d_sums,
                 d_sums2,
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
         "srad_reduce",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<fp>(d_sums, "cudaFree d_sums");
     dealloc_device<fp>(d_sums2, "cudaFree d_sums2");
 
     free(sums);
     free(sums2);
 
     return EXIT_SUCCESS;
 }