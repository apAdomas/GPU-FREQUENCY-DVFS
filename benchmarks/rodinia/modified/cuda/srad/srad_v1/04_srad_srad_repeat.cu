/**
 * 04_srad_srad_repeat.cu
 *
 * Isolated measurement for Rodinia SRAD v1 srad kernel.
 *
 * Measures:
 *   srad<<<blocks, threads>>>(lambda, Nr, Nc, Ne, d_iN, d_iS, d_jE, d_jW,
 *                             d_dN, d_dS, d_dE, d_dW, q0sqr, d_c, d_I)
 *
 * Usage:
 *   ./04_srad_srad_repeat.exe <Nr> <Nc> <lambda>
 *
 * Example:
 *   ./04_srad_srad_repeat.exe 2048 2048 0.5
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
 
 #define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
 
 /*
  * Original SRAD v1 srad kernel.
  */
 __global__ void srad(
     fp d_lambda,
     int d_Nr,
     int d_Nc,
     long d_Ne,
     int* d_iN,
     int* d_iS,
     int* d_jE,
     int* d_jW,
     fp* d_dN,
     fp* d_dS,
     fp* d_dE,
     fp* d_dW,
     fp d_q0sqr,
     fp* d_c,
     fp* d_I)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
     int ei = bx * NUMBER_THREADS + tx;
 
     int row;
     int col;
 
     fp d_Jc;
     fp d_dN_loc;
     fp d_dS_loc;
     fp d_dW_loc;
     fp d_dE_loc;
     fp d_c_loc;
     fp d_G2;
     fp d_L;
     fp d_num;
     fp d_den;
     fp d_qsqr;
 
     row = (ei + 1) % d_Nr - 1;
     col = (ei + 1) / d_Nr + 1 - 1;
 
     if ((ei + 1) % d_Nr == 0)
     {
         row = d_Nr - 1;
         col = col - 1;
     }
 
     if (ei < d_Ne)
     {
         d_Jc = d_I[ei];
 
         d_dN_loc = d_I[d_iN[row] + d_Nr * col] - d_Jc;
         d_dS_loc = d_I[d_iS[row] + d_Nr * col] - d_Jc;
         d_dW_loc = d_I[row + d_Nr * d_jW[col]] - d_Jc;
         d_dE_loc = d_I[row + d_Nr * d_jE[col]] - d_Jc;
 
         d_G2 =
             (d_dN_loc * d_dN_loc +
              d_dS_loc * d_dS_loc +
              d_dW_loc * d_dW_loc +
              d_dE_loc * d_dE_loc) /
             (d_Jc * d_Jc);
 
         d_L =
             (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) /
             d_Jc;
 
         d_num = (0.5f * d_G2) - ((1.0f / 16.0f) * (d_L * d_L));
         d_den = 1.0f + (0.25f * d_L);
         d_qsqr = d_num / (d_den * d_den);
 
         d_den =
             (d_qsqr - d_q0sqr) /
             (d_q0sqr * (1.0f + d_q0sqr));
 
         d_c_loc = 1.0f / (1.0f + d_den);
 
         if (d_c_loc < 0.0f)
         {
             d_c_loc = 0.0f;
         }
         else if (d_c_loc > 1.0f)
         {
             d_c_loc = 1.0f;
         }
 
         d_dN[ei] = d_dN_loc;
         d_dS[ei] = d_dS_loc;
         d_dW[ei] = d_dW_loc;
         d_dE[ei] = d_dE_loc;
         d_c[ei] = d_c_loc;
     }
 
     (void)d_lambda;
     (void)d_Nc;
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
      * Simulate image after extract.
      * Values are positive to avoid division by zero in srad.
      */
     for (long i = 0; i < Ne; i++)
     {
         image[i] = expf((1.0f + (fp)(i % 255)) / 255.0f);
     }
 }
 
 static void initialize_indices(
     int* iN,
     int* iS,
     int* jE,
     int* jW,
     int Nr,
     int Nc)
 {
     for (int i = 0; i < Nr; i++)
     {
         iN[i] = i - 1;
         iS[i] = i + 1;
     }
 
     for (int j = 0; j < Nc; j++)
     {
         jW[j] = j - 1;
         jE[j] = j + 1;
     }
 
     iN[0] = 0;
     iS[Nr - 1] = Nr - 1;
     jW[0] = 0;
     jE[Nc - 1] = Nc - 1;
 }
 
 static void reset_device_state(
     fp* d_I,
     fp* d_dN,
     fp* d_dS,
     fp* d_dE,
     fp* d_dW,
     fp* d_c,
     const fp* image,
     long Ne)
 {
     upload_device<fp>(
         d_I,
         image,
         (size_t)Ne,
         "copy d_I");
 
     checkCuda(
         cudaMemset(d_dN, 0, sizeof(fp) * (size_t)Ne),
         "memset d_dN");
 
     checkCuda(
         cudaMemset(d_dS, 0, sizeof(fp) * (size_t)Ne),
         "memset d_dS");
 
     checkCuda(
         cudaMemset(d_dE, 0, sizeof(fp) * (size_t)Ne),
         "memset d_dE");
 
     checkCuda(
         cudaMemset(d_dW, 0, sizeof(fp) * (size_t)Ne),
         "memset d_dW");
 
     checkCuda(
         cudaMemset(d_c, 0, sizeof(fp) * (size_t)Ne),
         "memset d_c");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_srad_once(
     fp lambda,
     int Nr,
     int Nc,
     long Ne,
     int* d_iN,
     int* d_iS,
     int* d_jE,
     int* d_jW,
     fp* d_dN,
     fp* d_dS,
     fp* d_dE,
     fp* d_dW,
     fp q0sqr,
     fp* d_c,
     fp* d_I,
     int blocks_x)
 {
     dim3 threads(NUMBER_THREADS);
     dim3 blocks(blocks_x);
 
     srad<<<blocks, threads>>>(
         lambda,
         Nr,
         Nc,
         Ne,
         d_iN,
         d_iS,
         d_jE,
         d_jW,
         d_dN,
         d_dS,
         d_dE,
         d_dW,
         q0sqr,
         d_c,
         d_I);
 
     checkCuda(cudaGetLastError(), "launch srad_srad");
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
     fprintf(stderr, "Usage: %s <Nr> <Nc> <lambda>\n", program);
     fprintf(stderr, "Example: %s 2048 2048 0.5\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 4)
     {
         usage(argv[0]);
     }
 
     int Nr = atoi(argv[1]);
     int Nc = atoi(argv[2]);
     fp lambda = (fp)atof(argv[3]);
 
     if (Nr <= 0 || Nc <= 0 || lambda <= 0.0f)
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
 
     /*
      * Fixed representative q0sqr.
      * In the full app this is computed from prepare + reduce.
      */
     fp q0sqr = 0.5f;
 
     printf(
         "Loaded SRAD v1 config: kernel=srad Nr=%d Nc=%d Ne=%ld lambda=%.6f q0sqr=%.6f blocks=%d threads=%d\n",
         Nr,
         Nc,
         Ne,
         lambda,
         q0sqr,
         blocks_x,
         NUMBER_THREADS);
 
     fp* image = (fp*)malloc(sizeof(fp) * (size_t)Ne);
 
     int* iN = (int*)malloc(sizeof(int) * (size_t)Nr);
     int* iS = (int*)malloc(sizeof(int) * (size_t)Nr);
     int* jE = (int*)malloc(sizeof(int) * (size_t)Nc);
     int* jW = (int*)malloc(sizeof(int) * (size_t)Nc);
 
     if (
         image == NULL ||
         iN == NULL ||
         iS == NULL ||
         jE == NULL ||
         jW == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_image(image, Ne);
     initialize_indices(iN, iS, jE, jW, Nr, Nc);
 
     fp* d_I =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_I");
 
     fp* d_dN =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_dN");
 
     fp* d_dS =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_dS");
 
     fp* d_dE =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_dE");
 
     fp* d_dW =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_dW");
 
     fp* d_c =
         alloc_device<fp>((size_t)Ne, "cudaMalloc d_c");
 
     int* d_iN =
         alloc_device<int>((size_t)Nr, "cudaMalloc d_iN");
 
     int* d_iS =
         alloc_device<int>((size_t)Nr, "cudaMalloc d_iS");
 
     int* d_jE =
         alloc_device<int>((size_t)Nc, "cudaMalloc d_jE");
 
     int* d_jW =
         alloc_device<int>((size_t)Nc, "cudaMalloc d_jW");
 
     upload_device<int>(d_iN, iN, (size_t)Nr, "copy d_iN");
     upload_device<int>(d_iS, iS, (size_t)Nr, "copy d_iS");
     upload_device<int>(d_jE, jE, (size_t)Nc, "copy d_jE");
     upload_device<int>(d_jW, jW, (size_t)Nc, "copy d_jW");
 
     reset_device_state(
         d_I,
         d_dN,
         d_dS,
         d_dE,
         d_dW,
         d_c,
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
             launch_srad_once(
                 lambda,
                 Nr,
                 Nc,
                 Ne,
                 d_iN,
                 d_iS,
                 d_jE,
                 d_jW,
                 d_dN,
                 d_dS,
                 d_dE,
                 d_dW,
                 q0sqr,
                 d_c,
                 d_I,
                 blocks_x);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because srad mutates d_dN, d_dS, d_dE, d_dW, and d_c.
      */
     reset_device_state(
         d_I,
         d_dN,
         d_dS,
         d_dE,
         d_dW,
         d_c,
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
             launch_srad_once(
                 lambda,
                 Nr,
                 Nc,
                 Ne,
                 d_iN,
                 d_iS,
                 d_jE,
                 d_jW,
                 d_dN,
                 d_dS,
                 d_dE,
                 d_dW,
                 q0sqr,
                 d_c,
                 d_I,
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
         "srad_srad",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<fp>(d_I, "cudaFree d_I");
     dealloc_device<fp>(d_dN, "cudaFree d_dN");
     dealloc_device<fp>(d_dS, "cudaFree d_dS");
     dealloc_device<fp>(d_dE, "cudaFree d_dE");
     dealloc_device<fp>(d_dW, "cudaFree d_dW");
     dealloc_device<fp>(d_c, "cudaFree d_c");
 
     dealloc_device<int>(d_iN, "cudaFree d_iN");
     dealloc_device<int>(d_iS, "cudaFree d_iS");
     dealloc_device<int>(d_jE, "cudaFree d_jE");
     dealloc_device<int>(d_jW, "cudaFree d_jW");
 
     free(image);
     free(iN);
     free(iS);
     free(jE);
     free(jW);
 
     return EXIT_SUCCESS;
 }