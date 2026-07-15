/**
 * 01_myocyte_solver_2_repeat.cu
 *
 * Isolated measurement for Rodinia Myocyte solver_2.
 *
 * Measures a single solver_2 launch repeatedly.
 *
 * Setup:
 * 1) allocate x/y/params/temp buffers once
 * 2) read y.txt and params.txt once
 * 3) copy initial x/y/params to GPU
 * 4) warm up with repeated solver_2 launches
 * 5) reset x/y/params/temp buffers
 * 6) measure repeated solver_2 launches
 *    - CUDA events for runtime
 *    - NVML total energy for energy
 * 7) print RESULT lines
 * 8) exit
 *
 * Usage:
 *   ./01_myocyte_solver_2_repeat.exe <xmax> <workload>
 *
 * Example:
 *   ./01_myocyte_solver_2_repeat.exe 100 256
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <math.h>
 #include <string.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include "measurement_common.h"
 
 /*
  * Original Myocyte definitions and helpers.
  *
  * These files define:
  * - fp
  * - EQUATIONS
  * - PARAMETERS
  * - NUMBER_THREADS
  * - read(...)
  * - kernel_ecc_2
  * - kernel_cam_2
  * - kernel_fin_2
  * - kernel_2
  * - embedded_fehlberg_7_8_2
  * - solver_2
  */
 #include "define.c"
 #include "file.c"
 
 #include "kernel_fin_2.cu"
 #include "kernel_ecc_2.cu"
 #include "kernel_cam_2.cu"
 #include "kernel_2.cu"
 #include "embedded_fehlberg_7_8_2.cu"
 #include "solver_2.cu"
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
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
     fprintf(stderr, "Usage: %s <xmax> <workload>\n", program);
     fprintf(stderr, "Example: %s 100 256\n", program);
     exit(EXIT_FAILURE);
 }
 
 static void initialize_host_inputs(
     int xmax,
     int workload,
     fp* x,
     fp* y,
     fp* params)
 {
     int i;
     int pointer;
 
     /*
      * x layout:
      *   x[workload][xmax + 1]
      */
     for (i = 0; i < workload; i++)
     {
         pointer = i * (xmax + 1);
         x[pointer] = 0;
     }
 
     /*
      * y layout:
      *   y[workload][xmax + 1][EQUATIONS]
      *
      * Original Rodinia reads the same y.txt for every workload instance.
      */
     for (i = 0; i < workload; i++)
     {
         pointer = i * ((xmax + 1) * EQUATIONS);
         read(
             "../../data/myocyte/y.txt",
             &y[pointer],
             EQUATIONS,
             1,
             0);
     }
 
     /*
      * params layout:
      *   params[workload][PARAMETERS]
      *
      * Original Rodinia reads the same params.txt for every workload instance.
      */
     for (i = 0; i < workload; i++)
     {
         pointer = i * PARAMETERS;
         read(
             "../../data/myocyte/params.txt",
             &params[pointer],
             PARAMETERS,
             1,
             0);
     }
 }
 
 static void reset_device_state(
     int xmax,
     int workload,
     fp* d_x,
     fp* d_y,
     fp* d_params,
     fp* d_com,
     fp* d_err,
     fp* d_scale,
     fp* d_yy,
     fp* d_initvalu_temp,
     fp* d_finavalu_temp,
     const fp* x,
     const fp* y,
     const fp* params,
     size_t x_count,
     size_t y_count,
     size_t params_count,
     size_t com_count,
     size_t err_count,
     size_t scale_count,
     size_t yy_count,
     size_t initvalu_temp_count,
     size_t finavalu_temp_count)
 {
     (void)xmax;
     (void)workload;
 
     upload_device<fp>(d_x, x, x_count, "copy d_x");
     upload_device<fp>(d_y, y, y_count, "copy d_y");
     upload_device<fp>(d_params, params, params_count, "copy d_params");
 
     checkCuda(cudaMemset(d_com, 0, sizeof(fp) * com_count), "memset d_com");
     checkCuda(cudaMemset(d_err, 0, sizeof(fp) * err_count), "memset d_err");
     checkCuda(cudaMemset(d_scale, 0, sizeof(fp) * scale_count), "memset d_scale");
     checkCuda(cudaMemset(d_yy, 0, sizeof(fp) * yy_count), "memset d_yy");
     checkCuda(cudaMemset(d_initvalu_temp, 0, sizeof(fp) * initvalu_temp_count), "memset d_initvalu_temp");
     checkCuda(cudaMemset(d_finavalu_temp, 0, sizeof(fp) * finavalu_temp_count), "memset d_finavalu_temp");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_solver_2_once(
     int xmax,
     int workload,
     fp* d_x,
     fp* d_y,
     fp* d_params,
     fp* d_com,
     fp* d_err,
     fp* d_scale,
     fp* d_yy,
     fp* d_initvalu_temp,
     fp* d_finavalu_temp,
     dim3 blocks,
     dim3 threads)
 {
     solver_2<<<blocks, threads>>>(
         workload,
         xmax,
         d_x,
         d_y,
         d_params,
         d_com,
         d_err,
         d_scale,
         d_yy,
         d_initvalu_temp,
         d_finavalu_temp);
 
     checkCuda(cudaGetLastError(), "launch myocyte_solver_2");
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 3)
     {
         usage(argv[0]);
     }
 
     int xmax = atoi(argv[1]);
     int workload = atoi(argv[2]);
 
     if (xmax <= 0 || workload <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     printf("Loaded Myocyte config: xmax=%d workload=%d\n", xmax, workload);
     printf("WG size of kernel = %d\n", NUMBER_THREADS);
 
     /*
      * Match original memory guard from work_2.cu.
      */
     long long memory_check =
         (long long)workload * (long long)(xmax + 1) * (long long)EQUATIONS * 4LL;
 
     if (memory_check > 1000000000LL)
     {
         fatal("trying to allocate more than 1.0GB of y memory");
     }
 
     size_t y_count = (size_t)workload * (size_t)(xmax + 1) * (size_t)EQUATIONS;
     size_t x_count = (size_t)workload * (size_t)(xmax + 1);
     size_t params_count = (size_t)workload * (size_t)PARAMETERS;
 
     size_t com_count = (size_t)workload * 3u;
     size_t err_count = (size_t)workload * (size_t)EQUATIONS;
     size_t scale_count = (size_t)workload * (size_t)EQUATIONS;
     size_t yy_count = (size_t)workload * (size_t)EQUATIONS;
     size_t initvalu_temp_count = (size_t)workload * (size_t)EQUATIONS;
     size_t finavalu_temp_count = (size_t)workload * 13u * (size_t)EQUATIONS;
 
     fp* y = (fp*)malloc(sizeof(fp) * y_count);
     fp* x = (fp*)malloc(sizeof(fp) * x_count);
     fp* params = (fp*)malloc(sizeof(fp) * params_count);
 
     if (y == NULL || x == NULL || params == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     memset(y, 0, sizeof(fp) * y_count);
     memset(x, 0, sizeof(fp) * x_count);
     memset(params, 0, sizeof(fp) * params_count);
 
     initialize_host_inputs(
         xmax,
         workload,
         x,
         y,
         params);
 
     fp* d_y = alloc_device<fp>(y_count, "cudaMalloc d_y");
     fp* d_x = alloc_device<fp>(x_count, "cudaMalloc d_x");
     fp* d_params = alloc_device<fp>(params_count, "cudaMalloc d_params");
 
     fp* d_com = alloc_device<fp>(com_count, "cudaMalloc d_com");
     fp* d_err = alloc_device<fp>(err_count, "cudaMalloc d_err");
     fp* d_scale = alloc_device<fp>(scale_count, "cudaMalloc d_scale");
     fp* d_yy = alloc_device<fp>(yy_count, "cudaMalloc d_yy");
     fp* d_initvalu_temp = alloc_device<fp>(initvalu_temp_count, "cudaMalloc d_initvalu_temp");
     fp* d_finavalu_temp = alloc_device<fp>(finavalu_temp_count, "cudaMalloc d_finavalu_temp");
 
     reset_device_state(
         xmax,
         workload,
         d_x,
         d_y,
         d_params,
         d_com,
         d_err,
         d_scale,
         d_yy,
         d_initvalu_temp,
         d_finavalu_temp,
         x,
         y,
         params,
         x_count,
         y_count,
         params_count,
         com_count,
         err_count,
         scale_count,
         yy_count,
         initvalu_temp_count,
         finavalu_temp_count);
 
     dim3 threads;
     dim3 blocks;
 
     if (workload == 1)
     {
         threads.x = 32;
         threads.y = 1;
         threads.z = 1;
 
         blocks.x = 4;
         blocks.y = 1;
         blocks.z = 1;
     }
     else
     {
         threads.x = NUMBER_THREADS;
         threads.y = 1;
         threads.z = 1;
 
         int blocks_x = workload / threads.x;
         if (workload % threads.x != 0)
         {
             blocks_x++;
         }
 
         blocks.x = blocks_x;
         blocks.y = 1;
         blocks.z = 1;
     }
 
     printf(
         "threads=[%u,%u,%u] blocks=[%u,%u,%u]\n",
         threads.x,
         threads.y,
         threads.z,
         blocks.x,
         blocks.y,
         blocks.z);
 
     nvmlDevice_t nvml_device;
     unsigned long long energy_start_mj = 0;
     unsigned long long energy_end_mj = 0;
 
     checkNvml(nvmlInit(), "nvmlInit");
     checkNvml(
         nvmlDeviceGetHandleByIndex(GPU_DEVICE, &nvml_device),
         "nvmlDeviceGetHandleByIndex");
 
     /*
      * Warmup.
      */
     int warmup_launches = 0;
     {
         double warmup_start = now_seconds();
 
         while ((now_seconds() - warmup_start) < WARMUP_SECONDS)
         {
             launch_solver_2_once(
                 xmax,
                 workload,
                 d_x,
                 d_y,
                 d_params,
                 d_com,
                 d_err,
                 d_scale,
                 d_yy,
                 d_initvalu_temp,
                 d_finavalu_temp,
                 blocks,
                 threads);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     /*
      * Reset all device state before measurement because solver_2 modifies x/y
      * and temporary arrays.
      */
     reset_device_state(
         xmax,
         workload,
         d_x,
         d_y,
         d_params,
         d_com,
         d_err,
         d_scale,
         d_yy,
         d_initvalu_temp,
         d_finavalu_temp,
         x,
         y,
         params,
         x_count,
         y_count,
         params_count,
         com_count,
         err_count,
         scale_count,
         yy_count,
         initvalu_temp_count,
         finavalu_temp_count);
 
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
             launch_solver_2_once(
                 xmax,
                 workload,
                 d_x,
                 d_y,
                 d_params,
                 d_com,
                 d_err,
                 d_scale,
                 d_yy,
                 d_initvalu_temp,
                 d_finavalu_temp,
                 blocks,
                 threads);
 
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
         "myocyte_solver_2",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<fp>(d_y, "cudaFree d_y");
     dealloc_device<fp>(d_x, "cudaFree d_x");
     dealloc_device<fp>(d_params, "cudaFree d_params");
     dealloc_device<fp>(d_com, "cudaFree d_com");
     dealloc_device<fp>(d_err, "cudaFree d_err");
     dealloc_device<fp>(d_scale, "cudaFree d_scale");
     dealloc_device<fp>(d_yy, "cudaFree d_yy");
     dealloc_device<fp>(d_initvalu_temp, "cudaFree d_initvalu_temp");
     dealloc_device<fp>(d_finavalu_temp, "cudaFree d_finavalu_temp");
 
     free(y);
     free(x);
     free(params);
 
     return EXIT_SUCCESS;
 }