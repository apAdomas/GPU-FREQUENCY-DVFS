/**
 * 01_hotspot3d_hotspotOpt1_repeat.cu
 *
 * Isolated measurement for Rodinia Hotspot3D hotspotOpt1.
 *
 * Measures a single hotspotOpt1 launch repeatedly.
 *
 * Usage:
 *   ./01_hotspot3d_hotspotOpt1_repeat.exe <rows/cols> <layers> <iterations> <powerFile> <tempFile> [outputFile]
 *
 * Example:
 *   ./01_hotspot3d_hotspotOpt1_repeat.exe 512 8 60 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <math.h>
 #include <string.h>
 #include <sys/time.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include "measurement_common.h"
 
 #define STR_SIZE 256
 
 #define MAX_PD        (3.0e6f)
 #define PRECISION     0.001f
 #define SPEC_HEAT_SI  1.75e6f
 #define K_SI          100.0f
 #define FACTOR_CHIP   0.5f
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 float t_chip = 0.0005f;
 float chip_height = 0.016f;
 float chip_width = 0.016f;
 float amb_temp = 80.0f;
 
 static void fatal(const char* s)
 {
     fprintf(stderr, "ERROR: %s\n", s);
     exit(EXIT_FAILURE);
 }
 
 static void readinput(
     float* vect,
     int grid_rows,
     int grid_cols,
     int layers,
     const char* file_name)
 {
     FILE* fp = fopen(file_name, "r");
 
     if (fp == NULL)
     {
         fprintf(stderr, "ERROR: could not open input file: %s\n", file_name);
         exit(EXIT_FAILURE);
     }
 
     char str[STR_SIZE];
     float val;
 
     for (int i = 0; i < grid_rows; i++)
     {
         for (int j = 0; j < grid_cols; j++)
         {
             for (int k = 0; k < layers; k++)
             {
                 if (fgets(str, STR_SIZE, fp) == NULL)
                 {
                     fatal("not enough lines in input file");
                 }
 
                 if (sscanf(str, "%f", &val) != 1)
                 {
                     fatal("invalid input file format");
                 }
 
                 vect[i * grid_cols + j + k * grid_rows * grid_cols] = val;
             }
         }
     }
 
     fclose(fp);
 }
 
 template <typename T>
 T* alloc_device(int N, const char* name)
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
 void upload_device(T* dst, const T* src, int N, const char* name)
 {
     checkCuda(
         cudaMemcpy(
             (void*)dst,
             (const void*)src,
             sizeof(T) * N,
             cudaMemcpyHostToDevice),
         name);
 }
 
 /*
  * Target kernel from Rodinia Hotspot3D opt1.cu.
  */
 __global__ void hotspotOpt1(
     float* p,
     float* tIn,
     float* tOut,
     float sdc,
     int nx,
     int ny,
     int nz,
     float ce,
     float cw,
     float cn,
     float cs,
     float ct,
     float cb,
     float cc)
 {
     float amb_temp_local = 80.0f;
 
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     int j = blockDim.y * blockIdx.y + threadIdx.y;
 
     if (i >= nx || j >= ny)
     {
         return;
     }
 
     int c = i + j * nx;
     int xy = nx * ny;
 
     int W = (i == 0)      ? c : c - 1;
     int E = (i == nx - 1) ? c : c + 1;
     int N = (j == 0)      ? c : c - nx;
     int S = (j == ny - 1) ? c : c + nx;
 
     float temp1;
     float temp2;
     float temp3;
 
     temp1 = tIn[c];
     temp2 = tIn[c];
 
     if (nz > 1)
     {
         temp3 = tIn[c + xy];
     }
     else
     {
         temp3 = tIn[c];
     }
 
     tOut[c] =
         cc * temp2 +
         cw * tIn[W] +
         ce * tIn[E] +
         cs * tIn[S] +
         cn * tIn[N] +
         cb * temp1 +
         ct * temp3 +
         sdc * p[c] +
         ct * amb_temp_local;
 
     c += xy;
     W += xy;
     E += xy;
     N += xy;
     S += xy;
 
     for (int k = 1; k < nz - 1; ++k)
     {
         temp1 = temp2;
         temp2 = temp3;
         temp3 = tIn[c + xy];
 
         tOut[c] =
             cc * temp2 +
             cw * tIn[W] +
             ce * tIn[E] +
             cs * tIn[S] +
             cn * tIn[N] +
             cb * temp1 +
             ct * temp3 +
             sdc * p[c] +
             ct * amb_temp_local;
 
         c += xy;
         W += xy;
         E += xy;
         N += xy;
         S += xy;
     }
 
     if (nz > 1)
     {
         temp1 = temp2;
         temp2 = temp3;
 
         tOut[c] =
             cc * temp2 +
             cw * tIn[W] +
             ce * tIn[E] +
             cs * tIn[S] +
             cn * tIn[N] +
             cb * temp1 +
             ct * temp3 +
             sdc * p[c] +
             ct * amb_temp_local;
     }
 }
 
 static void launch_hotspotOpt1_once(
     float* p_d,
     float* tIn_d,
     float* tOut_d,
     float stepDivCap,
     int nx,
     int ny,
     int nz,
     float ce,
     float cw,
     float cn,
     float cs,
     float ct,
     float cb,
     float cc,
     dim3 grid_dim,
     dim3 block_dim)
 {
     hotspotOpt1<<<grid_dim, block_dim>>>(
         p_d,
         tIn_d,
         tOut_d,
         stepDivCap,
         nx,
         ny,
         nz,
         ce,
         cw,
         cn,
         cs,
         ct,
         cb,
         cc);
 
     checkCuda(cudaGetLastError(), "launch hotspot3d_hotspotOpt1");
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
     fprintf(stderr,
         "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> [outputFile]\n",
         program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 6 && argc != 7)
     {
         usage(argv[0]);
     }
 
     int nx = atoi(argv[1]);
     int ny = atoi(argv[1]);
     int nz = atoi(argv[2]);
     int iterations = atoi(argv[3]);
 
     const char* power_file = argv[4];
     const char* temp_file = argv[5];
 
     if (nx <= 0 || ny <= 0 || nz <= 0 || iterations <= 0)
     {
         usage(argv[0]);
     }
 
     printf("WG size of kernel = 64 x 4 x 1\n");
     printf(
         "Loaded Hotspot3D config: grid=%d x %d layers=%d iterations=%d\n",
         nx,
         ny,
         nz,
         iterations);
 
     GPU_argv_init_measurement();
 
     int size = nx * ny * nz;
 
     float* h_power = (float*)calloc(size, sizeof(float));
     float* h_temp = (float*)calloc(size, sizeof(float));
 
     if (h_power == NULL || h_temp == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     readinput(h_power, ny, nx, nz, power_file);
     readinput(h_temp, ny, nx, nz, temp_file);
 
     float dx = chip_height / ny;
     float dy = chip_width / nx;
     float dz = t_chip / nz;
 
     float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
     float Rx = dy / (2.0f * K_SI * t_chip * dx);
     float Ry = dx / (2.0f * K_SI * t_chip * dy);
     float Rz = dz / (K_SI * dx * dy);
 
     float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
     float dt = PRECISION / max_slope;
 
     float stepDivCap = dt / Cap;
 
     float ce;
     float cw;
     float cn;
     float cs;
     float ct;
     float cb;
     float cc;
 
     ce = cw = stepDivCap / Rx;
     cn = cs = stepDivCap / Ry;
     ct = cb = stepDivCap / Rz;
     cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);
 
     float* p_d = alloc_device<float>(size, "cudaMalloc p_d");
     float* tIn_d = alloc_device<float>(size, "cudaMalloc tIn_d");
     float* tOut_d = alloc_device<float>(size, "cudaMalloc tOut_d");
 
     upload_device<float>(p_d, h_power, size, "copy p_d");
     upload_device<float>(tIn_d, h_temp, size, "copy tIn_d");
 
     checkCuda(
         cudaMemset(tOut_d, 0, sizeof(float) * size),
         "memset tOut_d");
 
     checkCuda(
         cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1),
         "cudaFuncSetCacheConfig hotspotOpt1");
 
     dim3 block_dim(64, 4, 1);
 
     /*
      * Original Rodinia uses:
      *   dim3 grid_dim(nx / 64, ny / 4, 1);
      *
      * This version uses ceil division so it does not silently skip cells
      * if the dimensions are not exact multiples.
      */
     dim3 grid_dim(
         (nx + block_dim.x - 1) / block_dim.x,
         (ny + block_dim.y - 1) / block_dim.y,
         1);
 
     printf(
         "block_dim=[%u,%u,%u] grid_dim=[%u,%u,%u]\n",
         block_dim.x,
         block_dim.y,
         block_dim.z,
         grid_dim.x,
         grid_dim.y,
         grid_dim.z);
 
     checkCuda(cudaDeviceSynchronize(), "sync after setup");
 
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
             launch_hotspotOpt1_once(
                 p_d,
                 tIn_d,
                 tOut_d,
                 stepDivCap,
                 nx,
                 ny,
                 nz,
                 ce,
                 cw,
                 cn,
                 cs,
                 ct,
                 cb,
                 cc,
                 grid_dim,
                 block_dim);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(tOut_d, 0, sizeof(float) * size),
         "reset tOut_d");
 
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
             launch_hotspotOpt1_once(
                 p_d,
                 tIn_d,
                 tOut_d,
                 stepDivCap,
                 nx,
                 ny,
                 nz,
                 ce,
                 cw,
                 cn,
                 cs,
                 ct,
                 cb,
                 cc,
                 grid_dim,
                 block_dim);
 
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
         "hotspot3d_hotspotOpt1",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<float>(p_d, "cudaFree p_d");
     dealloc_device<float>(tIn_d, "cudaFree tIn_d");
     dealloc_device<float>(tOut_d, "cudaFree tOut_d");
 
     free(h_power);
     free(h_temp);
 
     return EXIT_SUCCESS;
 }