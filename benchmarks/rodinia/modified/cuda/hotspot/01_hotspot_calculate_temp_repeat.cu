/**
 * 01_hotspot_calculate_temp_repeat.cu
 *
 * Isolated measurement for Rodinia Hotspot calculate_temp.
 *
 * Measures a single calculate_temp launch repeatedly.
 *
 * Usage:
 *   ./01_hotspot_calculate_temp_repeat.exe <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> [output_file]
 *
 * Example:
 *   ./01_hotspot_calculate_temp_repeat.exe 512 1 60 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out
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
 
 #ifdef RD_WG_SIZE_0_0
     #define BLOCK_SIZE RD_WG_SIZE_0_0
 #elif defined(RD_WG_SIZE_0)
     #define BLOCK_SIZE RD_WG_SIZE_0
 #elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE RD_WG_SIZE
 #else
     #define BLOCK_SIZE 16
 #endif
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 #define STR_SIZE 256
 
 #define MAX_PD        (3.0e6)
 #define PRECISION     0.001f
 #define SPEC_HEAT_SI  1.75e6f
 #define K_SI          100.0f
 #define FACTOR_CHIP   0.5f
 
 #define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
 #define MIN(a, b) ((a) <= (b) ? (a) : (b))
 
 float t_chip = 0.0005f;
 float chip_height = 0.016f;
 float chip_width = 0.016f;
 
 static void fatal(const char* s)
 {
     fprintf(stderr, "ERROR: %s\n", s);
     exit(EXIT_FAILURE);
 }
 
 static void readinput(float* vect, int grid_rows, int grid_cols, const char* file_name)
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
             if (fgets(str, STR_SIZE, fp) == NULL)
             {
                 fatal("not enough lines in input file");
             }
 
             if (sscanf(str, "%f", &val) != 1)
             {
                 fatal("invalid input file format");
             }
 
             vect[i * grid_cols + j] = val;
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
         cudaMemcpy((void*)dst, (const void*)src, sizeof(T) * N, cudaMemcpyHostToDevice),
         name);
 }
 
 /*
  * Target kernel from Rodinia Hotspot.
  */
 __global__ void calculate_temp(
     int iteration,
     float* power,
     float* temp_src,
     float* temp_dst,
     int grid_cols,
     int grid_rows,
     int border_cols,
     int border_rows,
     float Cap,
     float Rx,
     float Ry,
     float Rz,
     float step,
     float time_elapsed)
 {
     __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
     __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
     __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];
 
     float amb_temp = 80.0f;
     float step_div_Cap;
     float Rx_1;
     float Ry_1;
     float Rz_1;
 
     int bx = blockIdx.x;
     int by = blockIdx.y;
 
     int tx = threadIdx.x;
     int ty = threadIdx.y;
 
     step_div_Cap = step / Cap;
 
     Rx_1 = 1.0f / Rx;
     Ry_1 = 1.0f / Ry;
     Rz_1 = 1.0f / Rz;
 
     int small_block_rows = BLOCK_SIZE - iteration * 2;
     int small_block_cols = BLOCK_SIZE - iteration * 2;
 
     int blkY = small_block_rows * by - border_rows;
     int blkX = small_block_cols * bx - border_cols;
     int blkYmax = blkY + BLOCK_SIZE - 1;
     int blkXmax = blkX + BLOCK_SIZE - 1;
 
     int yidx = blkY + ty;
     int xidx = blkX + tx;
 
     int loadYidx = yidx;
     int loadXidx = xidx;
     int index = grid_cols * loadYidx + loadXidx;
 
     if (IN_RANGE(loadYidx, 0, grid_rows - 1) &&
         IN_RANGE(loadXidx, 0, grid_cols - 1))
     {
         temp_on_cuda[ty][tx] = temp_src[index];
         power_on_cuda[ty][tx] = power[index];
     }
 
     __syncthreads();
 
     int validYmin = (blkY < 0) ? -blkY : 0;
     int validYmax =
         (blkYmax > grid_rows - 1) ?
         BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) :
         BLOCK_SIZE - 1;
 
     int validXmin = (blkX < 0) ? -blkX : 0;
     int validXmax =
         (blkXmax > grid_cols - 1) ?
         BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) :
         BLOCK_SIZE - 1;
 
     int N = ty - 1;
     int S = ty + 1;
     int W = tx - 1;
     int E = tx + 1;
 
     N = (N < validYmin) ? validYmin : N;
     S = (S > validYmax) ? validYmax : S;
     W = (W < validXmin) ? validXmin : W;
     E = (E > validXmax) ? validXmax : E;
 
     bool computed = false;
 
     for (int i = 0; i < iteration; i++)
     {
         computed = false;
 
         if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
             IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
             IN_RANGE(tx, validXmin, validXmax) &&
             IN_RANGE(ty, validYmin, validYmax))
         {
             computed = true;
 
             temp_t[ty][tx] =
                 temp_on_cuda[ty][tx] +
                 step_div_Cap *
                     (power_on_cuda[ty][tx] +
                      (temp_on_cuda[S][tx] +
                       temp_on_cuda[N][tx] -
                       2.0f * temp_on_cuda[ty][tx]) *
                          Ry_1 +
                      (temp_on_cuda[ty][E] +
                       temp_on_cuda[ty][W] -
                       2.0f * temp_on_cuda[ty][tx]) *
                          Rx_1 +
                      (amb_temp - temp_on_cuda[ty][tx]) *
                          Rz_1);
         }
 
         __syncthreads();
 
         if (i == iteration - 1)
         {
             break;
         }
 
         if (computed)
         {
             temp_on_cuda[ty][tx] = temp_t[ty][tx];
         }
 
         __syncthreads();
     }
 
     if (computed)
     {
         temp_dst[index] = temp_t[ty][tx];
     }
 }
 
 static void launch_calculate_temp_once(
     int iteration,
     float* MatrixPower,
     float* MatrixTempSrc,
     float* MatrixTempDst,
     int grid_cols,
     int grid_rows,
     int border_cols,
     int border_rows,
     float Cap,
     float Rx,
     float Ry,
     float Rz,
     float step,
     float time_elapsed,
     int blockCols,
     int blockRows)
 {
     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
     dim3 dimGrid(blockCols, blockRows);
 
     calculate_temp<<<dimGrid, dimBlock>>>(
         iteration,
         MatrixPower,
         MatrixTempSrc,
         MatrixTempDst,
         grid_cols,
         grid_rows,
         border_cols,
         border_rows,
         Cap,
         Rx,
         Ry,
         Rz,
         step,
         time_elapsed);
 
     checkCuda(cudaGetLastError(), "launch hotspot_calculate_temp");
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
         "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> [output_file]\n",
         program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
 
     if (argc != 6 && argc != 7)
     {
         usage(argv[0]);
     }
 
     int grid_rows = atoi(argv[1]);
     int grid_cols = atoi(argv[1]);
     int pyramid_height = atoi(argv[2]);
     int total_iterations = atoi(argv[3]);
     const char* temp_file = argv[4];
     const char* power_file = argv[5];
 
     if (grid_rows <= 0 ||
         grid_cols <= 0 ||
         pyramid_height <= 0 ||
         total_iterations <= 0)
     {
         usage(argv[0]);
     }
 
     int iteration = MIN(pyramid_height, total_iterations);
 
     int borderCols = pyramid_height;
     int borderRows = pyramid_height;
 
     int smallBlockCol = BLOCK_SIZE - pyramid_height * 2;
     int smallBlockRow = BLOCK_SIZE - pyramid_height * 2;
 
     if (smallBlockCol <= 0 || smallBlockRow <= 0)
     {
         fatal("pyramid_height too large for BLOCK_SIZE");
     }
 
     int blockCols = grid_cols / smallBlockCol +
                     ((grid_cols % smallBlockCol == 0) ? 0 : 1);
 
     int blockRows = grid_rows / smallBlockRow +
                     ((grid_rows % smallBlockRow == 0) ? 0 : 1);
 
     int size = grid_rows * grid_cols;
 
     printf("Loaded Hotspot config: grid=%d x %d pyramid_height=%d total_iterations=%d\n",
            grid_rows,
            grid_cols,
            pyramid_height,
            total_iterations);
 
     printf("pyramidHeight: %d\n"
            "gridSize: [%d, %d]\n"
            "border: [%d, %d]\n"
            "blockGrid: [%d, %d]\n"
            "targetBlock: [%d, %d]\n",
            pyramid_height,
            grid_cols,
            grid_rows,
            borderCols,
            borderRows,
            blockCols,
            blockRows,
            smallBlockCol,
            smallBlockRow);
 
     GPU_argv_init_measurement();
 
     float* h_temp = (float*)malloc(sizeof(float) * size);
     float* h_power = (float*)malloc(sizeof(float) * size);
 
     if (h_temp == NULL || h_power == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     readinput(h_temp, grid_rows, grid_cols, temp_file);
     readinput(h_power, grid_rows, grid_cols, power_file);
 
     float* MatrixTempSrc = alloc_device<float>(size, "cudaMalloc MatrixTempSrc");
     float* MatrixTempDst = alloc_device<float>(size, "cudaMalloc MatrixTempDst");
     float* MatrixPower = alloc_device<float>(size, "cudaMalloc MatrixPower");
 
     upload_device<float>(MatrixTempSrc, h_temp, size, "copy MatrixTempSrc");
     upload_device<float>(MatrixPower, h_power, size, "copy MatrixPower");
 
     checkCuda(
         cudaMemset(MatrixTempDst, 0, sizeof(float) * size),
         "memset MatrixTempDst");
 
     float grid_height = chip_height / grid_rows;
     float grid_width = chip_width / grid_cols;
 
     float Cap =
         FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
 
     float Rx =
         grid_width / (2.0f * K_SI * t_chip * grid_height);
 
     float Ry =
         grid_height / (2.0f * K_SI * t_chip * grid_width);
 
     float Rz =
         t_chip / (K_SI * grid_height * grid_width);
 
     float max_slope =
         MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
 
     float step = PRECISION / max_slope;
     float time_elapsed = 0.001f;
 
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
             launch_calculate_temp_once(
                 iteration,
                 MatrixPower,
                 MatrixTempSrc,
                 MatrixTempDst,
                 grid_cols,
                 grid_rows,
                 borderCols,
                 borderRows,
                 Cap,
                 Rx,
                 Ry,
                 Rz,
                 step,
                 time_elapsed,
                 blockCols,
                 blockRows);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(MatrixTempDst, 0, sizeof(float) * size),
         "reset MatrixTempDst");
 
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
             launch_calculate_temp_once(
                 iteration,
                 MatrixPower,
                 MatrixTempSrc,
                 MatrixTempDst,
                 grid_cols,
                 grid_rows,
                 borderCols,
                 borderRows,
                 Cap,
                 Rx,
                 Ry,
                 Rz,
                 step,
                 time_elapsed,
                 blockCols,
                 blockRows);
 
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
         "hotspot_calculate_temp",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<float>(MatrixTempSrc, "cudaFree MatrixTempSrc");
     dealloc_device<float>(MatrixTempDst, "cudaFree MatrixTempDst");
     dealloc_device<float>(MatrixPower, "cudaFree MatrixPower");
 
     free(h_temp);
     free(h_power);
 
     return EXIT_SUCCESS;
 }