/**
 * 01_pathfinder_dynproc_repeat.cu
 *
 * Isolated measurement for Rodinia Pathfinder dynproc_kernel.
 *
 * Usage:
 *   ./01_pathfinder_dynproc_repeat.exe <cols> <rows> <pyramid_height>
 *
 * Example:
 *   ./01_pathfinder_dynproc_repeat.exe 100000 100 20
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
 
 #define BLOCK_SIZE 256
 #define HALO 1
 #define M_SEED 9
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 #define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
 #define MIN(a, b) ((a) <= (b) ? (a) : (b))
 
 __global__ void dynproc_kernel(
     int iteration,
     int* gpuWall,
     int* gpuSrc,
     int* gpuResults,
     int cols,
     int rows,
     int startStep,
     int border)
 {
     __shared__ int prev[BLOCK_SIZE];
     __shared__ int result[BLOCK_SIZE];
 
     int bx = blockIdx.x;
     int tx = threadIdx.x;
 
     int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
 
     int blkX = small_block_cols * bx - border;
     int blkXmax = blkX + BLOCK_SIZE - 1;
 
     int xidx = blkX + tx;
 
     int validXmin = (blkX < 0) ? -blkX : 0;
     int validXmax =
         (blkXmax > cols - 1)
             ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
             : BLOCK_SIZE - 1;
 
     int W = tx - 1;
     int E = tx + 1;
 
     W = (W < validXmin) ? validXmin : W;
     E = (E > validXmax) ? validXmax : E;
 
     bool isValid = IN_RANGE(tx, validXmin, validXmax);
 
     if (IN_RANGE(xidx, 0, cols - 1))
     {
         prev[tx] = gpuSrc[xidx];
     }
 
     __syncthreads();
 
     bool computed = false;
 
     for (int i = 0; i < iteration; i++)
     {
         computed = false;
 
         if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid)
         {
             computed = true;
 
             int left = prev[W];
             int up = prev[tx];
             int right = prev[E];
 
             int shortest = MIN(left, up);
             shortest = MIN(shortest, right);
 
             int index = cols * (startStep + i) + xidx;
 
             result[tx] = shortest + gpuWall[index];
         }
 
         __syncthreads();
 
         if (i == iteration - 1)
         {
             break;
         }
 
         if (computed)
         {
             prev[tx] = result[tx];
         }
 
         __syncthreads();
     }
 
     if (computed)
     {
         gpuResults[xidx] = result[tx];
     }
 
     (void)rows;
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
 
 static void initialize_data(
     int* data,
     int rows,
     int cols)
 {
     srand(M_SEED);
 
     for (int i = 0; i < rows; i++)
     {
         for (int j = 0; j < cols; j++)
         {
             data[i * cols + j] = rand() % 10;
         }
     }
 }
 
 static void reset_device_state(
     int* d_gpuWall,
     int* d_gpuResult0,
     int* d_gpuResult1,
     const int* data,
     int rows,
     int cols)
 {
     int size = rows * cols;
 
     upload_device<int>(
         d_gpuResult0,
         data,
         (size_t)cols,
         "copy d_gpuResult0");
 
     checkCuda(
         cudaMemset(d_gpuResult1, 0, sizeof(int) * (size_t)cols),
         "memset d_gpuResult1");
 
     upload_device<int>(
         d_gpuWall,
         data + cols,
         (size_t)(size - cols),
         "copy d_gpuWall");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_dynproc_once(
     int* d_gpuWall,
     int* d_gpuSrc,
     int* d_gpuResults,
     int cols,
     int rows,
     int pyramid_height,
     int blockCols,
     int borderCols)
 {
     dim3 dimBlock(BLOCK_SIZE);
     dim3 dimGrid(blockCols);
 
     int iteration = MIN(pyramid_height, rows - 1);
     int startStep = 0;
 
     dynproc_kernel<<<dimGrid, dimBlock>>>(
         iteration,
         d_gpuWall,
         d_gpuSrc,
         d_gpuResults,
         cols,
         rows,
         startStep,
         borderCols);
 
     checkCuda(cudaGetLastError(), "launch pathfinder_dynproc");
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
     fprintf(stderr, "Usage: %s <cols> <rows> <pyramid_height>\n", program);
     fprintf(stderr, "Example: %s 100000 100 20\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 4)
     {
         usage(argv[0]);
     }
 
     int cols = atoi(argv[1]);
     int rows = atoi(argv[2]);
     int pyramid_height = atoi(argv[3]);
 
     if (cols <= 0 || rows <= 1 || pyramid_height <= 0)
     {
         usage(argv[0]);
     }
 
     if ((BLOCK_SIZE - pyramid_height * HALO * 2) <= 0)
     {
         fatal("pyramid_height too large for BLOCK_SIZE");
     }
 
     GPU_argv_init_measurement();
 
     int borderCols = pyramid_height * HALO;
     int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
     int blockCols =
         cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);
 
     int size = rows * cols;
 
     printf(
         "Loaded Pathfinder config: cols=%d rows=%d pyramid_height=%d blockCols=%d borderCols=%d smallBlockCol=%d\n",
         cols,
         rows,
         pyramid_height,
         blockCols,
         borderCols,
         smallBlockCol);
 
     int* data = (int*)malloc(sizeof(int) * (size_t)size);
 
     if (data == NULL)
     {
         fatal("unable to allocate host data");
     }
 
     initialize_data(data, rows, cols);
 
     int* d_gpuWall =
         alloc_device<int>((size_t)(size - cols), "cudaMalloc d_gpuWall");
 
     int* d_gpuResult0 =
         alloc_device<int>((size_t)cols, "cudaMalloc d_gpuResult0");
 
     int* d_gpuResult1 =
         alloc_device<int>((size_t)cols, "cudaMalloc d_gpuResult1");
 
     reset_device_state(
         d_gpuWall,
         d_gpuResult0,
         d_gpuResult1,
         data,
         rows,
         cols);
 
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
             launch_dynproc_once(
                 d_gpuWall,
                 d_gpuResult0,
                 d_gpuResult1,
                 cols,
                 rows,
                 pyramid_height,
                 blockCols,
                 borderCols);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because dynproc_kernel writes gpuResults.
      */
     reset_device_state(
         d_gpuWall,
         d_gpuResult0,
         d_gpuResult1,
         data,
         rows,
         cols);
 
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
             launch_dynproc_once(
                 d_gpuWall,
                 d_gpuResult0,
                 d_gpuResult1,
                 cols,
                 rows,
                 pyramid_height,
                 blockCols,
                 borderCols);
 
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
         "pathfinder_dynproc",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<int>(d_gpuWall, "cudaFree d_gpuWall");
     dealloc_device<int>(d_gpuResult0, "cudaFree d_gpuResult0");
     dealloc_device<int>(d_gpuResult1, "cudaFree d_gpuResult1");
 
     free(data);
 
     return EXIT_SUCCESS;
 }