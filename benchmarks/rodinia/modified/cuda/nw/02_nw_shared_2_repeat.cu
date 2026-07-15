/**
 * 01_nw_shared_1_repeat.cu / 02_nw_shared_2_repeat.cu
 *
 * Isolated measurement for Rodinia Needleman-Wunsch kernels.
 *
 * Compile with:
 *   -DMEASURE_KERNEL=1  -> needle_cuda_shared_1
 *   -DMEASURE_KERNEL=2  -> needle_cuda_shared_2
 *
 * Usage:
 *   ./01_nw_shared_1_repeat.exe <dimension> <penalty>
 *   ./02_nw_shared_2_repeat.exe <dimension> <penalty>
 *
 * Example:
 *   ./01_nw_shared_1_repeat.exe 4096 10
 *   ./02_nw_shared_2_repeat.exe 4096 10
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
 
 #ifndef BLOCK_SIZE
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
 
 #ifndef MEASURE_KERNEL
 #define MEASURE_KERNEL 1
 #endif
 
 __device__ __host__ int maximum(int a, int b, int c)
 {
     int k;
 
     if (a <= b)
     {
         k = b;
     }
     else
     {
         k = a;
     }
 
     if (k <= c)
     {
         return c;
     }
     else
     {
         return k;
     }
 }
 
 __global__ void needle_cuda_shared_1(
     int* referrence,
     int* matrix_cuda,
     int cols,
     int penalty,
     int i,
     int block_width)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
 
     int b_index_x = bx;
     int b_index_y = i - 1 - bx;
 
     int index =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         tx +
         (cols + 1);
 
     int index_n =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         tx +
         1;
 
     int index_w =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         cols;
 
     int index_nw =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x;
 
     __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
     __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];
 
     if (tx == 0)
     {
         temp[tx][0] = matrix_cuda[index_nw];
     }
 
     for (int ty = 0; ty < BLOCK_SIZE; ty++)
     {
         ref[ty][tx] = referrence[index + cols * ty];
     }
 
     __syncthreads();
 
     temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
 
     __syncthreads();
 
     temp[0][tx + 1] = matrix_cuda[index_n];
 
     __syncthreads();
 
     for (int m = 0; m < BLOCK_SIZE; m++)
     {
         if (tx <= m)
         {
             int t_index_x = tx + 1;
             int t_index_y = m - tx + 1;
 
             temp[t_index_y][t_index_x] =
                 maximum(
                     temp[t_index_y - 1][t_index_x - 1] +
                         ref[t_index_y - 1][t_index_x - 1],
                     temp[t_index_y][t_index_x - 1] - penalty,
                     temp[t_index_y - 1][t_index_x] - penalty);
         }
 
         __syncthreads();
     }
 
     for (int m = BLOCK_SIZE - 2; m >= 0; m--)
     {
         if (tx <= m)
         {
             int t_index_x = tx + BLOCK_SIZE - m;
             int t_index_y = BLOCK_SIZE - tx;
 
             temp[t_index_y][t_index_x] =
                 maximum(
                     temp[t_index_y - 1][t_index_x - 1] +
                         ref[t_index_y - 1][t_index_x - 1],
                     temp[t_index_y][t_index_x - 1] - penalty,
                     temp[t_index_y - 1][t_index_x] - penalty);
         }
 
         __syncthreads();
     }
 
     for (int ty = 0; ty < BLOCK_SIZE; ty++)
     {
         matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
     }
 }
 
 __global__ void needle_cuda_shared_2(
     int* referrence,
     int* matrix_cuda,
     int cols,
     int penalty,
     int i,
     int block_width)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
 
     int b_index_x = bx + block_width - i;
     int b_index_y = block_width - bx - 1;
 
     int index =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         tx +
         (cols + 1);
 
     int index_n =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         tx +
         1;
 
     int index_w =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x +
         cols;
 
     int index_nw =
         cols * BLOCK_SIZE * b_index_y +
         BLOCK_SIZE * b_index_x;
 
     __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
     __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];
 
     for (int ty = 0; ty < BLOCK_SIZE; ty++)
     {
         ref[ty][tx] = referrence[index + cols * ty];
     }
 
     __syncthreads();
 
     if (tx == 0)
     {
         temp[tx][0] = matrix_cuda[index_nw];
     }
 
     temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
 
     __syncthreads();
 
     temp[0][tx + 1] = matrix_cuda[index_n];
 
     __syncthreads();
 
     for (int m = 0; m < BLOCK_SIZE; m++)
     {
         if (tx <= m)
         {
             int t_index_x = tx + 1;
             int t_index_y = m - tx + 1;
 
             temp[t_index_y][t_index_x] =
                 maximum(
                     temp[t_index_y - 1][t_index_x - 1] +
                         ref[t_index_y - 1][t_index_x - 1],
                     temp[t_index_y][t_index_x - 1] - penalty,
                     temp[t_index_y - 1][t_index_x] - penalty);
         }
 
         __syncthreads();
     }
 
     for (int m = BLOCK_SIZE - 2; m >= 0; m--)
     {
         if (tx <= m)
         {
             int t_index_x = tx + BLOCK_SIZE - m;
             int t_index_y = BLOCK_SIZE - tx;
 
             temp[t_index_y][t_index_x] =
                 maximum(
                     temp[t_index_y - 1][t_index_x - 1] +
                         ref[t_index_y - 1][t_index_x - 1],
                     temp[t_index_y][t_index_x - 1] - penalty,
                     temp[t_index_y - 1][t_index_x] - penalty);
         }
 
         __syncthreads();
     }
 
     for (int ty = 0; ty < BLOCK_SIZE; ty++)
     {
         matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
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
 
 static void initialize_inputs(
     int* input_itemsets,
     int* referrence,
     int max_rows,
     int max_cols,
     int penalty)
 {
     int dim = max_rows - 1;
 
     for (int i = 0; i < max_rows * max_cols; i++)
     {
         input_itemsets[i] = 0;
         referrence[i] = 0;
     }
 
     /*
      * Deterministic synthetic sequences.
      */
     for (int i = 1; i < max_rows; i++)
     {
         input_itemsets[i * max_cols] = (i * 7) % 10 + 1;
     }
 
     for (int j = 1; j < max_cols; j++)
     {
         input_itemsets[j] = (j * 11) % 10 + 1;
     }
 
     /*
      * Synthetic substitution score.
      * Similar role to BLOSUM lookup but avoids embedding huge table.
      */
     for (int i = 1; i < max_rows; i++)
     {
         for (int j = 1; j < max_cols; j++)
         {
             int a = input_itemsets[i * max_cols];
             int b = input_itemsets[j];
 
             if (a == b)
             {
                 referrence[i * max_cols + j] = 5;
             }
             else
             {
                 referrence[i * max_cols + j] = -3 + ((a + b) % 3);
             }
         }
     }
 
     for (int i = 1; i < max_rows; i++)
     {
         input_itemsets[i * max_cols] = -i * penalty;
     }
 
     for (int j = 1; j < max_cols; j++)
     {
         input_itemsets[j] = -j * penalty;
     }
 
     (void)dim;
 }
 
 static void launch_target_once(
     int* referrence_cuda,
     int* matrix_cuda,
     int max_cols,
     int penalty,
     int block_width)
 {
     dim3 dimBlock(BLOCK_SIZE, 1);
     dim3 dimGrid;
 
 #if MEASURE_KERNEL == 1
     int i = block_width;
     dimGrid.x = i;
     dimGrid.y = 1;
 
     needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
         referrence_cuda,
         matrix_cuda,
         max_cols,
         penalty,
         i,
         block_width);
 
     checkCuda(cudaGetLastError(), "launch nw_shared_1");
 #elif MEASURE_KERNEL == 2
     int i = block_width - 1;
     dimGrid.x = i;
     dimGrid.y = 1;
 
     needle_cuda_shared_2<<<dimGrid, dimBlock>>>(
         referrence_cuda,
         matrix_cuda,
         max_cols,
         penalty,
         i,
         block_width);
 
     checkCuda(cudaGetLastError(), "launch nw_shared_2");
 #else
 #error "MEASURE_KERNEL must be 1 or 2"
 #endif
 }
 
 static void setup_for_shared_2(
     int* referrence_cuda,
     int* matrix_cuda,
     int max_cols,
     int penalty,
     int block_width)
 {
 #if MEASURE_KERNEL == 2
     dim3 dimBlock(BLOCK_SIZE, 1);
     dim3 dimGrid;
 
     /*
      * Run the first half once so shared_2 sees a realistic matrix state.
      * This setup is outside measurement.
      */
     for (int i = 1; i <= block_width; i++)
     {
         dimGrid.x = i;
         dimGrid.y = 1;
 
         needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
             referrence_cuda,
             matrix_cuda,
             max_cols,
             penalty,
             i,
             block_width);
 
         checkCuda(cudaGetLastError(), "setup needle_cuda_shared_1");
     }
 
     checkCuda(cudaDeviceSynchronize(), "sync after setup_for_shared_2");
 #else
     (void)referrence_cuda;
     (void)matrix_cuda;
     (void)max_cols;
     (void)penalty;
     (void)block_width;
 #endif
 }
 
 static const char* kernel_name()
 {
 #if MEASURE_KERNEL == 1
     return "nw_shared_1";
 #else
     return "nw_shared_2";
 #endif
 }
 
 static void print_result(
     const char* kernel_name_value,
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
 
     printf("RESULT kernel=%s\n", kernel_name_value);
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
     fprintf(stderr, "Usage: %s <dimension> <penalty>\n", program);
     fprintf(stderr, "Example: %s 4096 10\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 3)
     {
         usage(argv[0]);
     }
 
     int dimension = atoi(argv[1]);
     int penalty = atoi(argv[2]);
 
     if (dimension <= 0 || penalty <= 0 || (dimension % BLOCK_SIZE) != 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     int max_rows = dimension + 1;
     int max_cols = dimension + 1;
     int block_width = dimension / BLOCK_SIZE;
     int size = max_rows * max_cols;
 
 #if MEASURE_KERNEL == 2
     if (block_width < 2)
     {
         fatal("dimension too small for shared_2");
     }
 #endif
 
     printf(
         "Loaded NW config: kernel=%s dimension=%d penalty=%d block_width=%d\n",
         kernel_name(),
         dimension,
         penalty,
         block_width);
 
     printf("WG size of kernel = %d\n", BLOCK_SIZE);
 
     int* input_itemsets = (int*)malloc(sizeof(int) * (size_t)size);
     int* referrence = (int*)malloc(sizeof(int) * (size_t)size);
 
     if (input_itemsets == NULL || referrence == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_inputs(
         input_itemsets,
         referrence,
         max_rows,
         max_cols,
         penalty);
 
     int* matrix_cuda = alloc_device<int>((size_t)size, "cudaMalloc matrix_cuda");
     int* referrence_cuda = alloc_device<int>((size_t)size, "cudaMalloc referrence_cuda");
 
     upload_device<int>(
         referrence_cuda,
         referrence,
         (size_t)size,
         "copy referrence_cuda");
 
     upload_device<int>(
         matrix_cuda,
         input_itemsets,
         (size_t)size,
         "copy matrix_cuda");
 
     setup_for_shared_2(
         referrence_cuda,
         matrix_cuda,
         max_cols,
         penalty,
         block_width);
 
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
             launch_target_once(
                 referrence_cuda,
                 matrix_cuda,
                 max_cols,
                 penalty,
                 block_width);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset matrix before measurement.
      */
     upload_device<int>(
         matrix_cuda,
         input_itemsets,
         (size_t)size,
         "reset matrix_cuda");
 
     setup_for_shared_2(
         referrence_cuda,
         matrix_cuda,
         max_cols,
         penalty,
         block_width);
 
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
             launch_target_once(
                 referrence_cuda,
                 matrix_cuda,
                 max_cols,
                 penalty,
                 block_width);
 
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
         kernel_name(),
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<int>(matrix_cuda, "cudaFree matrix_cuda");
     dealloc_device<int>(referrence_cuda, "cudaFree referrence_cuda");
 
     free(input_itemsets);
     free(referrence);
 
     return EXIT_SUCCESS;
 }