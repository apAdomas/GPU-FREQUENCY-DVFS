/**
 * 01_streamcluster_compute_cost_repeat.cu
 *
 * Isolated measurement for Rodinia Streamcluster CUDA kernel_compute_cost.
 *
 * Usage:
 *   ./01_streamcluster_compute_cost_repeat.exe <num_points> <dim> <num_centers>
 *
 * Example:
 *   ./01_streamcluster_compute_cost_repeat.exe 1000000 32 64
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <limits.h>
 #include <math.h>
 #include <stdbool.h>
 
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
 
 #define THREADS_PER_BLOCK 512
 #define MAXBLOCKS 65536
 
 typedef struct {
     float weight;
     float* coord;
     long assign;
     float cost;
 } Point;
 
 __device__ float d_dist(
     int p1,
     int p2,
     int num,
     int dim,
     float* coord_d)
 {
     float retval = 0.0f;
 
     for (int i = 0; i < dim; i++)
     {
         float tmp = coord_d[(i * num) + p1] - coord_d[(i * num) + p2];
         retval += tmp * tmp;
     }
 
     return retval;
 }
 
 /*
  * Original Streamcluster CUDA kernel.
  */
 __global__ void kernel_compute_cost(
     int num,
     int dim,
     long x,
     Point* p,
     int K,
     int stride,
     float* coord_d,
     float* work_mem_d,
     int* center_table_d,
     bool* switch_membership_d)
 {
     const int bid = blockIdx.x + gridDim.x * blockIdx.y;
     const int tid = blockDim.x * bid + threadIdx.x;
 
     if (tid < num)
     {
         float* lower = &work_mem_d[tid * stride];
 
         float x_cost =
             d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
 
         if (x_cost < p[tid].cost)
         {
             switch_membership_d[tid] = 1;
             lower[K] += x_cost - p[tid].cost;
         }
         else
         {
             lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
         }
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
     Point* points,
     float* coord_h,
     int* center_table,
     int num,
     int dim,
     int num_centers)
 {
     /*
      * Synthetic deterministic point cloud.
      * coord_h is stored in the same layout as original streamcluster CUDA:
      * coord_h[(dimension * num_points) + point_index].
      */
     for (int d = 0; d < dim; d++)
     {
         for (int i = 0; i < num; i++)
         {
             unsigned int v =
                 (unsigned int)(i * 1103515245u + d * 12345u + 1337u);
 
             coord_h[d * num + i] =
                 (float)(v % 10000u) / 10000.0f;
         }
     }
 
     for (int i = 0; i < num; i++)
     {
         points[i].weight = 1.0f;
         points[i].coord = NULL;
         points[i].assign = i % num_centers;
 
         /*
          * Positive baseline assignment cost.
          * Some points will switch, some will not.
          */
         points[i].cost = 0.25f + (float)(i % 1000) / 1000.0f;
     }
 
     for (int i = 0; i < num; i++)
     {
         if (i < num_centers)
         {
             center_table[i] = i;
         }
         else
         {
             center_table[i] = i % num_centers;
         }
     }
 }
 
 static void reset_device_state(
     Point* d_points,
     float* d_coord,
     int* d_center_table,
     bool* d_switch_membership,
     float* d_work_mem,
     const Point* points,
     const float* coord_h,
     const int* center_table,
     int num,
     int dim,
     int stride)
 {
     upload_device<Point>(
         d_points,
         points,
         (size_t)num,
         "copy d_points");
 
     upload_device<float>(
         d_coord,
         coord_h,
         (size_t)num * (size_t)dim,
         "copy d_coord");
 
     upload_device<int>(
         d_center_table,
         center_table,
         (size_t)num,
         "copy d_center_table");
 
     checkCuda(
         cudaMemset(
             d_switch_membership,
             0,
             sizeof(bool) * (size_t)num),
         "memset d_switch_membership");
 
     checkCuda(
         cudaMemset(
             d_work_mem,
             0,
             sizeof(float) * (size_t)stride * ((size_t)num + 1u)),
         "memset d_work_mem");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static dim3 make_grid(int num)
 {
     int num_blocks =
         (int)ceil((double)num / (double)THREADS_PER_BLOCK);
 
     int num_blocks_y =
         (int)ceil((double)num_blocks / (double)MAXBLOCKS);
 
     int num_blocks_x =
         (int)ceil((double)num_blocks / (double)num_blocks_y);
 
     return dim3(num_blocks_x, num_blocks_y, 1);
 }
 
 static void launch_compute_cost_once(
     int num,
     int dim,
     long x,
     Point* d_points,
     int K,
     int stride,
     float* d_coord,
     float* d_work_mem,
     int* d_center_table,
     bool* d_switch_membership,
     dim3 grid_size)
 {
     kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK>>>(
         num,
         dim,
         x,
         d_points,
         K,
         stride,
         d_coord,
         d_work_mem,
         d_center_table,
         d_switch_membership);
 
     checkCuda(cudaGetLastError(), "launch streamcluster_compute_cost");
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
     fprintf(stderr, "Usage: %s <num_points> <dim> <num_centers>\n", program);
     fprintf(stderr, "Example: %s 1000000 32 64\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 4)
     {
         usage(argv[0]);
     }
 
     int num = atoi(argv[1]);
     int dim = atoi(argv[2]);
     int num_centers = atoi(argv[3]);
 
     if (num <= 0 || dim <= 0 || num_centers <= 0 || num_centers > num)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     int K = num_centers;
     int stride = num_centers + 1;
     long x = num / 2;
 
     dim3 grid_size = make_grid(num);
 
     printf(
         "Loaded Streamcluster config: kernel=compute_cost num=%d dim=%d num_centers=%d stride=%d x=%ld grid=(%u,%u,%u) threads=%d\n",
         num,
         dim,
         num_centers,
         stride,
         x,
         grid_size.x,
         grid_size.y,
         grid_size.z,
         THREADS_PER_BLOCK);
 
     Point* points =
         (Point*)malloc(sizeof(Point) * (size_t)num);
 
     float* coord_h =
         (float*)malloc(sizeof(float) * (size_t)num * (size_t)dim);
 
     int* center_table =
         (int*)malloc(sizeof(int) * (size_t)num);
 
     if (points == NULL || coord_h == NULL || center_table == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_host_state(
         points,
         coord_h,
         center_table,
         num,
         dim,
         num_centers);
 
     Point* d_points =
         alloc_device<Point>((size_t)num, "cudaMalloc d_points");
 
     float* d_coord =
         alloc_device<float>(
             (size_t)num * (size_t)dim,
             "cudaMalloc d_coord");
 
     int* d_center_table =
         alloc_device<int>((size_t)num, "cudaMalloc d_center_table");
 
     bool* d_switch_membership =
         alloc_device<bool>((size_t)num, "cudaMalloc d_switch_membership");
 
     float* d_work_mem =
         alloc_device<float>(
             (size_t)stride * ((size_t)num + 1u),
             "cudaMalloc d_work_mem");
 
     reset_device_state(
         d_points,
         d_coord,
         d_center_table,
         d_switch_membership,
         d_work_mem,
         points,
         coord_h,
         center_table,
         num,
         dim,
         stride);
 
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
             launch_compute_cost_once(
                 num,
                 dim,
                 x,
                 d_points,
                 K,
                 stride,
                 d_coord,
                 d_work_mem,
                 d_center_table,
                 d_switch_membership,
                 grid_size);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because kernel mutates work_mem and switch_membership.
      */
     reset_device_state(
         d_points,
         d_coord,
         d_center_table,
         d_switch_membership,
         d_work_mem,
         points,
         coord_h,
         center_table,
         num,
         dim,
         stride);
 
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
             launch_compute_cost_once(
                 num,
                 dim,
                 x,
                 d_points,
                 K,
                 stride,
                 d_coord,
                 d_work_mem,
                 d_center_table,
                 d_switch_membership,
                 grid_size);
 
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
         "streamcluster_compute_cost",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<Point>(d_points, "cudaFree d_points");
     dealloc_device<float>(d_coord, "cudaFree d_coord");
     dealloc_device<int>(d_center_table, "cudaFree d_center_table");
     dealloc_device<bool>(d_switch_membership, "cudaFree d_switch_membership");
     dealloc_device<float>(d_work_mem, "cudaFree d_work_mem");
 
     free(points);
     free(coord_h);
     free(center_table);
 
     return EXIT_SUCCESS;
 }