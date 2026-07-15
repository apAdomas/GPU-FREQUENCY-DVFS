/**
 * 01_kmeans_kmeansPoint_repeat.cu
 *
 * Isolated measurement for Rodinia Kmeans kmeansPoint.
 *
 * Setup:
 * 1) generate deterministic feature data
 * 2) allocate feature_flipped_d in row-major layout
 * 3) run invert_mapping once to create feature_d in column-major layout
 * 4) initialize clusters
 * 5) copy clusters to constant memory
 *
 * Measurement:
 * repeatedly launches only kmeansPoint.
 *
 * Usage:
 *   ./01_kmeans_kmeansPoint_repeat.exe <npoints> <nfeatures> <nclusters>
 *
 * Example:
 *   ./01_kmeans_kmeansPoint_repeat.exe 1000000 34 32
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <math.h>
 #include <float.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include "measurement_common.h"
 
 #ifndef THREADS_PER_DIM
 #define THREADS_PER_DIM 16
 #endif
 
 #ifndef THREADS_PER_BLOCK
 #define THREADS_PER_BLOCK (THREADS_PER_DIM * THREADS_PER_DIM)
 #endif
 
 #ifndef ASSUMED_NR_CLUSTERS
 #define ASSUMED_NR_CLUSTERS 32
 #endif
 
 #ifndef ASSUMED_MAX_FEATURES
 #define ASSUMED_MAX_FEATURES 34
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
 
 __constant__ float c_clusters[ASSUMED_NR_CLUSTERS * ASSUMED_MAX_FEATURES];
 
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
 
 static void fatal(const char* msg)
 {
     fprintf(stderr, "ERROR: %s\n", msg);
     exit(EXIT_FAILURE);
 }
 
 /*
  * Setup kernel.
  * Converts row-major:
  *   point0[f0,f1,...], point1[f0,f1,...]
  *
  * into column-major:
  *   f0[p0,p1,...], f1[p0,p1,...]
  *
  * This matches the original Rodinia layout used by kmeansPoint.
  */
 __global__ void invert_mapping(
     float* input,
     float* output,
     int npoints,
     int nfeatures)
 {
     int point_id = threadIdx.x + blockDim.x * blockIdx.x;
 
     if (point_id < npoints)
     {
         for (int i = 0; i < nfeatures; i++)
         {
             output[point_id + npoints * i] =
                 input[point_id * nfeatures + i];
         }
     }
 }
 
 static void launch_invert_mapping_once(
     float* feature_flipped_d,
     float* feature_d,
     int npoints,
     int nfeatures,
     int num_blocks,
     int num_threads)
 {
     invert_mapping<<<num_blocks, num_threads>>>(
         feature_flipped_d,
         feature_d,
         npoints,
         nfeatures);
 
     checkCuda(cudaGetLastError(), "launch invert_mapping");
 }
 
 /*
  * Target kernel.
  *
  * This is the core Rodinia Kmeans GPU kernel:
  * each thread assigns one point to the nearest cluster.
  *
  * Differences from original:
  * - uses direct global loads instead of legacy texture references
  * - keeps cluster centers in constant memory, like original c_clusters
  * - GPU delta / center reductions are disabled, matching the common CPU-reduction path
  */
 __global__ void kmeansPoint(
     float* features,
     int nfeatures,
     int npoints,
     int nclusters,
     int* membership)
 {
     const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
     const unsigned int point_id =
         block_id * blockDim.x * blockDim.y + threadIdx.x;
 
     if (point_id >= (unsigned int)npoints)
     {
         return;
     }
 
     int index = -1;
     float min_dist = FLT_MAX;
 
     for (int i = 0; i < nclusters; i++)
     {
         int cluster_base_index = i * nfeatures;
         float dist = 0.0f;
 
         for (int j = 0; j < nfeatures; j++)
         {
             int addr = point_id + j * npoints;
 
             float diff =
                 features[addr] -
                 c_clusters[cluster_base_index + j];
 
             dist += diff * diff;
         }
 
         if (dist < min_dist)
         {
             min_dist = dist;
             index = i;
         }
     }
 
     membership[point_id] = index;
 }
 
 static void launch_kmeansPoint_once(
     float* feature_d,
     int nfeatures,
     int npoints,
     int nclusters,
     int* membership_d,
     dim3 grid,
     dim3 threads)
 {
     kmeansPoint<<<grid, threads>>>(
         feature_d,
         nfeatures,
         npoints,
         nclusters,
         membership_d);
 
     checkCuda(cudaGetLastError(), "launch kmeans_kmeansPoint");
 }
 
 static void generate_features(
     float* features,
     int npoints,
     int nfeatures)
 {
     /*
      * Deterministic synthetic feature data.
      * Row-major layout: features[point * nfeatures + feature].
      */
     for (int i = 0; i < npoints; i++)
     {
         for (int j = 0; j < nfeatures; j++)
         {
             unsigned int x =
                 (unsigned int)(i * 1103515245u + j * 12345u + 123u);
 
             features[i * nfeatures + j] =
                 (float)(x % 10000u) / 10000.0f;
         }
     }
 }
 
 static void initialize_clusters_from_features(
     float* clusters,
     const float* features,
     int npoints,
     int nfeatures,
     int nclusters)
 {
     /*
      * Simple deterministic cluster initialization:
      * use evenly spaced points from the generated dataset.
      */
     for (int c = 0; c < nclusters; c++)
     {
         int point = (c * npoints) / nclusters;
 
         if (point >= npoints)
         {
             point = npoints - 1;
         }
 
         for (int f = 0; f < nfeatures; f++)
         {
             clusters[c * nfeatures + f] =
                 features[point * nfeatures + f];
         }
     }
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
     fprintf(stderr, "Usage: %s <npoints> <nfeatures> <nclusters>\n", program);
     fprintf(stderr, "Example: %s 1000000 34 32\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 4)
     {
         usage(argv[0]);
     }
 
     int npoints = atoi(argv[1]);
     int nfeatures = atoi(argv[2]);
     int nclusters = atoi(argv[3]);
 
     if (npoints <= 0 || nfeatures <= 0 || nclusters <= 0)
     {
         usage(argv[0]);
     }
 
     if (nclusters > ASSUMED_NR_CLUSTERS)
     {
         fatal("nclusters exceeds ASSUMED_NR_CLUSTERS");
     }
 
     if (nfeatures > ASSUMED_MAX_FEATURES)
     {
         fatal("nfeatures exceeds ASSUMED_MAX_FEATURES");
     }
 
     printf(
         "Loaded Kmeans config: npoints=%d nfeatures=%d nclusters=%d\n",
         npoints,
         nfeatures,
         nclusters);
 
     printf(
         "WG size of kernel: invert_mapping=%d, kmeansPoint=%d\n",
         THREADS_PER_BLOCK,
         THREADS_PER_BLOCK);
 
     GPU_argv_init_measurement();
 
     size_t feature_count = (size_t)npoints * (size_t)nfeatures;
     size_t cluster_count = (size_t)nclusters * (size_t)nfeatures;
 
     float* h_features =
         (float*)malloc(sizeof(float) * feature_count);
 
     float* h_clusters =
         (float*)malloc(sizeof(float) * cluster_count);
 
     if (h_features == NULL || h_clusters == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     generate_features(h_features, npoints, nfeatures);
 
     initialize_clusters_from_features(
         h_clusters,
         h_features,
         npoints,
         nfeatures,
         nclusters);
 
     float* feature_flipped_d =
         alloc_device<float>(feature_count, "cudaMalloc feature_flipped_d");
 
     float* feature_d =
         alloc_device<float>(feature_count, "cudaMalloc feature_d");
 
     int* membership_d =
         alloc_device<int>((size_t)npoints, "cudaMalloc membership_d");
 
     upload_device<float>(
         feature_flipped_d,
         h_features,
         feature_count,
         "copy feature_flipped_d");
 
     checkCuda(
         cudaMemset(membership_d, 0xff, sizeof(int) * (size_t)npoints),
         "memset membership_d");
 
     checkCuda(
         cudaMemcpyToSymbol(
             c_clusters,
             h_clusters,
             sizeof(float) * cluster_count),
         "copy c_clusters");
 
     int num_threads = THREADS_PER_BLOCK;
     int num_blocks = npoints / num_threads;
 
     if (npoints % num_threads > 0)
     {
         num_blocks++;
     }
 
     int num_blocks_perdim = (int)sqrt((double)num_blocks);
 
     while (num_blocks_perdim * num_blocks_perdim < num_blocks)
     {
         num_blocks_perdim++;
     }
 
     int padded_num_blocks = num_blocks_perdim * num_blocks_perdim;
 
     dim3 grid(num_blocks_perdim, num_blocks_perdim);
     dim3 threads(num_threads);
 
     printf(
         "num_threads=%d num_blocks=%d padded_num_blocks=%d grid=[%d,%d]\n",
         num_threads,
         num_blocks,
         padded_num_blocks,
         num_blocks_perdim,
         num_blocks_perdim);
 
     /*
      * Setup only: convert feature layout once.
      */
     launch_invert_mapping_once(
         feature_flipped_d,
         feature_d,
         npoints,
         nfeatures,
         padded_num_blocks,
         num_threads);
 
     checkCuda(cudaDeviceSynchronize(), "sync after invert_mapping");
 
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
             launch_kmeansPoint_once(
                 feature_d,
                 nfeatures,
                 npoints,
                 nclusters,
                 membership_d,
                 grid,
                 threads);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(membership_d, 0xff, sizeof(int) * (size_t)npoints),
         "reset membership_d");
 
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
             launch_kmeansPoint_once(
                 feature_d,
                 nfeatures,
                 npoints,
                 nclusters,
                 membership_d,
                 grid,
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
         "kmeans_kmeansPoint",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<float>(feature_flipped_d, "cudaFree feature_flipped_d");
     dealloc_device<float>(feature_d, "cudaFree feature_d");
     dealloc_device<int>(membership_d, "cudaFree membership_d");
 
     free(h_features);
     free(h_clusters);
 
     return EXIT_SUCCESS;
 }