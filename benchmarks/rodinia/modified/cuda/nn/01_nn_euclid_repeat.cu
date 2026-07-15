/**
 * 01_nn_euclid_repeat.cu
 *
 * Isolated measurement for Rodinia NN euclid kernel.
 *
 * Measures a single euclid launch repeatedly.
 *
 * Setup:
 * 1) generate deterministic LatLong records
 * 2) allocate/copy locations once
 * 3) allocate distances once
 * 4) warm up with repeated euclid launches
 * 5) reset output buffer
 * 6) measure repeated euclid launches
 *    - CUDA events for runtime
 *    - NVML total energy for energy
 * 7) print RESULT lines
 * 8) exit
 *
 * Usage:
 *   ./01_nn_euclid_repeat.exe <num_records>
 *
 * Example:
 *   ./01_nn_euclid_repeat.exe 10000000
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
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 #ifndef DEFAULT_THREADS_PER_BLOCK
 #define DEFAULT_THREADS_PER_BLOCK 256
 #endif
 
 #define ceilDiv(a, b) (((a) + (b) - 1) / (b))
 
 typedef struct latLong
 {
     float lat;
     float lng;
 } LatLong;
 
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
 
 /*
  * Target Rodinia NN kernel.
  */
 __global__ void euclid(
     LatLong* d_locations,
     float* d_distances,
     int numRecords,
     float lat,
     float lng)
 {
     int globalId =
         blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
 
     if (globalId < numRecords)
     {
         LatLong* latLong = d_locations + globalId;
         float* dist = d_distances + globalId;
 
         *dist =
             sqrtf(
                 (lat - latLong->lat) * (lat - latLong->lat) +
                 (lng - latLong->lng) * (lng - latLong->lng));
     }
 }
 
 static void launch_euclid_once(
     LatLong* d_locations,
     float* d_distances,
     int numRecords,
     float lat,
     float lng,
     dim3 gridDim,
     unsigned long threadsPerBlock)
 {
     euclid<<<gridDim, threadsPerBlock>>>(
         d_locations,
         d_distances,
         numRecords,
         lat,
         lng);
 
     checkCuda(cudaGetLastError(), "launch nn_euclid");
 }
 
 static void generate_locations(
     LatLong* locations,
     int numRecords)
 {
     /*
      * Deterministic synthetic coordinates.
      * Values are kept in a similar rough range to latitude/longitude data.
      */
     for (int i = 0; i < numRecords; i++)
     {
         unsigned int x =
             (unsigned int)(i * 1103515245u + 12345u);
 
         float a = (float)((x >> 0) & 0xffffu) / 65535.0f;
         float b = (float)((x >> 16) & 0xffffu) / 65535.0f;
 
         locations[i].lat = -90.0f + 180.0f * a;
         locations[i].lng = -180.0f + 360.0f * b;
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
     fprintf(stderr, "Usage: %s <num_records>\n", program);
     fprintf(stderr, "Example: %s 10000000\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 2)
     {
         usage(argv[0]);
     }
 
     int numRecords = atoi(argv[1]);
 
     if (numRecords <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     printf("Loaded NN config: numRecords=%d\n", numRecords);
 
     LatLong* locations =
         (LatLong*)malloc(sizeof(LatLong) * (size_t)numRecords);
 
     if (locations == NULL)
     {
         fatal("unable to allocate host locations");
     }
 
     generate_locations(locations, numRecords);
 
     LatLong* d_locations =
         alloc_device<LatLong>((size_t)numRecords, "cudaMalloc d_locations");
 
     float* d_distances =
         alloc_device<float>((size_t)numRecords, "cudaMalloc d_distances");
 
     upload_device<LatLong>(
         d_locations,
         locations,
         (size_t)numRecords,
         "copy d_locations");
 
     checkCuda(
         cudaMemset(d_distances, 0, sizeof(float) * (size_t)numRecords),
         "memset d_distances");
 
     cudaDeviceProp deviceProp;
     checkCuda(
         cudaGetDeviceProperties(&deviceProp, GPU_DEVICE),
         "cudaGetDeviceProperties");
 
     unsigned long maxGridX = deviceProp.maxGridSize[0];
 
     unsigned long threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
     if ((unsigned long)deviceProp.maxThreadsPerBlock < threadsPerBlock)
     {
         threadsPerBlock = (unsigned long)deviceProp.maxThreadsPerBlock;
     }
 
     unsigned long blocks =
         ceilDiv((unsigned long)numRecords, threadsPerBlock);
 
     unsigned long gridY = ceilDiv(blocks, maxGridX);
     unsigned long gridX = ceilDiv(blocks, gridY);
 
     dim3 gridDim(gridX, gridY);
 
     printf(
         "threadsPerBlock=%lu blocks=%lu grid=[%lu,%lu]\n",
         threadsPerBlock,
         blocks,
         gridX,
         gridY);
 
     float target_lat = 30.0f;
     float target_lng = 90.0f;
 
     checkCuda(cudaDeviceSynchronize(), "sync after setup");
 
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
             launch_euclid_once(
                 d_locations,
                 d_distances,
                 numRecords,
                 target_lat,
                 target_lng,
                 gridDim,
                 threadsPerBlock);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(d_distances, 0, sizeof(float) * (size_t)numRecords),
         "reset d_distances");
 
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
             launch_euclid_once(
                 d_locations,
                 d_distances,
                 numRecords,
                 target_lat,
                 target_lng,
                 gridDim,
                 threadsPerBlock);
 
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
         "nn_euclid",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<LatLong>(d_locations, "cudaFree d_locations");
     dealloc_device<float>(d_distances, "cudaFree d_distances");
 
     free(locations);
 
     return EXIT_SUCCESS;
 }