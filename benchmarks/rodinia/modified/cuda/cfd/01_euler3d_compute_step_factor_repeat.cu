/**
 * 01_euler3d_compute_step_factor_repeat.cu
 *
 * Isolated measurement for regular Rodinia CFD euler3d compute_step_factor.
 *
 * Measures a single cuda_compute_step_factor launch repeatedly for
 * energy/time measurement.
 *
 * Protocol:
 * 1) load CFD mesh once
 * 2) allocate/copy once
 * 3) initialize variables once
 * 4) warm up with repeated cuda_compute_step_factor launches
 * 5) reset output buffer
 * 6) measure repeated cuda_compute_step_factor launches
 *    - CUDA events for runtime
 *    - NVML total energy for energy
 * 7) print RESULT lines
 * 8) exit
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <stdint.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include <iostream>
 #include <fstream>
 
 #include "measurement_common.h"
 
 /*
  * Options
  */
 #define GAMMA 1.4f
 #define NDIM 3
 #define NNB 4
 #define RK 3
 #define ff_mach 1.2f
 #define deg_angle_of_attack 0.0f
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 /*
  * Rodinia work-group sizes.
  */
 #ifdef RD_WG_SIZE_0_0
     #define BLOCK_SIZE_0 RD_WG_SIZE_0_0
 #elif defined(RD_WG_SIZE_0)
     #define BLOCK_SIZE_0 RD_WG_SIZE_0
 #elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE_0 RD_WG_SIZE
 #else
     #define BLOCK_SIZE_0 192
 #endif
 
 #ifdef RD_WG_SIZE_1_0
     #define BLOCK_SIZE_1 RD_WG_SIZE_1_0
 #elif defined(RD_WG_SIZE_1)
     #define BLOCK_SIZE_1 RD_WG_SIZE_1
 #elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE_1 RD_WG_SIZE
 #else
     #define BLOCK_SIZE_1 192
 #endif
 
 #ifdef RD_WG_SIZE_2_0
     #define BLOCK_SIZE_2 RD_WG_SIZE_2_0
 #elif defined(RD_WG_SIZE_2)
     #define BLOCK_SIZE_2 RD_WG_SIZE_2
 #elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE_2 RD_WG_SIZE
 #else
     #define BLOCK_SIZE_2 192
 #endif
 
 #define VAR_DENSITY 0
 #define VAR_MOMENTUM 1
 #define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
 #define NVAR (VAR_DENSITY_ENERGY + 1)
 
 /*
  * Generic CUDA allocation helpers.
  */
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
  * CFD constants.
  */
 __constant__ float ff_variable[NVAR];
 
 /*
  * Init kernel.
  * Used only for setup, not measured.
  */
 __global__ void cuda_initialize_variables(int nelr, float* variables)
 {
     const int i = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (i < nelr)
     {
         for (int j = 0; j < NVAR; j++)
         {
             variables[i + j * nelr] = ff_variable[j];
         }
     }
 }
 
 static void initialize_variables(int nelr, float* variables)
 {
     dim3 Db(BLOCK_SIZE_1);
     dim3 Dg(nelr / BLOCK_SIZE_1);
 
     cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
     checkCuda(cudaGetLastError(), "launch cuda_initialize_variables");
 }
 
 /*
  * Device math helpers needed by compute_step_factor.
  */
 __device__ inline void compute_velocity(float& density, float3& momentum, float3& velocity)
 {
     velocity.x = momentum.x / density;
     velocity.y = momentum.y / density;
     velocity.z = momentum.z / density;
 }
 
 __device__ inline float compute_speed_sqd(float3& velocity)
 {
     return velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z;
 }
 
 __device__ inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
 {
     return (float(GAMMA) - float(1.0f)) *
            (density_energy - float(0.5f) * density * speed_sqd);
 }
 
 __device__ inline float compute_speed_of_sound(float& density, float& pressure)
 {
     return sqrtf(float(GAMMA) * pressure / density);
 }
 
 /*
  * Target kernel.
  */
 __global__ void cuda_compute_step_factor(
     int nelr,
     float* variables,
     float* areas,
     float* step_factors)
 {
     const int i = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (i < nelr)
     {
         float density = variables[i + VAR_DENSITY * nelr];
 
         float3 momentum;
         momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
         momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
         momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];
 
         float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
 
         float3 velocity;
         compute_velocity(density, momentum, velocity);
 
         float speed_sqd = compute_speed_sqd(velocity);
         float pressure = compute_pressure(density, density_energy, speed_sqd);
         float speed_of_sound = compute_speed_of_sound(density, pressure);
 
         step_factors[i] =
             float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
     }
 }
 
 static void launch_compute_step_factor_once(
     int nelr,
     float* variables,
     float* areas,
     float* step_factors)
 {
     dim3 Db(BLOCK_SIZE_2);
     dim3 Dg(nelr / BLOCK_SIZE_2);
 
     cuda_compute_step_factor<<<Dg, Db>>>(
         nelr,
         variables,
         areas,
         step_factors);
 
     checkCuda(cudaGetLastError(), "launch euler3d_compute_step_factor");
 }
 
 /*
  * Far-field setup.
  * Only ff_variable is needed for this repeat file because initialize_variables
  * uses it to initialize variables.
  */
 static void setup_far_field()
 {
     float h_ff_variable[NVAR];
 
     const float angle_of_attack =
         float(3.1415926535897931f / 180.0f) * float(deg_angle_of_attack);
 
     h_ff_variable[VAR_DENSITY] = float(1.4f);
 
     float ff_pressure = float(1.0f);
     float ff_speed_of_sound =
         sqrtf(float(GAMMA) * ff_pressure / h_ff_variable[VAR_DENSITY]);
 
     float ff_speed = float(ff_mach) * ff_speed_of_sound;
 
     float3 ff_velocity;
     ff_velocity.x = ff_speed * cosf(angle_of_attack);
     ff_velocity.y = ff_speed * sinf(angle_of_attack);
     ff_velocity.z = 0.0f;
 
     h_ff_variable[VAR_MOMENTUM + 0] =
         h_ff_variable[VAR_DENSITY] * ff_velocity.x;
     h_ff_variable[VAR_MOMENTUM + 1] =
         h_ff_variable[VAR_DENSITY] * ff_velocity.y;
     h_ff_variable[VAR_MOMENTUM + 2] =
         h_ff_variable[VAR_DENSITY] * ff_velocity.z;
 
     h_ff_variable[VAR_DENSITY_ENERGY] =
         h_ff_variable[VAR_DENSITY] *
             (float(0.5f) * (ff_speed * ff_speed)) +
         (ff_pressure / float(GAMMA - 1.0f));
 
     checkCuda(
         cudaMemcpyToSymbol(
             ff_variable,
             h_ff_variable,
             NVAR * sizeof(float)),
         "copy ff_variable");
 }
 
 /*
  * Load CFD geometry.
  *
  * For compute_step_factor, only areas are needed by the measured kernel.
  * However, this follows the original euler3d parsing behavior closely and
  * reads the full mesh line format correctly.
  */
 static void load_geometry(
     const char* data_file_name,
     int* nel_out,
     int* nelr_out,
     float** areas_out)
 {
     std::ifstream file(data_file_name);
 
     if (!file.is_open())
     {
         fprintf(stderr, "ERROR: could not open CFD data file: %s\n", data_file_name);
         exit(EXIT_FAILURE);
     }
 
     int nel = 0;
     file >> nel;
 
     if (nel <= 0)
     {
         fprintf(stderr, "ERROR: invalid nel read from CFD data file\n");
         exit(EXIT_FAILURE);
     }
 
     int nelr =
         BLOCK_SIZE_0 * ((nel / BLOCK_SIZE_0) + ((nel % BLOCK_SIZE_0) ? 1 : 0));
 
     float* h_areas = new float[nelr];
 
     /*
      * We still read surrounding elements and normals from the file because
      * the mesh format contains them after area. For this measured kernel
      * they are not uploaded or used.
      */
     int tmp_neighbor;
     float tmp_normal;
 
     for (int i = 0; i < nel; i++)
     {
         file >> h_areas[i];
 
         for (int j = 0; j < NNB; j++)
         {
             file >> tmp_neighbor;
 
             for (int k = 0; k < NDIM; k++)
             {
                 file >> tmp_normal;
             }
         }
     }
 
     int last = nel - 1;
     for (int i = nel; i < nelr; i++)
     {
         h_areas[i] = h_areas[last];
     }
 
     float* areas = alloc_device<float>(nelr, "cudaMalloc areas");
     upload_device<float>(areas, h_areas, nelr, "copy areas");
 
     delete[] h_areas;
 
     *nel_out = nel;
     *nelr_out = nelr;
     *areas_out = areas;
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
 
 int main(int argc, char** argv)
 {
     printf(
         "WG size of kernel:initialize = %d, "
         "WG size of kernel:compute_step_factor = %d\n",
         BLOCK_SIZE_1,
         BLOCK_SIZE_2);
 
     if (argc < 2)
     {
         printf("Usage: %s <cfd_data_file>\n", argv[0]);
         return EXIT_FAILURE;
     }
 
     const char* data_file_name = argv[1];
 
     GPU_argv_init_measurement();
 
     setup_far_field();
 
     int nel = 0;
     int nelr = 0;
     float* areas = NULL;
 
     load_geometry(
         data_file_name,
         &nel,
         &nelr,
         &areas);
 
     printf("Loaded CFD mesh: nel=%d nelr=%d\n", nel, nelr);
 
     float* variables =
         alloc_device<float>(nelr * NVAR, "cudaMalloc variables");
 
     float* step_factors =
         alloc_device<float>(nelr, "cudaMalloc step_factors");
 
     initialize_variables(nelr, variables);
 
     checkCuda(
         cudaMemset(step_factors, 0, sizeof(float) * nelr),
         "memset step_factors");
 
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
             launch_compute_step_factor_once(
                 nelr,
                 variables,
                 areas,
                 step_factors);
 
             checkCuda(
                 cudaDeviceSynchronize(),
                 "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     /*
      * Reset output buffer before measurement.
      */
     checkCuda(
         cudaMemset(step_factors, 0, sizeof(float) * nelr),
         "reset step_factors");
 
     checkCuda(cudaDeviceSynchronize(), "sync before measure");
 
     cudaEvent_t measure_start;
     cudaEvent_t measure_stop;
 
     checkCuda(
         cudaEventCreate(&measure_start),
         "cudaEventCreate measure_start");
 
     checkCuda(
         cudaEventCreate(&measure_stop),
         "cudaEventCreate measure_stop");
 
     int measured_launches = 0;
     float measured_cuda_ms = 0.0f;
 
     checkNvml(
         nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_start_mj),
         "energy_start");
 
     checkCuda(
         cudaEventRecord(measure_start),
         "record measure_start");

#ifdef NCU_PROFILE
    cudaProfilerStart();
#endif
 
     {
         double measure_wall_start = now_seconds();
 
         while ((now_seconds() - measure_wall_start) < MEASURE_SECONDS)
         {
             launch_compute_step_factor_once(
                 nelr,
                 variables,
                 areas,
                 step_factors);
 
             checkCuda(
                 cudaDeviceSynchronize(),
                 "sync after measured launch");
 
             measured_launches++;
         }
     }
 
#ifdef NCU_PROFILE
    cudaProfilerStop();
#endif

     checkCuda(
         cudaEventRecord(measure_stop),
         "record measure_stop");
 
     checkCuda(
         cudaEventSynchronize(measure_stop),
         "sync measure_stop");
 
     checkCuda(
         cudaDeviceSynchronize(),
         "final sync before energy_end");
 
     checkNvml(
         nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_end_mj),
         "energy_end");
 
     checkCuda(
         cudaEventElapsedTime(&measured_cuda_ms, measure_start, measure_stop),
         "elapsed measure");
 
     print_result(
         "euler3d_compute_step_factor",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<float>(areas, "cudaFree areas");
     dealloc_device<float>(variables, "cudaFree variables");
     dealloc_device<float>(step_factors, "cudaFree step_factors");
 
     return EXIT_SUCCESS;
 }