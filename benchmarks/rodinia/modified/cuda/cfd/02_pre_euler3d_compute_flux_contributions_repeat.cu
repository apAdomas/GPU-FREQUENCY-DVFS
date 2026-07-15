/**
 * 02_pre_euler3d_compute_flux_contributions_repeat.cu
 *
 * Isolated measurement for Rodinia CFD pre_euler3d compute_flux_contributions.
 *
 * Measures a single cuda_compute_flux_contributions launch repeatedly.
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
 
 #define GAMMA 1.4f
 #define NDIM 3
 #define NNB 4
 #define RK 3
 #define ff_mach 1.2f
 #define deg_angle_of_attack 0.0f
 
 #ifndef block_length
 #define block_length 192
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
 
 #if block_length > 128
 #warning "Rodinia pre_euler3d originally warns block_length > 128 may fail on some systems"
 #endif
 
 #define VAR_DENSITY 0
 #define VAR_MOMENTUM 1
 #define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
 #define NVAR (VAR_DENSITY_ENERGY + 1)
 
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
 
 __constant__ float ff_variable[NVAR];
 
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
     dim3 Db(block_length);
     dim3 Dg(nelr / block_length);
 
     cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
     checkCuda(cudaGetLastError(), "launch cuda_initialize_variables");
 }
 
 __device__ __host__ inline void compute_flux_contribution(
     float& density,
     float3& momentum,
     float& density_energy,
     float& pressure,
     float3& velocity,
     float3& fc_momentum_x,
     float3& fc_momentum_y,
     float3& fc_momentum_z,
     float3& fc_density_energy)
 {
     fc_momentum_x.x = velocity.x * momentum.x + pressure;
     fc_momentum_x.y = velocity.x * momentum.y;
     fc_momentum_x.z = velocity.x * momentum.z;
 
     fc_momentum_y.x = fc_momentum_x.y;
     fc_momentum_y.y = velocity.y * momentum.y + pressure;
     fc_momentum_y.z = velocity.y * momentum.z;
 
     fc_momentum_z.x = fc_momentum_x.z;
     fc_momentum_z.y = fc_momentum_y.z;
     fc_momentum_z.z = velocity.z * momentum.z + pressure;
 
     float de_p = density_energy + pressure;
     fc_density_energy.x = velocity.x * de_p;
     fc_density_energy.y = velocity.y * de_p;
     fc_density_energy.z = velocity.z * de_p;
 }
 
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
 
 /*
  * Target kernel.
  */
 __global__ void cuda_compute_flux_contributions(
     int nelr,
     float* variables,
     float* fc_momentum_x,
     float* fc_momentum_y,
     float* fc_momentum_z,
     float* fc_density_energy)
 {
     const int i = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (i >= nelr)
     {
         return;
     }
 
     float density_i = variables[i + VAR_DENSITY * nelr];
 
     float3 momentum_i;
     momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
     momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
     momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];
 
     float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];
 
     float3 velocity_i;
     compute_velocity(density_i, momentum_i, velocity_i);
 
     float speed_sqd_i = compute_speed_sqd(velocity_i);
     float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
 
     float3 fc_i_momentum_x;
     float3 fc_i_momentum_y;
     float3 fc_i_momentum_z;
     float3 fc_i_density_energy;
 
     compute_flux_contribution(
         density_i,
         momentum_i,
         density_energy_i,
         pressure_i,
         velocity_i,
         fc_i_momentum_x,
         fc_i_momentum_y,
         fc_i_momentum_z,
         fc_i_density_energy);
 
     fc_momentum_x[i + 0 * nelr] = fc_i_momentum_x.x;
     fc_momentum_x[i + 1 * nelr] = fc_i_momentum_x.y;
     fc_momentum_x[i + 2 * nelr] = fc_i_momentum_x.z;
 
     fc_momentum_y[i + 0 * nelr] = fc_i_momentum_y.x;
     fc_momentum_y[i + 1 * nelr] = fc_i_momentum_y.y;
     fc_momentum_y[i + 2 * nelr] = fc_i_momentum_y.z;
 
     fc_momentum_z[i + 0 * nelr] = fc_i_momentum_z.x;
     fc_momentum_z[i + 1 * nelr] = fc_i_momentum_z.y;
     fc_momentum_z[i + 2 * nelr] = fc_i_momentum_z.z;
 
     fc_density_energy[i + 0 * nelr] = fc_i_density_energy.x;
     fc_density_energy[i + 1 * nelr] = fc_i_density_energy.y;
     fc_density_energy[i + 2 * nelr] = fc_i_density_energy.z;
 }
 
 static void launch_compute_flux_contributions_once(
     int nelr,
     float* variables,
     float* fc_momentum_x,
     float* fc_momentum_y,
     float* fc_momentum_z,
     float* fc_density_energy)
 {
     dim3 Db(block_length);
     dim3 Dg(nelr / block_length);
 
     cuda_compute_flux_contributions<<<Dg, Db>>>(
         nelr,
         variables,
         fc_momentum_x,
         fc_momentum_y,
         fc_momentum_z,
         fc_density_energy);
 
     checkCuda(cudaGetLastError(), "launch pre_euler3d_compute_flux_contributions");
 }
 
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
  * For this kernel, the measured kernel only needs variables.
  * We still parse the full CFD mesh format so the input file is consumed correctly.
  */
 static void load_geometry(
     const char* data_file_name,
     int* nel_out,
     int* nelr_out)
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
         block_length * ((nel / block_length) + ((nel % block_length) ? 1 : 0));
 
     float tmp_area;
     int tmp_neighbor;
     float tmp_normal;
 
     for (int i = 0; i < nel; i++)
     {
         file >> tmp_area;
 
         for (int j = 0; j < NNB; j++)
         {
             file >> tmp_neighbor;
 
             for (int k = 0; k < NDIM; k++)
             {
                 file >> tmp_normal;
             }
         }
     }
 
     *nel_out = nel;
     *nelr_out = nelr;
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
         "WG size of kernel:compute_flux_contributions = %d\n",
         block_length,
         block_length);
 
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
 
     load_geometry(
         data_file_name,
         &nel,
         &nelr);
 
     printf("Loaded CFD mesh: nel=%d nelr=%d\n", nel, nelr);
 
     float* variables =
         alloc_device<float>(nelr * NVAR, "cudaMalloc variables");
 
     float* fc_momentum_x =
         alloc_device<float>(nelr * NDIM, "cudaMalloc fc_momentum_x");
 
     float* fc_momentum_y =
         alloc_device<float>(nelr * NDIM, "cudaMalloc fc_momentum_y");
 
     float* fc_momentum_z =
         alloc_device<float>(nelr * NDIM, "cudaMalloc fc_momentum_z");
 
     float* fc_density_energy =
         alloc_device<float>(nelr * NDIM, "cudaMalloc fc_density_energy");
 
     initialize_variables(nelr, variables);
 
     checkCuda(
         cudaMemset(fc_momentum_x, 0, sizeof(float) * nelr * NDIM),
         "memset fc_momentum_x");
 
     checkCuda(
         cudaMemset(fc_momentum_y, 0, sizeof(float) * nelr * NDIM),
         "memset fc_momentum_y");
 
     checkCuda(
         cudaMemset(fc_momentum_z, 0, sizeof(float) * nelr * NDIM),
         "memset fc_momentum_z");
 
     checkCuda(
         cudaMemset(fc_density_energy, 0, sizeof(float) * nelr * NDIM),
         "memset fc_density_energy");
 
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
             launch_compute_flux_contributions_once(
                 nelr,
                 variables,
                 fc_momentum_x,
                 fc_momentum_y,
                 fc_momentum_z,
                 fc_density_energy);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(fc_momentum_x, 0, sizeof(float) * nelr * NDIM),
         "reset fc_momentum_x");
 
     checkCuda(
         cudaMemset(fc_momentum_y, 0, sizeof(float) * nelr * NDIM),
         "reset fc_momentum_y");
 
     checkCuda(
         cudaMemset(fc_momentum_z, 0, sizeof(float) * nelr * NDIM),
         "reset fc_momentum_z");
 
     checkCuda(
         cudaMemset(fc_density_energy, 0, sizeof(float) * nelr * NDIM),
         "reset fc_density_energy");
 
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
             launch_compute_flux_contributions_once(
                 nelr,
                 variables,
                 fc_momentum_x,
                 fc_momentum_y,
                 fc_momentum_z,
                 fc_density_energy);
 
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
         "pre_euler3d_compute_flux_contributions",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<float>(variables, "cudaFree variables");
     dealloc_device<float>(fc_momentum_x, "cudaFree fc_momentum_x");
     dealloc_device<float>(fc_momentum_y, "cudaFree fc_momentum_y");
     dealloc_device<float>(fc_momentum_z, "cudaFree fc_momentum_z");
     dealloc_device<float>(fc_density_energy, "cudaFree fc_density_energy");
 
     return EXIT_SUCCESS;
 }