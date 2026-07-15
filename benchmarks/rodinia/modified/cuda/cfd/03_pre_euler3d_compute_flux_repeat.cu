/**
 * 03_pre_euler3d_compute_flux_repeat.cu
 *
 * Isolated measurement for Rodinia CFD pre_euler3d compute_flux.
 *
 * Setup computes fc_* once using cuda_compute_flux_contributions.
 * Measurement repeatedly launches only cuda_compute_flux.
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
 __constant__ float3 ff_fc_momentum_x[1];
 __constant__ float3 ff_fc_momentum_y[1];
 __constant__ float3 ff_fc_momentum_z[1];
 __constant__ float3 ff_fc_density_energy[1];
 
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
 
 __device__ inline float compute_speed_of_sound(float& density, float& pressure)
 {
     return sqrtf(float(GAMMA) * pressure / density);
 }
 
 /*
  * Setup kernel for precomputed flux contributions.
  * Not measured in this file.
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
 
     checkCuda(cudaGetLastError(), "launch cuda_compute_flux_contributions");
 }
 
 /*
  * Target kernel.
  */
 __global__ void cuda_compute_flux(
     int nelr,
     int* elements_surrounding_elements,
     float* normals,
     float* variables,
     float* fc_momentum_x,
     float* fc_momentum_y,
     float* fc_momentum_z,
     float* fc_density_energy,
     float* fluxes)
 {
     const float smoothing_coefficient = float(0.2f);
     const int i = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (i >= nelr)
     {
         return;
     }
 
     int j;
     int nb;
     float3 normal;
     float normal_len;
     float factor;
 
     float density_i = variables[i + VAR_DENSITY * nelr];
 
     float3 momentum_i;
     momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
     momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
     momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];
 
     float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];
 
     float3 velocity_i;
     compute_velocity(density_i, momentum_i, velocity_i);
 
     float speed_sqd_i = compute_speed_sqd(velocity_i);
     float speed_i = sqrtf(speed_sqd_i);
     float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
     float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
 
     float3 fc_i_momentum_x;
     float3 fc_i_momentum_y;
     float3 fc_i_momentum_z;
     float3 fc_i_density_energy;
 
     fc_i_momentum_x.x = fc_momentum_x[i + 0 * nelr];
     fc_i_momentum_x.y = fc_momentum_x[i + 1 * nelr];
     fc_i_momentum_x.z = fc_momentum_x[i + 2 * nelr];
 
     fc_i_momentum_y.x = fc_momentum_y[i + 0 * nelr];
     fc_i_momentum_y.y = fc_momentum_y[i + 1 * nelr];
     fc_i_momentum_y.z = fc_momentum_y[i + 2 * nelr];
 
     fc_i_momentum_z.x = fc_momentum_z[i + 0 * nelr];
     fc_i_momentum_z.y = fc_momentum_z[i + 1 * nelr];
     fc_i_momentum_z.z = fc_momentum_z[i + 2 * nelr];
 
     fc_i_density_energy.x = fc_density_energy[i + 0 * nelr];
     fc_i_density_energy.y = fc_density_energy[i + 1 * nelr];
     fc_i_density_energy.z = fc_density_energy[i + 2 * nelr];
 
     float flux_i_density = float(0.0f);
 
     float3 flux_i_momentum;
     flux_i_momentum.x = float(0.0f);
     flux_i_momentum.y = float(0.0f);
     flux_i_momentum.z = float(0.0f);
 
     float flux_i_density_energy = float(0.0f);
 
     float3 velocity_nb;
     float density_nb;
     float density_energy_nb;
 
     float3 momentum_nb;
     float3 fc_nb_momentum_x;
     float3 fc_nb_momentum_y;
     float3 fc_nb_momentum_z;
     float3 fc_nb_density_energy;
 
     float speed_sqd_nb;
     float speed_of_sound_nb;
     float pressure_nb;
 
     #pragma unroll
     for (j = 0; j < NNB; j++)
     {
         nb = elements_surrounding_elements[i + j * nelr];
 
         normal.x = normals[i + (j + 0 * NNB) * nelr];
         normal.y = normals[i + (j + 1 * NNB) * nelr];
         normal.z = normals[i + (j + 2 * NNB) * nelr];
 
         normal_len = sqrtf(
             normal.x * normal.x +
             normal.y * normal.y +
             normal.z * normal.z);
 
         if (nb >= 0)
         {
             density_nb = variables[nb + VAR_DENSITY * nelr];
 
             momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
             momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
             momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
 
             density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
 
             compute_velocity(density_nb, momentum_nb, velocity_nb);
 
             speed_sqd_nb = compute_speed_sqd(velocity_nb);
             pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
             speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
 
             fc_nb_momentum_x.x = fc_momentum_x[nb + 0 * nelr];
             fc_nb_momentum_x.y = fc_momentum_x[nb + 1 * nelr];
             fc_nb_momentum_x.z = fc_momentum_x[nb + 2 * nelr];
 
             fc_nb_momentum_y.x = fc_momentum_y[nb + 0 * nelr];
             fc_nb_momentum_y.y = fc_momentum_y[nb + 1 * nelr];
             fc_nb_momentum_y.z = fc_momentum_y[nb + 2 * nelr];
 
             fc_nb_momentum_z.x = fc_momentum_z[nb + 0 * nelr];
             fc_nb_momentum_z.y = fc_momentum_z[nb + 1 * nelr];
             fc_nb_momentum_z.z = fc_momentum_z[nb + 2 * nelr];
 
             fc_nb_density_energy.x = fc_density_energy[nb + 0 * nelr];
             fc_nb_density_energy.y = fc_density_energy[nb + 1 * nelr];
             fc_nb_density_energy.z = fc_density_energy[nb + 2 * nelr];
 
             factor =
                 -normal_len *
                 smoothing_coefficient *
                 float(0.5f) *
                 (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
 
             flux_i_density += factor * (density_i - density_nb);
             flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
             flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
             flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
             flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);
 
             factor = float(0.5f) * normal.x;
             flux_i_density += factor * (momentum_nb.x + momentum_i.x);
             flux_i_density_energy +=
                 factor * (fc_nb_density_energy.x + fc_i_density_energy.x);
             flux_i_momentum.x +=
                 factor * (fc_nb_momentum_x.x + fc_i_momentum_x.x);
             flux_i_momentum.y +=
                 factor * (fc_nb_momentum_y.x + fc_i_momentum_y.x);
             flux_i_momentum.z +=
                 factor * (fc_nb_momentum_z.x + fc_i_momentum_z.x);
 
             factor = float(0.5f) * normal.y;
             flux_i_density += factor * (momentum_nb.y + momentum_i.y);
             flux_i_density_energy +=
                 factor * (fc_nb_density_energy.y + fc_i_density_energy.y);
             flux_i_momentum.x +=
                 factor * (fc_nb_momentum_x.y + fc_i_momentum_x.y);
             flux_i_momentum.y +=
                 factor * (fc_nb_momentum_y.y + fc_i_momentum_y.y);
             flux_i_momentum.z +=
                 factor * (fc_nb_momentum_z.y + fc_i_momentum_z.y);
 
             factor = float(0.5f) * normal.z;
             flux_i_density += factor * (momentum_nb.z + momentum_i.z);
             flux_i_density_energy +=
                 factor * (fc_nb_density_energy.z + fc_i_density_energy.z);
             flux_i_momentum.x +=
                 factor * (fc_nb_momentum_x.z + fc_i_momentum_x.z);
             flux_i_momentum.y +=
                 factor * (fc_nb_momentum_y.z + fc_i_momentum_y.z);
             flux_i_momentum.z +=
                 factor * (fc_nb_momentum_z.z + fc_i_momentum_z.z);
         }
         else if (nb == -1)
         {
             flux_i_momentum.x += normal.x * pressure_i;
             flux_i_momentum.y += normal.y * pressure_i;
             flux_i_momentum.z += normal.z * pressure_i;
         }
         else if (nb == -2)
         {
             factor = float(0.5f) * normal.x;
             flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
             flux_i_density_energy +=
                 factor * (ff_fc_density_energy[0].x + fc_i_density_energy.x);
             flux_i_momentum.x +=
                 factor * (ff_fc_momentum_x[0].x + fc_i_momentum_x.x);
             flux_i_momentum.y +=
                 factor * (ff_fc_momentum_y[0].x + fc_i_momentum_y.x);
             flux_i_momentum.z +=
                 factor * (ff_fc_momentum_z[0].x + fc_i_momentum_z.x);
 
             factor = float(0.5f) * normal.y;
             flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
             flux_i_density_energy +=
                 factor * (ff_fc_density_energy[0].y + fc_i_density_energy.y);
             flux_i_momentum.x +=
                 factor * (ff_fc_momentum_x[0].y + fc_i_momentum_x.y);
             flux_i_momentum.y +=
                 factor * (ff_fc_momentum_y[0].y + fc_i_momentum_y.y);
             flux_i_momentum.z +=
                 factor * (ff_fc_momentum_z[0].y + fc_i_momentum_z.y);
 
             factor = float(0.5f) * normal.z;
             flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
             flux_i_density_energy +=
                 factor * (ff_fc_density_energy[0].z + fc_i_density_energy.z);
             flux_i_momentum.x +=
                 factor * (ff_fc_momentum_x[0].z + fc_i_momentum_x.z);
             flux_i_momentum.y +=
                 factor * (ff_fc_momentum_y[0].z + fc_i_momentum_y.z);
             flux_i_momentum.z +=
                 factor * (ff_fc_momentum_z[0].z + fc_i_momentum_z.z);
         }
     }
 
     fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
     fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
     fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
     fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
     fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
 }
 
 static void launch_compute_flux_once(
     int nelr,
     int* elements_surrounding_elements,
     float* normals,
     float* variables,
     float* fc_momentum_x,
     float* fc_momentum_y,
     float* fc_momentum_z,
     float* fc_density_energy,
     float* fluxes)
 {
     dim3 Db(block_length);
     dim3 Dg(nelr / block_length);
 
     cuda_compute_flux<<<Dg, Db>>>(
         nelr,
         elements_surrounding_elements,
         normals,
         variables,
         fc_momentum_x,
         fc_momentum_y,
         fc_momentum_z,
         fc_density_energy,
         fluxes);
 
     checkCuda(cudaGetLastError(), "launch pre_euler3d_compute_flux");
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
 
     float3 h_ff_momentum;
     h_ff_momentum.x = h_ff_variable[VAR_MOMENTUM + 0];
     h_ff_momentum.y = h_ff_variable[VAR_MOMENTUM + 1];
     h_ff_momentum.z = h_ff_variable[VAR_MOMENTUM + 2];
 
     float3 h_ff_fc_momentum_x;
     float3 h_ff_fc_momentum_y;
     float3 h_ff_fc_momentum_z;
     float3 h_ff_fc_density_energy;
 
     compute_flux_contribution(
         h_ff_variable[VAR_DENSITY],
         h_ff_momentum,
         h_ff_variable[VAR_DENSITY_ENERGY],
         ff_pressure,
         ff_velocity,
         h_ff_fc_momentum_x,
         h_ff_fc_momentum_y,
         h_ff_fc_momentum_z,
         h_ff_fc_density_energy);
 
     checkCuda(cudaMemcpyToSymbol(ff_variable, h_ff_variable, NVAR * sizeof(float)), "copy ff_variable");
     checkCuda(cudaMemcpyToSymbol(ff_fc_momentum_x, &h_ff_fc_momentum_x, sizeof(float3)), "copy ff_fc_momentum_x");
     checkCuda(cudaMemcpyToSymbol(ff_fc_momentum_y, &h_ff_fc_momentum_y, sizeof(float3)), "copy ff_fc_momentum_y");
     checkCuda(cudaMemcpyToSymbol(ff_fc_momentum_z, &h_ff_fc_momentum_z, sizeof(float3)), "copy ff_fc_momentum_z");
     checkCuda(cudaMemcpyToSymbol(ff_fc_density_energy, &h_ff_fc_density_energy, sizeof(float3)), "copy ff_fc_density_energy");
 }
 
 static void load_geometry(
     const char* data_file_name,
     int* nel_out,
     int* nelr_out,
     int** elements_surrounding_elements_out,
     float** normals_out)
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
 
     float* h_areas = new float[nelr];
     int* h_elements_surrounding_elements = new int[nelr * NNB];
     float* h_normals = new float[nelr * NDIM * NNB];
 
     for (int i = 0; i < nel; i++)
     {
         file >> h_areas[i];
 
         for (int j = 0; j < NNB; j++)
         {
             file >> h_elements_surrounding_elements[i + j * nelr];
 
             if (h_elements_surrounding_elements[i + j * nelr] < 0)
             {
                 h_elements_surrounding_elements[i + j * nelr] = -1;
             }
 
             h_elements_surrounding_elements[i + j * nelr]--;
 
             for (int k = 0; k < NDIM; k++)
             {
                 file >> h_normals[i + (j + k * NNB) * nelr];
                 h_normals[i + (j + k * NNB) * nelr] =
                     -h_normals[i + (j + k * NNB) * nelr];
             }
         }
     }
 
     int last = nel - 1;
 
     for (int i = nel; i < nelr; i++)
     {
         h_areas[i] = h_areas[last];
 
         for (int j = 0; j < NNB; j++)
         {
             h_elements_surrounding_elements[i + j * nelr] =
                 h_elements_surrounding_elements[last + j * nelr];
 
             for (int k = 0; k < NDIM; k++)
             {
                 h_normals[i + (j + k * NNB) * nelr] =
                     h_normals[last + (j + k * NNB) * nelr];
             }
         }
     }
 
     int* elements_surrounding_elements =
         alloc_device<int>(nelr * NNB, "cudaMalloc elements_surrounding_elements");
 
     float* normals =
         alloc_device<float>(nelr * NDIM * NNB, "cudaMalloc normals");
 
     upload_device<int>(
         elements_surrounding_elements,
         h_elements_surrounding_elements,
         nelr * NNB,
         "copy elements_surrounding_elements");
 
     upload_device<float>(
         normals,
         h_normals,
         nelr * NDIM * NNB,
         "copy normals");
 
     delete[] h_areas;
     delete[] h_elements_surrounding_elements;
     delete[] h_normals;
 
     *nel_out = nel;
     *nelr_out = nelr;
     *elements_surrounding_elements_out = elements_surrounding_elements;
     *normals_out = normals;
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
         "WG size of kernel:compute_flux_contributions = %d, "
         "WG size of kernel:compute_flux = %d\n",
         block_length,
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
 
     int* elements_surrounding_elements = NULL;
     float* normals = NULL;
 
     load_geometry(
         data_file_name,
         &nel,
         &nelr,
         &elements_surrounding_elements,
         &normals);
 
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
 
     float* fluxes =
         alloc_device<float>(nelr * NVAR, "cudaMalloc fluxes");
 
     initialize_variables(nelr, variables);
 
     launch_compute_flux_contributions_once(
         nelr,
         variables,
         fc_momentum_x,
         fc_momentum_y,
         fc_momentum_z,
         fc_density_energy);
 
     checkCuda(
         cudaMemset(fluxes, 0, sizeof(float) * nelr * NVAR),
         "memset fluxes");
 
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
             launch_compute_flux_once(
                 nelr,
                 elements_surrounding_elements,
                 normals,
                 variables,
                 fc_momentum_x,
                 fc_momentum_y,
                 fc_momentum_z,
                 fc_density_energy,
                 fluxes);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     checkCuda(
         cudaMemset(fluxes, 0, sizeof(float) * nelr * NVAR),
         "reset fluxes");
 
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
             launch_compute_flux_once(
                 nelr,
                 elements_surrounding_elements,
                 normals,
                 variables,
                 fc_momentum_x,
                 fc_momentum_y,
                 fc_momentum_z,
                 fc_density_energy,
                 fluxes);
 
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
         "pre_euler3d_compute_flux",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<int>(elements_surrounding_elements, "cudaFree elements_surrounding_elements");
     dealloc_device<float>(normals, "cudaFree normals");
     dealloc_device<float>(variables, "cudaFree variables");
     dealloc_device<float>(fc_momentum_x, "cudaFree fc_momentum_x");
     dealloc_device<float>(fc_momentum_y, "cudaFree fc_momentum_y");
     dealloc_device<float>(fc_momentum_z, "cudaFree fc_momentum_z");
     dealloc_device<float>(fc_density_energy, "cudaFree fc_density_energy");
     dealloc_device<float>(fluxes, "cudaFree fluxes");
 
     return EXIT_SUCCESS;
 }