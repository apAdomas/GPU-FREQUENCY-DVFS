/**
 * 01_particlefilter_likelihood_repeat.cu
 *
 * Isolated measurement for Rodinia Particle Filter likelihood_kernel.
 *
 * Measures a single likelihood_kernel launch repeatedly.
 *
 * Usage:
 *   ./01_particlefilter_likelihood_repeat.exe <IszX> <IszY> <Nfr> <Nparticles>
 *
 * Example:
 *   ./01_particlefilter_likelihood_repeat.exe 128 128 10 100000
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <limits.h>
 #include <math.h>
 #include <string.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_profiler_api.h>
 #include <nvml.h>
 
 #include "measurement_common.h"
 
 #define PI 3.1415926535897932
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 const int threads_per_block = 512;
 
 __device__ double calcLikelihoodSum(
     unsigned char* I,
     int* ind,
     int numOnes,
     int index)
 {
     double likelihoodSum = 0.0;
 
     for (int x = 0; x < numOnes; x++)
     {
         unsigned char val = I[ind[index * numOnes + x]];
 
         likelihoodSum +=
             (pow((double)(val - 100), 2.0) -
              pow((double)(val - 228), 2.0)) / 50.0;
     }
 
     return likelihoodSum;
 }
 
 __device__ double d_randu(int* seed, int index)
 {
     int M = INT_MAX;
     int A = 1103515245;
     int C = 12345;
 
     int num = A * seed[index] + C;
     seed[index] = num % M;
 
     return fabs(seed[index] / ((double)M));
 }
 
 __device__ double d_randn(int* seed, int index)
 {
     double u = d_randu(seed, index);
     double v = d_randu(seed, index);
 
     double cosine = cos(2.0 * PI * v);
     double rt = -2.0 * log(u);
 
     return sqrt(rt) * cosine;
 }
 
 __device__ double dev_round_double(double value)
 {
     int newValue = (int)value;
 
     if (value - newValue < 0.5)
     {
         return newValue;
     }
     else
     {
         return newValue + 1;
     }
 }
 
 /*
  * Target kernel from Rodinia particlefilter_float.
  */
 __global__ void likelihood_kernel(
     double* arrayX,
     double* arrayY,
     double* xj,
     double* yj,
     double* CDF,
     int* ind,
     int* objxy,
     double* likelihood,
     unsigned char* I,
     double* u,
     double* weights,
     int Nparticles,
     int countOnes,
     int max_size,
     int k,
     int IszY,
     int Nfr,
     int* seed,
     double* partial_sums)
 {
     int block_id = blockIdx.x;
     int i = blockDim.x * block_id + threadIdx.x;
     int y;
 
     int indX;
     int indY;
 
     __shared__ double buffer[512];
 
     if (i < Nparticles)
     {
         arrayX[i] = xj[i];
         arrayY[i] = yj[i];
 
         weights[i] = 1.0 / ((double)Nparticles);
 
         arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
         arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);
     }
 
     __syncthreads();
 
     if (i < Nparticles)
     {
         for (y = 0; y < countOnes; y++)
         {
             indX = (int)dev_round_double(arrayX[i]) + objxy[y * 2 + 1];
             indY = (int)dev_round_double(arrayY[i]) + objxy[y * 2];
 
             ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
 
             if (ind[i * countOnes + y] >= max_size)
             {
                 ind[i * countOnes + y] = 0;
             }
         }
 
         likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);
         likelihood[i] = likelihood[i] / countOnes;
 
         weights[i] = weights[i] * exp(likelihood[i]);
     }
 
     buffer[threadIdx.x] = 0.0;
 
     __syncthreads();
 
     if (i < Nparticles)
     {
         buffer[threadIdx.x] = weights[i];
     }
 
     __syncthreads();
 
     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
     {
         if (threadIdx.x < s)
         {
             buffer[threadIdx.x] += buffer[threadIdx.x + s];
         }
 
         __syncthreads();
     }
 
     if (threadIdx.x == 0)
     {
         partial_sums[blockIdx.x] = buffer[0];
     }
 
     __syncthreads();
 
     (void)CDF;
     (void)u;
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
 
 static double round_double_host(double value)
 {
     int newValue = (int)value;
 
     if (value - newValue < 0.5)
     {
         return newValue;
     }
     else
     {
         return newValue + 1;
     }
 }
 
 static void make_disk_neighbors(
     int radius,
     int** objxy_out,
     int* countOnes_out)
 {
     int diameter = radius * 2 - 1;
     int center = radius - 1;
 
     int* disk = (int*)calloc((size_t)diameter * diameter, sizeof(int));
 
     if (disk == NULL)
     {
         fatal("unable to allocate disk");
     }
 
     int countOnes = 0;
 
     for (int x = 0; x < diameter; x++)
     {
         for (int y = 0; y < diameter; y++)
         {
             double distance =
                 sqrt(
                     pow((double)(x - radius + 1), 2.0) +
                     pow((double)(y - radius + 1), 2.0));
 
             if (distance < radius)
             {
                 disk[x * diameter + y] = 1;
                 countOnes++;
             }
         }
     }
 
     int* objxy = (int*)malloc(sizeof(int) * (size_t)countOnes * 2);
 
     if (objxy == NULL)
     {
         fatal("unable to allocate objxy");
     }
 
     int n = 0;
 
     for (int x = 0; x < diameter; x++)
     {
         for (int y = 0; y < diameter; y++)
         {
             if (disk[x * diameter + y])
             {
                 objxy[n * 2] = y - center;
                 objxy[n * 2 + 1] = x - center;
                 n++;
             }
         }
     }
 
     free(disk);
 
     *objxy_out = objxy;
     *countOnes_out = countOnes;
 }
 
 static void initialize_video(
     unsigned char* I,
     int IszX,
     int IszY,
     int Nfr)
 {
     int max_size = IszX * IszY * Nfr;
 
     for (int i = 0; i < max_size; i++)
     {
         I[i] = 100;
     }
 
     int x0 = (int)round_double_host(IszY / 2.0);
     int y0 = (int)round_double_host(IszX / 2.0);
 
     int radius = 5;
 
     for (int k = 0; k < Nfr; k++)
     {
         int xk = abs(x0 + (k - 1));
         int yk = abs(y0 - 2 * (k - 1));
 
         for (int dx = -radius; dx <= radius; dx++)
         {
             for (int dy = -radius; dy <= radius; dy++)
             {
                 if ((dx * dx + dy * dy) <= radius * radius)
                 {
                     int xx = xk + dx;
                     int yy = yk + dy;
 
                     if (xx >= 0 && xx < IszY && yy >= 0 && yy < IszX)
                     {
                         int pos = yy * IszY * Nfr + xx * Nfr + k;
 
                         if (pos >= 0 && pos < max_size)
                         {
                             I[pos] = 228;
                         }
                     }
                 }
             }
         }
     }
 
     /*
      * Deterministic light noise.
      */
     for (int i = 0; i < max_size; i++)
     {
         unsigned int x = (unsigned int)(i * 1103515245u + 12345u);
         int noise = (int)(x % 11u) - 5;
 
         int value = (int)I[i] + noise;
 
         if (value < 0)
         {
             value = 0;
         }
 
         if (value > 255)
         {
             value = 255;
         }
 
         I[i] = (unsigned char)value;
     }
 }
 
 static void initialize_host_state(
     int IszX,
     int IszY,
     int Nfr,
     int Nparticles,
     unsigned char* I,
     int* objxy,
     int countOnes,
     double* arrayX,
     double* arrayY,
     double* xj,
     double* yj,
     double* CDF,
     int* ind,
     double* likelihood,
     double* u,
     double* weights,
     int* seed,
     double* partial_sums,
     int num_blocks)
 {
     (void)objxy;
     (void)countOnes;
 
     initialize_video(I, IszX, IszY, Nfr);
 
     double xe = round_double_host(IszY / 2.0);
     double ye = round_double_host(IszX / 2.0);
 
     for (int i = 0; i < Nparticles; i++)
     {
         arrayX[i] = xe;
         arrayY[i] = ye;
         xj[i] = xe;
         yj[i] = ye;
         CDF[i] = 0.0;
         likelihood[i] = 0.0;
         u[i] = 0.0;
         weights[i] = 1.0 / ((double)Nparticles);
         seed[i] = 1337 + i * 17;
     }
 
     memset(ind, 0, sizeof(int) * (size_t)countOnes * (size_t)Nparticles);
 
     for (int i = 0; i < num_blocks; i++)
     {
         partial_sums[i] = 0.0;
     }
 }
 
 static void reset_device_state(
     int IszX,
     int IszY,
     int Nfr,
     int Nparticles,
     int countOnes,
     int num_blocks,
     unsigned char* d_I,
     int* d_objxy,
     double* d_arrayX,
     double* d_arrayY,
     double* d_xj,
     double* d_yj,
     double* d_CDF,
     int* d_ind,
     double* d_likelihood,
     double* d_u,
     double* d_weights,
     int* d_seed,
     double* d_partial_sums,
     const unsigned char* I,
     const int* objxy,
     const double* arrayX,
     const double* arrayY,
     const double* xj,
     const double* yj,
     const double* CDF,
     const int* ind,
     const double* likelihood,
     const double* u,
     const double* weights,
     const int* seed,
     const double* partial_sums)
 {
     size_t video_count = (size_t)IszX * (size_t)IszY * (size_t)Nfr;
 
     upload_device<unsigned char>(d_I, I, video_count, "copy d_I");
     upload_device<int>(d_objxy, objxy, (size_t)countOnes * 2u, "copy d_objxy");
 
     upload_device<double>(d_arrayX, arrayX, (size_t)Nparticles, "copy d_arrayX");
     upload_device<double>(d_arrayY, arrayY, (size_t)Nparticles, "copy d_arrayY");
     upload_device<double>(d_xj, xj, (size_t)Nparticles, "copy d_xj");
     upload_device<double>(d_yj, yj, (size_t)Nparticles, "copy d_yj");
     upload_device<double>(d_CDF, CDF, (size_t)Nparticles, "copy d_CDF");
     upload_device<int>(d_ind, ind, (size_t)countOnes * (size_t)Nparticles, "copy d_ind");
     upload_device<double>(d_likelihood, likelihood, (size_t)Nparticles, "copy d_likelihood");
     upload_device<double>(d_u, u, (size_t)Nparticles, "copy d_u");
     upload_device<double>(d_weights, weights, (size_t)Nparticles, "copy d_weights");
     upload_device<int>(d_seed, seed, (size_t)Nparticles, "copy d_seed");
     upload_device<double>(d_partial_sums, partial_sums, (size_t)num_blocks, "copy d_partial_sums");
 
     checkCuda(cudaDeviceSynchronize(), "sync after reset_device_state");
 }
 
 static void launch_likelihood_once(
     double* d_arrayX,
     double* d_arrayY,
     double* d_xj,
     double* d_yj,
     double* d_CDF,
     int* d_ind,
     int* d_objxy,
     double* d_likelihood,
     unsigned char* d_I,
     double* d_u,
     double* d_weights,
     int Nparticles,
     int countOnes,
     int max_size,
     int k,
     int IszY,
     int Nfr,
     int* d_seed,
     double* d_partial_sums,
     int num_blocks)
 {
     likelihood_kernel<<<num_blocks, threads_per_block>>>(
         d_arrayX,
         d_arrayY,
         d_xj,
         d_yj,
         d_CDF,
         d_ind,
         d_objxy,
         d_likelihood,
         d_I,
         d_u,
         d_weights,
         Nparticles,
         countOnes,
         max_size,
         k,
         IszY,
         Nfr,
         d_seed,
         d_partial_sums);
 
     checkCuda(cudaGetLastError(), "launch particlefilter_likelihood");
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
     fprintf(stderr, "Usage: %s <IszX> <IszY> <Nfr> <Nparticles>\n", program);
     fprintf(stderr, "Example: %s 128 128 10 100000\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 5)
     {
         usage(argv[0]);
     }
 
     int IszX = atoi(argv[1]);
     int IszY = atoi(argv[2]);
     int Nfr = atoi(argv[3]);
     int Nparticles = atoi(argv[4]);
 
     if (IszX <= 0 || IszY <= 0 || Nfr <= 1 || Nparticles <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     int max_size = IszX * IszY * Nfr;
     int k = 1;
     int num_blocks =
         (int)ceil((double)Nparticles / (double)threads_per_block);
 
     int* objxy = NULL;
     int countOnes = 0;
 
     make_disk_neighbors(5, &objxy, &countOnes);
 
     printf(
         "Loaded ParticleFilter config: kernel=likelihood IszX=%d IszY=%d Nfr=%d Nparticles=%d countOnes=%d num_blocks=%d\n",
         IszX,
         IszY,
         Nfr,
         Nparticles,
         countOnes,
         num_blocks);
 
     size_t video_count = (size_t)IszX * (size_t)IszY * (size_t)Nfr;
 
     unsigned char* I =
         (unsigned char*)malloc(sizeof(unsigned char) * video_count);
 
     double* arrayX =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* arrayY =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* xj =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* yj =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* CDF =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     int* ind =
         (int*)malloc(sizeof(int) * (size_t)countOnes * (size_t)Nparticles);
 
     double* likelihood =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* u =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     double* weights =
         (double*)malloc(sizeof(double) * (size_t)Nparticles);
 
     int* seed =
         (int*)malloc(sizeof(int) * (size_t)Nparticles);
 
     double* partial_sums =
         (double*)malloc(sizeof(double) * (size_t)num_blocks);
 
     if (
         I == NULL ||
         arrayX == NULL ||
         arrayY == NULL ||
         xj == NULL ||
         yj == NULL ||
         CDF == NULL ||
         ind == NULL ||
         likelihood == NULL ||
         u == NULL ||
         weights == NULL ||
         seed == NULL ||
         partial_sums == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     initialize_host_state(
         IszX,
         IszY,
         Nfr,
         Nparticles,
         I,
         objxy,
         countOnes,
         arrayX,
         arrayY,
         xj,
         yj,
         CDF,
         ind,
         likelihood,
         u,
         weights,
         seed,
         partial_sums,
         num_blocks);
 
     unsigned char* d_I =
         alloc_device<unsigned char>(video_count, "cudaMalloc d_I");
 
     int* d_objxy =
         alloc_device<int>((size_t)countOnes * 2u, "cudaMalloc d_objxy");
 
     double* d_arrayX =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_arrayX");
 
     double* d_arrayY =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_arrayY");
 
     double* d_xj =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_xj");
 
     double* d_yj =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_yj");
 
     double* d_CDF =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_CDF");
 
     int* d_ind =
         alloc_device<int>((size_t)countOnes * (size_t)Nparticles, "cudaMalloc d_ind");
 
     double* d_likelihood =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_likelihood");
 
     double* d_u =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_u");
 
     double* d_weights =
         alloc_device<double>((size_t)Nparticles, "cudaMalloc d_weights");
 
     int* d_seed =
         alloc_device<int>((size_t)Nparticles, "cudaMalloc d_seed");
 
     double* d_partial_sums =
         alloc_device<double>((size_t)num_blocks, "cudaMalloc d_partial_sums");
 
     reset_device_state(
         IszX,
         IszY,
         Nfr,
         Nparticles,
         countOnes,
         num_blocks,
         d_I,
         d_objxy,
         d_arrayX,
         d_arrayY,
         d_xj,
         d_yj,
         d_CDF,
         d_ind,
         d_likelihood,
         d_u,
         d_weights,
         d_seed,
         d_partial_sums,
         I,
         objxy,
         arrayX,
         arrayY,
         xj,
         yj,
         CDF,
         ind,
         likelihood,
         u,
         weights,
         seed,
         partial_sums);
 
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
             launch_likelihood_once(
                 d_arrayX,
                 d_arrayY,
                 d_xj,
                 d_yj,
                 d_CDF,
                 d_ind,
                 d_objxy,
                 d_likelihood,
                 d_I,
                 d_u,
                 d_weights,
                 Nparticles,
                 countOnes,
                 max_size,
                 k,
                 IszY,
                 Nfr,
                 d_seed,
                 d_partial_sums,
                 num_blocks);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
             warmup_launches++;
         }
     }
 
     /*
      * Reset before measurement because likelihood_kernel mutates:
      * arrayX, arrayY, weights, seed, likelihood, ind, partial_sums.
      */
     reset_device_state(
         IszX,
         IszY,
         Nfr,
         Nparticles,
         countOnes,
         num_blocks,
         d_I,
         d_objxy,
         d_arrayX,
         d_arrayY,
         d_xj,
         d_yj,
         d_CDF,
         d_ind,
         d_likelihood,
         d_u,
         d_weights,
         d_seed,
         d_partial_sums,
         I,
         objxy,
         arrayX,
         arrayY,
         xj,
         yj,
         CDF,
         ind,
         likelihood,
         u,
         weights,
         seed,
         partial_sums);
 
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
             launch_likelihood_once(
                 d_arrayX,
                 d_arrayY,
                 d_xj,
                 d_yj,
                 d_CDF,
                 d_ind,
                 d_objxy,
                 d_likelihood,
                 d_I,
                 d_u,
                 d_weights,
                 Nparticles,
                 countOnes,
                 max_size,
                 k,
                 IszY,
                 Nfr,
                 d_seed,
                 d_partial_sums,
                 num_blocks);
 
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
         "particlefilter_likelihood",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<unsigned char>(d_I, "cudaFree d_I");
     dealloc_device<int>(d_objxy, "cudaFree d_objxy");
     dealloc_device<double>(d_arrayX, "cudaFree d_arrayX");
     dealloc_device<double>(d_arrayY, "cudaFree d_arrayY");
     dealloc_device<double>(d_xj, "cudaFree d_xj");
     dealloc_device<double>(d_yj, "cudaFree d_yj");
     dealloc_device<double>(d_CDF, "cudaFree d_CDF");
     dealloc_device<int>(d_ind, "cudaFree d_ind");
     dealloc_device<double>(d_likelihood, "cudaFree d_likelihood");
     dealloc_device<double>(d_u, "cudaFree d_u");
     dealloc_device<double>(d_weights, "cudaFree d_weights");
     dealloc_device<int>(d_seed, "cudaFree d_seed");
     dealloc_device<double>(d_partial_sums, "cudaFree d_partial_sums");
 
     free(I);
     free(objxy);
     free(arrayX);
     free(arrayY);
     free(xj);
     free(yj);
     free(CDF);
     free(ind);
     free(likelihood);
     free(u);
     free(weights);
     free(seed);
     free(partial_sums);
 
     return EXIT_SUCCESS;
 }