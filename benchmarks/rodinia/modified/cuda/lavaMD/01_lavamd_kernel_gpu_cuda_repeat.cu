/**
 * 01_lavamd_kernel_gpu_cuda_repeat.cu
 *
 * Isolated measurement for Rodinia LavaMD kernel_gpu_cuda.
 *
 * Setup:
 * 1) generate synthetic LavaMD box grid
 * 2) generate rv/qv/fv arrays
 * 3) allocate/copy GPU memory once
 *
 * Measurement:
 * repeatedly launches only kernel_gpu_cuda.
 *
 * Usage:
 *   ./01_lavamd_kernel_gpu_cuda_repeat.exe <boxes1d>
 *
 * Example:
 *   ./01_lavamd_kernel_gpu_cuda_repeat.exe 10
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
 
 #ifndef GPU_DEVICE
 #define GPU_DEVICE 0
 #endif
 
 #ifndef WARMUP_SECONDS
 #define WARMUP_SECONDS 25.0
 #endif
 
 #ifndef MEASURE_SECONDS
 #define MEASURE_SECONDS 5.0
 #endif
 
 #define NUMBER_THREADS 128
 #define NUMBER_PAR_PER_BOX 100
 
 typedef float fp;
 
 typedef struct
 {
     fp x;
     fp y;
     fp z;
 } THREE_VECTOR;
 
 typedef struct
 {
     fp v;
     fp x;
     fp y;
     fp z;
 } FOUR_VECTOR;
 
 typedef struct
 {
     fp alpha;
 } par_str;
 
 typedef struct
 {
     int x;
     int y;
     int z;
     int number;
     long offset;
 } nei_str;
 
 typedef struct
 {
     int x;
     int y;
     int z;
     int number;
     long offset;
     int nn;
     nei_str nei[26];
 } box_str;
 
 typedef struct
 {
     int boxes1d_arg;
     long number_boxes;
     long box_mem;
     long space_elem;
     long space_mem;
     long space_mem2;
 } dim_str;
 
 #define DOT(A, B) ((A).v * (B).v + (A).x * (B).x + (A).y * (B).y + (A).z * (B).z)
 
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
  * Target kernel from Rodinia LavaMD, kept structurally equivalent.
  */
 __global__ void kernel_gpu_cuda(
     par_str d_par_gpu,
     dim_str d_dim_gpu,
     box_str* d_box_gpu,
     FOUR_VECTOR* d_rv_gpu,
     fp* d_qv_gpu,
     FOUR_VECTOR* d_fv_gpu)
 {
     int bx = blockIdx.x;
     int tx = threadIdx.x;
     int wtx = tx;
 
     if (bx < d_dim_gpu.number_boxes)
     {
         fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;
 
         int first_i;
         FOUR_VECTOR* rA;
         FOUR_VECTOR* fA;
         __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];
 
         int pointer;
         int k = 0;
         int first_j;
         FOUR_VECTOR* rB;
         fp* qB;
         int j = 0;
         __shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
         __shared__ fp qB_shared[NUMBER_PAR_PER_BOX];
 
         fp r2;
         fp u2;
         fp vij;
         fp fs;
         fp fxij;
         fp fyij;
         fp fzij;
         THREE_VECTOR d;
 
         first_i = d_box_gpu[bx].offset;
 
         rA = &d_rv_gpu[first_i];
         fA = &d_fv_gpu[first_i];
 
         while (wtx < NUMBER_PAR_PER_BOX)
         {
             rA_shared[wtx] = rA[wtx];
             wtx = wtx + NUMBER_THREADS;
         }
 
         wtx = tx;
 
         __syncthreads();
 
         for (k = 0; k < (1 + d_box_gpu[bx].nn); k++)
         {
             if (k == 0)
             {
                 pointer = bx;
             }
             else
             {
                 pointer = d_box_gpu[bx].nei[k - 1].number;
             }
 
             first_j = d_box_gpu[pointer].offset;
 
             rB = &d_rv_gpu[first_j];
             qB = &d_qv_gpu[first_j];
 
             while (wtx < NUMBER_PAR_PER_BOX)
             {
                 rB_shared[wtx] = rB[wtx];
                 qB_shared[wtx] = qB[wtx];
                 wtx = wtx + NUMBER_THREADS;
             }
 
             wtx = tx;
 
             __syncthreads();
 
             while (wtx < NUMBER_PAR_PER_BOX)
             {
                 for (j = 0; j < NUMBER_PAR_PER_BOX; j++)
                 {
                     r2 =
                         (fp)rA_shared[wtx].v +
                         (fp)rB_shared[j].v -
                         DOT(rA_shared[wtx], rB_shared[j]);
 
                     u2 = a2 * r2;
                     vij = expf(-u2);
                     fs = 2.0f * vij;
 
                     d.x = (fp)rA_shared[wtx].x - (fp)rB_shared[j].x;
                     fxij = fs * d.x;
 
                     d.y = (fp)rA_shared[wtx].y - (fp)rB_shared[j].y;
                     fyij = fs * d.y;
 
                     d.z = (fp)rA_shared[wtx].z - (fp)rB_shared[j].z;
                     fzij = fs * d.z;
 
                     fA[wtx].v += (fp)qB_shared[j] * vij;
                     fA[wtx].x += (fp)qB_shared[j] * fxij;
                     fA[wtx].y += (fp)qB_shared[j] * fyij;
                     fA[wtx].z += (fp)qB_shared[j] * fzij;
                 }
 
                 wtx = wtx + NUMBER_THREADS;
             }
 
             wtx = tx;
 
             __syncthreads();
         }
     }
 }
 
 static void launch_lavamd_once(
     par_str par_cpu,
     dim_str dim_cpu,
     box_str* d_box_gpu,
     FOUR_VECTOR* d_rv_gpu,
     fp* d_qv_gpu,
     FOUR_VECTOR* d_fv_gpu)
 {
     dim3 blocks(dim_cpu.number_boxes, 1, 1);
     dim3 threads(NUMBER_THREADS, 1, 1);
 
     kernel_gpu_cuda<<<blocks, threads>>>(
         par_cpu,
         dim_cpu,
         d_box_gpu,
         d_rv_gpu,
         d_qv_gpu,
         d_fv_gpu);
 
     checkCuda(cudaGetLastError(), "launch lavamd_kernel_gpu_cuda");
 }
 
 static long box_number(int x, int y, int z, int boxes1d)
 {
     return (long)z * boxes1d * boxes1d + (long)y * boxes1d + x;
 }
 
 static void generate_boxes(box_str* boxes, int boxes1d)
 {
     for (int z = 0; z < boxes1d; z++)
     {
         for (int y = 0; y < boxes1d; y++)
         {
             for (int x = 0; x < boxes1d; x++)
             {
                 long number = box_number(x, y, z, boxes1d);
                 box_str* box = &boxes[number];
 
                 box->x = x;
                 box->y = y;
                 box->z = z;
                 box->number = (int)number;
                 box->offset = number * NUMBER_PAR_PER_BOX;
                 box->nn = 0;
 
                 for (int dz = -1; dz <= 1; dz++)
                 {
                     for (int dy = -1; dy <= 1; dy++)
                     {
                         for (int dx = -1; dx <= 1; dx++)
                         {
                             if (dx == 0 && dy == 0 && dz == 0)
                             {
                                 continue;
                             }
 
                             int nx = x + dx;
                             int ny = y + dy;
                             int nz = z + dz;
 
                             if (nx >= 0 && nx < boxes1d &&
                                 ny >= 0 && ny < boxes1d &&
                                 nz >= 0 && nz < boxes1d)
                             {
                                 long nei_number = box_number(nx, ny, nz, boxes1d);
                                 int idx = box->nn;
 
                                 box->nei[idx].x = nx;
                                 box->nei[idx].y = ny;
                                 box->nei[idx].z = nz;
                                 box->nei[idx].number = (int)nei_number;
                                 box->nei[idx].offset =
                                     nei_number * NUMBER_PAR_PER_BOX;
 
                                 box->nn++;
                             }
                         }
                     }
                 }
             }
         }
     }
 }
 
 static void generate_particles(
     FOUR_VECTOR* rv,
     fp* qv,
     FOUR_VECTOR* fv,
     long space_elem)
 {
     for (long i = 0; i < space_elem; i++)
     {
         unsigned int x =
             (unsigned int)(i * 1103515245u + 12345u);
 
         float a = (float)((x >> 0) & 0xff) / 255.0f;
         float b = (float)((x >> 8) & 0xff) / 255.0f;
         float c = (float)((x >> 16) & 0xff) / 255.0f;
 
         rv[i].x = a;
         rv[i].y = b;
         rv[i].z = c;
         rv[i].v = a * a + b * b + c * c;
 
         qv[i] = 0.5f + 0.5f * a;
 
         fv[i].v = 0.0f;
         fv[i].x = 0.0f;
         fv[i].y = 0.0f;
         fv[i].z = 0.0f;
     }
 }
 
 static void reset_forces(FOUR_VECTOR* fv, long space_elem)
 {
     for (long i = 0; i < space_elem; i++)
     {
         fv[i].v = 0.0f;
         fv[i].x = 0.0f;
         fv[i].y = 0.0f;
         fv[i].z = 0.0f;
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
     fprintf(stderr, "Usage: %s <boxes1d>\n", program);
     fprintf(stderr, "Example: %s 10\n", program);
     exit(EXIT_FAILURE);
 }
 
 int main(int argc, char** argv)
 {
     if (argc != 2)
     {
         usage(argv[0]);
     }
 
     int boxes1d = atoi(argv[1]);
 
     if (boxes1d <= 0)
     {
         usage(argv[0]);
     }
 
     GPU_argv_init_measurement();
 
     par_str par_cpu;
     par_cpu.alpha = 0.5f;
 
     dim_str dim_cpu;
     dim_cpu.boxes1d_arg = boxes1d;
     dim_cpu.number_boxes = (long)boxes1d * boxes1d * boxes1d;
     dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
     dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
     dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
     dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);
 
     printf("Loaded LavaMD config: boxes1d=%d number_boxes=%ld particles=%ld\n",
            boxes1d,
            dim_cpu.number_boxes,
            dim_cpu.space_elem);
 
     printf("WG size of kernel = %d\n", NUMBER_THREADS);
 
     box_str* box_cpu =
         (box_str*)malloc(dim_cpu.box_mem);
 
     FOUR_VECTOR* rv_cpu =
         (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
 
     fp* qv_cpu =
         (fp*)malloc(dim_cpu.space_mem2);
 
     FOUR_VECTOR* fv_cpu =
         (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
 
     if (box_cpu == NULL || rv_cpu == NULL || qv_cpu == NULL || fv_cpu == NULL)
     {
         fatal("unable to allocate host memory");
     }
 
     generate_boxes(box_cpu, boxes1d);
     generate_particles(rv_cpu, qv_cpu, fv_cpu, dim_cpu.space_elem);
 
     box_str* d_box_gpu =
         alloc_device<box_str>(dim_cpu.number_boxes, "cudaMalloc d_box_gpu");
 
     FOUR_VECTOR* d_rv_gpu =
         alloc_device<FOUR_VECTOR>(dim_cpu.space_elem, "cudaMalloc d_rv_gpu");
 
     fp* d_qv_gpu =
         alloc_device<fp>(dim_cpu.space_elem, "cudaMalloc d_qv_gpu");
 
     FOUR_VECTOR* d_fv_gpu =
         alloc_device<FOUR_VECTOR>(dim_cpu.space_elem, "cudaMalloc d_fv_gpu");
 
     upload_device<box_str>(
         d_box_gpu,
         box_cpu,
         dim_cpu.number_boxes,
         "copy d_box_gpu");
 
     upload_device<FOUR_VECTOR>(
         d_rv_gpu,
         rv_cpu,
         dim_cpu.space_elem,
         "copy d_rv_gpu");
 
     upload_device<fp>(
         d_qv_gpu,
         qv_cpu,
         dim_cpu.space_elem,
         "copy d_qv_gpu");
 
     upload_device<FOUR_VECTOR>(
         d_fv_gpu,
         fv_cpu,
         dim_cpu.space_elem,
         "copy d_fv_gpu");
 
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
             launch_lavamd_once(
                 par_cpu,
                 dim_cpu,
                 d_box_gpu,
                 d_rv_gpu,
                 d_qv_gpu,
                 d_fv_gpu);
 
             checkCuda(cudaDeviceSynchronize(), "sync after warmup launch");
 
             warmup_launches++;
         }
     }
 
     reset_forces(fv_cpu, dim_cpu.space_elem);
 
     upload_device<FOUR_VECTOR>(
         d_fv_gpu,
         fv_cpu,
         dim_cpu.space_elem,
         "reset d_fv_gpu");
 
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
             launch_lavamd_once(
                 par_cpu,
                 dim_cpu,
                 d_box_gpu,
                 d_rv_gpu,
                 d_qv_gpu,
                 d_fv_gpu);
 
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
         "lavamd_kernel_gpu_cuda",
         warmup_launches,
         measured_launches,
         measured_cuda_ms,
         energy_start_mj,
         energy_end_mj);
 
     checkCuda(cudaEventDestroy(measure_start), "destroy measure_start");
     checkCuda(cudaEventDestroy(measure_stop), "destroy measure_stop");
 
     checkNvml(nvmlShutdown(), "nvmlShutdown");
 
     dealloc_device<box_str>(d_box_gpu, "cudaFree d_box_gpu");
     dealloc_device<FOUR_VECTOR>(d_rv_gpu, "cudaFree d_rv_gpu");
     dealloc_device<fp>(d_qv_gpu, "cudaFree d_qv_gpu");
     dealloc_device<FOUR_VECTOR>(d_fv_gpu, "cudaFree d_fv_gpu");
 
     free(box_cpu);
     free(rv_cpu);
     free(qv_cpu);
     free(fv_cpu);
 
     return EXIT_SUCCESS;
 }