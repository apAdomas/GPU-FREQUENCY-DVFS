/**
 * global_warmup_fdtd_sequence.cu
 *
 * Dedicated global GPU thermal warmup executable.
 *
 * Purpose:
 * - run the full FDTD sequence repeatedly for GLOBAL_WARMUP_SECONDS
 * - no time/energy measurement
 * - used only to bring GPU to a stable hot state before oracle runs
 *
 * Important:
 * - clocks are locked outside this executable by the shell script
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 
 #include "../FDTD-2D/fdtd2d.cuh"
 #include "../../common/polybench.h"
 #include "../../common/polybenchUtilFuncts.h"
 #include "measurement_common.h"
 
 #ifndef GLOBAL_WARMUP_SECONDS
 #define GLOBAL_WARMUP_SECONDS 180.0
 #endif
 
 #ifndef SEQUENCE_TMAX
 #define SEQUENCE_TMAX 20
 #endif
 
 void init_arrays(
     int tmax,
     int nx,
     int ny,
     DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX),
     DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
     DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
     DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
 {
     int i, j;
 
     for (i = 0; i < tmax; i++) {
         _fict_[i] = (DATA_TYPE)i;
     }
 
     for (i = 0; i < nx; i++) {
         for (j = 0; j < ny; j++) {
             ex[i][j] = ((DATA_TYPE)i * (j + 1) + 1) / NX;
             ey[i][j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / NX;
             hz[i][j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / NX;
         }
     }
 }
 
 __global__ void fdtd_step1_kernel(
     int nx,
     int ny,
     DATA_TYPE* _fict_,
     DATA_TYPE* ex,
     DATA_TYPE* ey,
     DATA_TYPE* hz,
     int t)
 {
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     int i = blockIdx.y * blockDim.y + threadIdx.y;
 
     if ((i < _PB_NX) && (j < _PB_NY)) {
         if (i == 0) {
             ey[i * NY + j] = _fict_[t];
         } else {
             ey[i * NY + j] =
                 ey[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
         }
     }
 }
 
 __global__ void fdtd_step2_kernel(
     int nx,
     int ny,
     DATA_TYPE* ex,
     DATA_TYPE* ey,
     DATA_TYPE* hz,
     int t)
 {
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     int i = blockIdx.y * blockDim.y + threadIdx.y;
 
     if ((i < _PB_NX) && (j < _PB_NY) && (j > 0)) {
         ex[i * NY + j] =
             ex[i * NY + j] - 0.5f * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
     }
 }
 
 __global__ void fdtd_step3_kernel(
     int nx,
     int ny,
     DATA_TYPE* ex,
     DATA_TYPE* ey,
     DATA_TYPE* hz,
     int t)
 {
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     int i = blockIdx.y * blockDim.y + threadIdx.y;
 
     if ((i < (_PB_NX - 1)) && (j < (_PB_NY - 1))) {
         hz[i * NY + j] =
             hz[i * NY + j] - 0.7f * (ex[i * NY + (j + 1)] - ex[i * NY + j]
                                    + ey[(i + 1) * NY + j] - ey[i * NY + j]);
     }
 }
 
 static void copy_inputs_to_device(
     DATA_TYPE* _fict_gpu,
     DATA_TYPE* ex_gpu,
     DATA_TYPE* ey_gpu,
     DATA_TYPE* hz_gpu,
     DATA_TYPE* _fict,
     DATA_TYPE (*ex)[NY],
     DATA_TYPE (*ey)[NY],
     DATA_TYPE (*hz)[NY],
     int tmax)
 {
     checkCuda(cudaMemcpy(_fict_gpu, _fict, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice), "copy _fict");
     checkCuda(cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy ex");
     checkCuda(cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy ey");
     checkCuda(cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice), "copy hz");
     checkCuda(cudaDeviceSynchronize(), "sync after input copy");
 }
 
 static void run_sequence_once(
     int tmax,
     int nx,
     int ny,
     DATA_TYPE* _fict_gpu,
     DATA_TYPE* ex_gpu,
     DATA_TYPE* ey_gpu,
     DATA_TYPE* hz_gpu,
     dim3 grid,
     dim3 block)
 {
     for (int t = 0; t < tmax; t++) {
         fdtd_step1_kernel<<<grid, block>>>(nx, ny, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
         checkCuda(cudaGetLastError(), "launch step1");
 
         fdtd_step2_kernel<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu, t);
         checkCuda(cudaGetLastError(), "launch step2");
 
         fdtd_step3_kernel<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu, t);
         checkCuda(cudaGetLastError(), "launch step3");
     }
 }
 
 int main()
 {
     int tmax = SEQUENCE_TMAX;
     int nx = NX;
     int ny = NY;
 
     POLYBENCH_1D_ARRAY_DECL(_fict_, DATA_TYPE, TMAX, TMAX);
     POLYBENCH_2D_ARRAY_DECL(ex, DATA_TYPE, NX, NY, nx, ny);
     POLYBENCH_2D_ARRAY_DECL(ey, DATA_TYPE, NX, NY, nx, ny);
     POLYBENCH_2D_ARRAY_DECL(hz, DATA_TYPE, NX, NY, nx, ny);
 
     DATA_TYPE* _fict_gpu = NULL;
     DATA_TYPE* ex_gpu = NULL;
     DATA_TYPE* ey_gpu = NULL;
     DATA_TYPE* hz_gpu = NULL;
 
     init_arrays(
         tmax, nx, ny,
         POLYBENCH_ARRAY(_fict_),
         POLYBENCH_ARRAY(ex),
         POLYBENCH_ARRAY(ey),
         POLYBENCH_ARRAY(hz));
 
     GPU_argv_init_measurement();
 
     checkCuda(cudaMalloc((void**)&_fict_gpu, sizeof(DATA_TYPE) * tmax), "cudaMalloc _fict_gpu");
     checkCuda(cudaMalloc((void**)&ex_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ex_gpu");
     checkCuda(cudaMalloc((void**)&ey_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc ey_gpu");
     checkCuda(cudaMalloc((void**)&hz_gpu, sizeof(DATA_TYPE) * NX * NY), "cudaMalloc hz_gpu");
 
     dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
     dim3 grid(
         (size_t)ceil(((float)NY) / ((float)block.x)),
         (size_t)ceil(((float)NX) / ((float)block.y)));
 
     copy_inputs_to_device(
         _fict_gpu, ex_gpu, ey_gpu, hz_gpu,
         POLYBENCH_ARRAY(_fict_),
         POLYBENCH_ARRAY(ex),
         POLYBENCH_ARRAY(ey),
         POLYBENCH_ARRAY(hz),
         tmax);
 
     int completed_repeats = 0;
     double start = now_seconds();
 
     while ((now_seconds() - start) < GLOBAL_WARMUP_SECONDS) {
         run_sequence_once(tmax, nx, ny, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, grid, block);
         checkCuda(cudaDeviceSynchronize(), "sync after warmup sequence");
         completed_repeats++;
     }
 
     printf("RESULT global_warmup_seconds_target=%.3f\n", (double)GLOBAL_WARMUP_SECONDS);
     printf("RESULT completed_sequence_repeats=%d\n", completed_repeats);
 
     checkCuda(cudaFree(_fict_gpu), "cudaFree _fict_gpu");
     checkCuda(cudaFree(ex_gpu), "cudaFree ex_gpu");
     checkCuda(cudaFree(ey_gpu), "cudaFree ey_gpu");
     checkCuda(cudaFree(hz_gpu), "cudaFree hz_gpu");
 
     POLYBENCH_FREE_ARRAY(_fict_);
     POLYBENCH_FREE_ARRAY(ex);
     POLYBENCH_FREE_ARRAY(ey);
     POLYBENCH_FREE_ARRAY(hz);
 
     return 0;
 }
 
 #include "../../common/polybench.c"