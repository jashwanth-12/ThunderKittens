// =====================================================================================
// Level 02 -- naive bf16 GEMM.  C = A @ B,  all row-major,  N x N x N.
//
// Structurally identical to level 01. Only the datatype changes: inputs and output are
// __nv_bfloat16 (2 bytes) instead of float (4 bytes).
//
// Halving the bytes should make it faster. It won't. Figure out why.
//
//   make LEVEL=02 run
// =====================================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

#define HARNESS_DTYPE __nv_bfloat16


__global__ void kernel(
      __nv_bfloat16* A,
      __nv_bfloat16* B,
      __nv_bfloat16* C,
      int N
  ) {
      constexpr int T = 16;
      constexpr int OG = 4;

      int tx = threadIdx.y;
      int ty = threadIdx.x;
      int bx = blockIdx.y;
      int by = blockIdx.x;

     __shared__ __nv_bfloat16 ATile[T][T];
     __shared__ __nv_bfloat16 BTile[T][OG * T];

     int tile = T*OG;

     // each thred = 1x16 @ 16x64
     float c[4] = {0.0, 0.0, 0.0, 0.0};
     for(int i=0;i<N/T;i+=T) {
        // Load A and B tiles
        ATile[tx][ty] = A[(bx*T+tx)*N+i*T+ty];
        for(int j=0;j<tile;j+=OG) {
            BTile[tx][ty+j] = B[(i*T+ty)*N + by*tile+j];
        }
        __syncthreads();

        for(int j=0;j<4;j++) {
            for(int k=0;k<T;k++) {
                c[j] += __bfloat162float(ATile[tx][k] * BTile[k][ty+j]);
            }
        }
        __syncthreads();
     }

     for(int i=0;i<4;i++) {
        C[(bx*T+tx) * N + by*tile+ty+i*T] = c[i];
     }

  }

  void matmul(
      __nv_bfloat16* A,
      __nv_bfloat16* B,
      __nv_bfloat16* C,
      int N
  ) {
      constexpr int T = 16;
      constexpr int OG = 4;

      dim3 threads(T, T);
      dim3 blocks(N / (OG * T), N / T);

      kernel<<<blocks, threads>>>(A, B, C, N);
  }

  #include "harness.cu"