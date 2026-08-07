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

constexpr int OG = 4;
constexpr int T = 16;
constexpr int TILE = OG * T;

#define HARNESS_DTYPE __nv_bfloat16

__device__ __nv_bfloat16 getElem(__nv_bfloat16* A, int x, int y, int N) {
    if (x>N || y>N) {
        printf("Out of bound elem [%d, %d] accessed, of dim %d", x, y, N);
    }
    return A[x*N+y];
}

__global__ void kernel(
      __nv_bfloat16* A,
      __nv_bfloat16* B,
      __nv_bfloat16* C,
      int N
  ) {
      int tx = threadIdx.y;
      int ty = threadIdx.x;
      int bx = blockIdx.y;
      int by = blockIdx.x;

      __shared__ __nv_bfloat16 ATile[TILE][TILE];
      __shared__ __nv_bfloat16 BTile[TILE][TILE];

      /**
       * Each thred : fill 4x4 in A, B 
       * Gen 4x4 matric in C, multiply a 4x64 in A with 64x4 in B
       */

      float c[4][4];
      #pragma unroll
      for(int j=0;j<4;j++) {
            for(int k=0;k<4;k++) {
                c[j][k]=0.0;
            }
        }

      for (int i = 0 ;i < N / TILE ; i++) {
        // Load A and B into shared mem
        #pragma unroll
        for(int j=0;j<4;j++) {
            for(int k=0;k<4;k++) {
                ATile[tx+j][ty+k] = getElem(A, bx*TILE+j*T+tx, i*TILE+k*T+ty, N);
                BTile[tx+j][ty+k] = getElem(B, i*TILE+j*T+tx, by*TILE+k*T+ty, N);
            }
        }

        __syncthreads();

        #pragma unroll
        for(int j=0;j<4;j++) {
            for(int k=0;k<4;k++) {
                for(int l=0;l<16;l++) {
                    c[j][k] += __bfloat162float(ATile[tx*OG + j][l] * BTile[l][ty*OG +k]);
                }
            }
        }

        for(int j=0;j<4;j++) {
            for(int k=0;k<4;k++) {
                C[(bx*TILE+tx+j)*N + by*TILE+ty+k] = c[j][k];
            }
        }
      }

  }

  void matmul(
      __nv_bfloat16* A,
      __nv_bfloat16* B,
      __nv_bfloat16* C,
      int N
  ) {

      dim3 threads(T, T);
      dim3 blocks(N / TILE, N / TILE);

      kernel<<<blocks, threads>>>(A, B, C, N);
  }

  #include "harness.cu"