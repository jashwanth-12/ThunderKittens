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

      __shared__ __nv_bfloat16 ATile[T][T];
      __shared__ __nv_bfloat16 BTile[T][T];

      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int row = blockIdx.y * T + ty;
      int col = blockIdx.x * T + tx;

      float totalSum = 0.0f;

      for (int tile = 0; tile < N / T; tile++) {
          ATile[ty][tx] = A[row * N + tile * T + tx];
          BTile[ty][tx] = B[(tile * T + ty) * N + col];

          __syncthreads();

          for (int j = 0; j < T; j++) {
              totalSum += __bfloat162float(
                  ATile[ty][j] * BTile[j][tx]
              );
          }

          __syncthreads();
      }

      C[row * N + col] = __float2bfloat16(totalSum);
  }

  void matmul(
      __nv_bfloat16* A,
      __nv_bfloat16* B,
      __nv_bfloat16* C,
      int N
  ) {
      constexpr int T = 16;

      dim3 threads(T, T);
      dim3 blocks(N / T, N / T);

      kernel<<<blocks, threads>>>(A, B, C, N);
  }

  #include "harness.cu"