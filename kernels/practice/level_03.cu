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


__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    __shared__ __nv_bfloat16 ATile[16][16];
    __shared__ __nv_bfloat16 BTile[16][16];
    __shared__ __nv_bfloat16 CTile[16][16];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    CTile[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    float totalSum = 0.0;
    for (int i=0;i<N/16;i++) {
        // Load ith tile in A and B into shared mem
        int aRow = blockIdx.x * blockDim.x + threadIdx.x;
        int aCol = i * blockDim.y + threadIdx.y;
        int bRow = i * blockDim.x + threadIdx.x;
        int bCol = blockIdx.y * blockDim.y + threadIdx.y;
        ATile[threadIdx.x][threadIdx.y] = A[aRow * N + aCol];
        BTile[threadIdx.x][threadIdx.y] = B[bRow * N + bCol];

        __syncthreads();

        float sum = 0.0;
        for(int j=0;j<16;j++) {
            sum += __bfloat162float(ATile[threadIdx.x][threadIdx.y] * BTile[threadIdx.x][threadIdx.y]);
        }
        CTile[threadIdx.x][threadIdx.y] += sum;
    }
    __syncthreads();

    C[row * N + col] = __float2bfloat16(CTile[threadIdx.x][threadIdx.y]);
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    int T = 16;
    dim3 threads(T, T);
    dim3 blocks(N / T, N /T);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "harness.cu"
