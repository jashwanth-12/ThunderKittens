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

#define HARNESS_DTYPE __nv_bfloat16

__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= N*N) return;
    int row = elem / N;
    int col = elem % N;

    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += __bfloat162float(A[row*N + i] * B[N*i + col]);
    }
    C[elem] = __float2bfloat16(sum);
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    int threadCnt = 32*32;
    int blockCnt = (N*N + threadCnt - 1) / threadCnt;
    kernel<<<blockCnt, threadCnt>>>(A, B, C, N);
}

#include "harness.cu"
