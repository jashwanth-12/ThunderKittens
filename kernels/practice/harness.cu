// =====================================================================================
// Shared benchmark harness for the practice GEMM ladder.
//
// You write a kernel plus a launcher with this signature:
//
//     void matmul(T* A, T* B, T* C, int N);      // T = float or __nv_bfloat16
//
// then `#include "harness.cu"` at the bottom of your file. The harness supplies main().
//
// Everything is square N x N x N, all row-major:  C = A @ B
//
// Configure by #define-ing before the include:
//     HARNESS_DTYPE   float | __nv_bfloat16     (default __nv_bfloat16)
//     HARNESS_N       problem size              (default 4096)
//     HARNESS_TOL     abs error tolerance       (default: 0.01 fp32, 0.2 bf16)
//     HARNESS_ITERS   timed iterations          (default 10)
//
// Notes on the measurement, since these matter when you compare rungs:
//   - Timed with CUDA events, not std::chrono, over HARNESS_ITERS iterations after two
//     warmups. A single chrono-timed iteration is mostly launch overhead and clock ramp.
//   - The reference GEMM runs on the GPU (from ../common.cuh), not the CPU. A 4096^3 CPU
//     reference takes minutes.
//   - No L2 flush between iterations. For a 4096^3 GEMM the working set is 100MB against
//     a 50MB L2, so this is a minor effect, but it does flatter the memory-bound rungs.
// =====================================================================================

#ifndef PRACTICE_HARNESS_INCLUDED
#define PRACTICE_HARNESS_INCLUDED

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../gemm/common.cuh"

#ifndef HARNESS_DTYPE
#define HARNESS_DTYPE __nv_bfloat16
#endif
#ifndef HARNESS_N
#define HARNESS_N 4096
#endif
#ifndef HARNESS_ITERS
#define HARNESS_ITERS 10
#endif
using htype = HARNESS_DTYPE;

// bf16 has ~3 decimal digits of mantissa, so a 4096-long dot product accumulates real
// error against the reference. fp32 should be far tighter.
#ifdef HARNESS_TOL
static constexpr double kTol = HARNESS_TOL;
#else
static constexpr double kTol = std::is_same_v<htype, float> ? 0.01 : 0.2;
#endif

// H100 SXM dense peaks, for the "% of peak" line.
static constexpr double PEAK_TFLOPS_BF16_TENSOR = 989.4;
static constexpr double PEAK_TFLOPS_FP32_CUDA   = 67.0;
static constexpr double PEAK_HBM_TBS            = 3.35;

static int run_benchmark(size_t M, size_t N, size_t K) {
    const bool is_fp32 = std::is_same_v<htype, float>;
    std::cout << "-------------------- M=" << M << " N=" << N << " K=" << K
              << "  dtype=" << (is_fp32 ? "fp32" : "bf16") << " --------------------\n";

    htype *d_A, *d_B, *d_C, *d_C_ref;
    CUDACHECK(cudaMalloc(&d_A,     M * K * sizeof(htype)));
    CUDACHECK(cudaMalloc(&d_B,     K * N * sizeof(htype)));
    CUDACHECK(cudaMalloc(&d_C,     M * N * sizeof(htype)));
    CUDACHECK(cudaMalloc(&d_C_ref, M * N * sizeof(htype)));

    fill<htype, FillMode::RANDOM>(d_A, M * K, /*seed=*/42, -0.5f, 0.5f);
    fill<htype, FillMode::RANDOM>(d_B, K * N, /*seed=*/43, -0.5f, 0.5f);
    fill<htype, FillMode::CONSTANT>(d_C, M * N, 0.0f);
    fill<htype, FillMode::CONSTANT>(d_C_ref, M * N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    reference_gemm<htype, htype, /*transpose_b=*/false>(d_C_ref, d_A, d_B, M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    for (int i = 0; i < 2; i++) matmul(d_A, d_B, d_C, (int)N);   // warmup
    CUDACHECK(cudaDeviceSynchronize());
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "launch failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        return -1;
    }

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < HARNESS_ITERS; i++) matmul(d_A, d_B, d_C, (int)N);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
    const double us     = ms * 1e3 / HARNESS_ITERS;
    const double flops  = 2.0 * double(M) * double(N) * double(K);
    const double tflops = (flops / us) / 1e6;
    const double peak   = is_fp32 ? PEAK_TFLOPS_FP32_CUDA : PEAK_TFLOPS_BF16_TENSOR;

    // Correctness. Compare on host against the GPU reference.
    std::vector<htype> h_C(M * N), h_C_ref(M * N);
    CUDACHECK(cudaMemcpy(h_C.data(),     d_C,     M * N * sizeof(htype), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(htype), cudaMemcpyDeviceToHost));

    double err_max = 0.0, err_sum = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < M * N; i++) {
        const float v = kittens::base_types::convertor<float, htype>::convert(h_C[i]);
        const float r = kittens::base_types::convertor<float, htype>::convert(h_C_ref[i]);
        const double e = std::abs(double(v) - double(r));
        err_sum += e;
        if (e > err_max) err_max = e;
        if (e > kTol) {
            if (bad < 10)
                std::cout << "  mismatch at (" << i / N << "," << i % N << "): "
                          << v << " vs " << r << " (ref)\n";
            bad++;
        }
    }

    std::cout << std::fixed << std::setprecision(2)
              << "time      : " << us / 1000 << " ms\n"
              << "throughput: " << tflops << " TFLOPs  ("
              << (100.0 * tflops / peak) << "% of " << peak << " peak)\n"
              << std::setprecision(6)
              << "err mean  : " << (err_sum / double(M * N)) << "\n"
              << "err max   : " << err_max << "\n"
              << (bad == 0 ? "RESULT    : PASS\n"
                           : "RESULT    : FAIL (" + std::to_string(bad) + " elements over tol)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    return bad == 0 ? 0 : 1;
}

int main() {
    return run_benchmark(HARNESS_N, HARNESS_N, HARNESS_N);
}

#endif // PRACTICE_HARNESS_INCLUDED
