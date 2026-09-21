// ============================================================
//  LOGOS — VedicGEMM.cpp
//  Urdhva Tiryagbhyam (Vertically & Crosswise) Matrix Multiply
//  LLM ka 70-80% computation yahi handle karega
//
//  Algorithm idea:
//  Normal GEMM: C[i][j] += A[i][k] * B[k][j]  (triple loop)
//  Vedic GEMM:  Column partial products compute karo crosswise,
//               accumulate with carry — digit-level parallelism
//               Float version: block-level crosswise accumulation
//               with SIMD-friendly inner loop (compiler auto-vectorizes)
// ============================================================

#ifndef LOGOS_VEDIC_GEMM_CPP
#define LOGOS_VEDIC_GEMM_CPP

#include "../include/Tensor.hpp"
#include <cstring>
#include <string>
#include <stdexcept>

// Block size for cache-tiling (L1 cache ~32KB, floats = 4B)
// 64×64 block = 16KB — fits two blocks in L1
static constexpr int BLOCK = 64;

// ── Reference GEMM (for unit testing / correctness check) ────
Tensor reference_gemm(const Tensor& A, const Tensor& B) {
    int M = A.rows(), K = A.cols(), N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument("GEMM dimension mismatch: A.cols != B.rows");
    Tensor C({M, N}, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C.at(i,j) += A.at(i,k) * B.at(k,j);
    return C;
}

// ── Vedic GEMM Core ──────────────────────────────────────────
// Urdhva Tiryagbhyam principle applied to floating-point blocks:
// Each output tile C[ib..ib+BLOCK][jb..jb+BLOCK] is computed
// by "crosswise" accumulation over K-blocks — same math as
// Vedic partial products, reorganized for cache lines.
//
// Key property: inner loop (j) is contiguous in memory → CPU
// prefetcher fills cache line → near-SIMD speed without explicit
// intrinsics (-O3 -march=native auto-vectorizes this pattern).

Tensor vedic_gemm(const Tensor& A, const Tensor& B) {
    int M = A.rows(), K = A.cols(), N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument("VedicGEMM: A(" + std::to_string(M) + "x" +
            std::to_string(K) + ") × B(" + std::to_string(B.rows()) + "x" +
            std::to_string(N) + ") — cols/rows mismatch");

    Tensor C({M, N}, 0.0f);
    const float* a = A.data.data();
    const float* b = B.data.data();
    float*       c = C.data.data();

    // Tiled triple loop — Urdhva crosswise block accumulation
    for (int ib = 0; ib < M; ib += BLOCK) {
        int iEnd = std::min(ib + BLOCK, M);

        for (int kb = 0; kb < K; kb += BLOCK) {
            int kEnd = std::min(kb + BLOCK, K);

            for (int jb = 0; jb < N; jb += BLOCK) {
                int jEnd = std::min(jb + BLOCK, N);

                // ── Vedic crosswise kernel ─────────────────────
                // For each row i: partial products across k-strip,
                // accumulated into j-strip of C (contiguous → SIMD)
                for (int i = ib; i < iEnd; ++i) {
                    for (int k = kb; k < kEnd; ++k) {
                        float a_ik = a[i * K + k];  // scalar — broadcast
                        // Inner j-loop: contiguous B row → auto-vectorized
                        for (int j = jb; j < jEnd; ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
                // ── End crosswise kernel ───────────────────────
            }
        }
    }
    return C;
}

// ── Fused Vedic GEMM + Bias Add ──────────────────────────────
// Used in Feed-Forward layer: C = A×W + bias
Tensor vedic_gemm_bias(const Tensor& A, const Tensor& W, const Tensor& bias) {
    Tensor C = vedic_gemm(A, W);
    int M = C.rows(), N = C.cols();
    if (bias.total_size != N)
        throw std::invalid_argument("Bias size mismatch");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C.at(i,j) += bias[j];
    return C;
}

// ── Unit Test ─────────────────────────────────────────────────
// Call this to verify Vedic GEMM gives same result as reference
#ifdef LOGOS_TEST_VEDIC
#include <iostream>
#include <cmath>
int main() {
    srand(42);
    Tensor A({64, 128}); A.fill_random(-1.f, 1.f);
    Tensor B({128, 64}); B.fill_random(-1.f, 1.f);

    Tensor C_ref   = reference_gemm(A, B);
    Tensor C_vedic = vedic_gemm(A, B);

    float max_err = 0.0f;
    for (int i = 0; i < C_ref.total_size; ++i)
        max_err = std::max(max_err, std::abs(C_ref[i] - C_vedic[i]));

    std::cout << "Max error (Vedic vs Reference): " << max_err << "\n";
    if (max_err < 1e-4f)
        std::cout << "✅ VedicGEMM PASS — same answer as reference\n";
    else
        std::cout << "❌ VedicGEMM FAIL — check implementation\n";
    return 0;
}
#endif

#endif
