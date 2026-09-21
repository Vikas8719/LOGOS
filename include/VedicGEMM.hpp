#pragma once
// ============================================================
//  LOGOS — VedicGEMM.hpp
//  Urdhva Tiryagbhyam tiled GEMM — header-only
//  #pragma once ensures no multiple-definition errors
// ============================================================
#include "Tensor.hpp"
#include <stdexcept>
#include <string>

static constexpr int VEDIC_BLOCK = 64;   // Renamed: no collision

inline Tensor reference_gemm(const Tensor& A, const Tensor& B) {
    int M = A.rows(), K = A.cols(), N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument("GEMM dimension mismatch");
    Tensor C({M, N}, 0.0f);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C.at(i,j) += A.at(i,k) * B.at(k,j);
    return C;
}

inline Tensor vedic_gemm(const Tensor& A, const Tensor& B) {
    int M = A.rows(), K = A.cols(), N = B.cols();
    if (B.rows() != K)
        throw std::invalid_argument(
            "VedicGEMM: A(" + std::to_string(M) + "x" + std::to_string(K) +
            ") x B(" + std::to_string(B.rows()) + "x" + std::to_string(N) + ") mismatch");

    Tensor C({M, N}, 0.0f);
    const float* a = A.data.data();
    const float* b = B.data.data();
    float*       c = C.data.data();

    for (int ib = 0; ib < M; ib += VEDIC_BLOCK) {
        int iEnd = std::min(ib + VEDIC_BLOCK, M);
        for (int kb = 0; kb < K; kb += VEDIC_BLOCK) {
            int kEnd = std::min(kb + VEDIC_BLOCK, K);
            for (int jb = 0; jb < N; jb += VEDIC_BLOCK) {
                int jEnd = std::min(jb + VEDIC_BLOCK, N);
                for (int i = ib; i < iEnd; ++i) {
                    for (int k = kb; k < kEnd; ++k) {
                        float a_ik = a[i * K + k];
                        for (int j = jb; j < jEnd; ++j)
                            c[i * N + j] += a_ik * b[k * N + j];
                    }
                }
            }
        }
    }
    return C;
}

inline Tensor vedic_gemm_bias(const Tensor& A, const Tensor& W, const Tensor& bias) {
    Tensor C = vedic_gemm(A, W);
    int M = C.rows(), N = C.cols();
    if (bias.total_size != N)
        throw std::invalid_argument("Bias size mismatch");
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C.at(i,j) += bias[j];
    return C;
}
