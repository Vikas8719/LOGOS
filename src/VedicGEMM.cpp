

#ifndef LOGOS_VEDIC_GEMM_CPP
#define LOGOS_VEDIC_GEMM_CPP

#include "../include/Tensor.hpp"
#include <cstring>
#include <string>
#include <stdexcept>


static constexpr int BLOCK = 64;

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

    
    for (int ib = 0; ib < M; ib += BLOCK) {
        int iEnd = std::min(ib + BLOCK, M);

        for (int kb = 0; kb < K; kb += BLOCK) {
            int kEnd = std::min(kb + BLOCK, K);

            for (int jb = 0; jb < N; jb += BLOCK) {
                int jEnd = std::min(jb + BLOCK, N);

                
                for (int i = ib; i < iEnd; ++i) {
                    for (int k = kb; k < kEnd; ++k) {
                        float a_ik = a[i * K + k];  
                        
                        for (int j = jb; j < jEnd; ++j) {
                            c[i * N + j] += a_ik * b[k * N + j];
                        }
                    }
                }
                
            }
        }
    }
    return C;
}


Tensor vedic_gemm_bias(const Tensor& A, const Tensor& W, const Tensor& bias) {
    Tensor C = vedic_gemm(A, W);
    int M = C.rows(), N = C.cols();
    if (bias.total_size != N || bias.ndim() != 1)
        throw std::invalid_argument("Bias must be a 1-D tensor of size " +
                                    std::to_string(N) + ", got shape " +
                                    [&]{ std::string s="(";
                                         for (int i=0;i<(int)bias.shape.size();++i)
                                             s+=std::to_string(bias.shape[i])+(i+1<(int)bias.shape.size()?",":"");
                                         return s+")"; }());
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C.at(i,j) += bias[j];
    return C;
}


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
