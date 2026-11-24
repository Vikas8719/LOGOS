// Backward pass kernels + run_backward(), extracted verbatim from train_gpu.cu (logic unchanged) so that it
// lives in logos_core and can be unit-tested (see tests/test_gradcheck.cu). One deliberate change:
// softmax_bwd_kernel now uses a block reduction for rowsum(P*dP) instead of every thread looping over the row.
#include "BackwardGPU.cuh"
#include <cmath>

// kernels defined in other translation units of logos_core
extern __global__ void boltzmann_softmax_kernel(const float*, float*, int, int, float);
extern __global__ void causal_mask_kernel(float*, int);

__global__ void embedding_bwd_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ dX,
    float* __restrict__ d_emb, float* __restrict__ d_pos,
    int seq, int d, int vocab_size)
{
    int s = blockIdx.x, di = blockIdx.y * blockDim.x + threadIdx.x;
    if (s >= seq || di >= d) return;
    int tok = token_ids[s];
    if (tok < 0 || tok >= vocab_size) return;
    float g = dX[s * d + di];
    atomicAdd(&d_emb[tok * d + di], g);
    atomicAdd(&d_pos[s * d + di], g);
}

__global__ void vec_add_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] += src[idx];
}

__global__ void scale_grads_kernel(float* grad, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grad[idx] *= scale;
}

__global__ void vec_mul_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] *= src[idx];
}

__global__ void gelu_bwd_kernel(
    const float* __restrict__ ffn_H, const float* __restrict__ d_ffn_A,
    float* __restrict__ d_ffn_H, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float x = ffn_H[idx];
    float k = 0.7978845608f * (x + 0.044715f * x * x * x);
    float th = tanhf(k), sech2 = 1.0f - th * th;
    float gp = 0.5f * (1.0f + th) + 0.5f * x * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
    d_ffn_H[idx] = d_ffn_A[idx] * gp;
}

__global__ void layernorm_bwd_kernel(
    const float* __restrict__ X, const float* __restrict__ gamma,
    const float* __restrict__ dY, float* __restrict__ dX,
    float* __restrict__ d_gamma, float* __restrict__ d_beta,
    int seq, int d, float eps)
{
    int row = blockIdx.x;
    if (row >= seq) return;
    const float* x = X + row * d; const float* dy = dY + row * d; float* dx = dX + row * d;
    __shared__ float smean, sinv_std, s_sum_dy, s_sum_dy_xhat;
    float ts = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) ts += x[j];
    for (int o = 16; o > 0; o >>= 1) ts += __shfl_down_sync(0xffffffff, ts, o);
    __shared__ float sarr[8];
    if (threadIdx.x % 32 == 0) sarr[threadIdx.x / 32] = ts;
    __syncthreads();
    float tot = 0.0f; if (threadIdx.x < 8) tot = sarr[threadIdx.x];
    for (int o = 4; o > 0; o >>= 1) tot += __shfl_down_sync(0xffffffff, tot, o);
    if (threadIdx.x == 0) smean = tot / d; __syncthreads();
    float tv = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) { float diff = x[j] - smean; tv += diff * diff; }
    for (int o = 16; o > 0; o >>= 1) tv += __shfl_down_sync(0xffffffff, tv, o);
    __shared__ float varr[8];
    if (threadIdx.x % 32 == 0) varr[threadIdx.x / 32] = tv;
    __syncthreads();
    float vt = 0.0f; if (threadIdx.x < 8) vt = varr[threadIdx.x];
    for (int o = 4; o > 0; o >>= 1) vt += __shfl_down_sync(0xffffffff, vt, o);
    if (threadIdx.x == 0) sinv_std = rsqrtf(vt / d + eps); __syncthreads();
    float tsdy = 0.0f, tsdyx = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xhat = (x[j] - smean) * sinv_std, dys = dy[j] * gamma[j];
        atomicAdd(&d_gamma[j], dy[j] * xhat); atomicAdd(&d_beta[j], dy[j]);
        tsdy += dys; tsdyx += dys * xhat;
    }
    for (int o = 16; o > 0; o >>= 1) { tsdy += __shfl_down_sync(0xffffffff, tsdy, o); tsdyx += __shfl_down_sync(0xffffffff, tsdyx, o); }
    __shared__ float sda[8], sdxa[8];
    if (threadIdx.x % 32 == 0) { sda[threadIdx.x / 32] = tsdy; sdxa[threadIdx.x / 32] = tsdyx; }
    __syncthreads();
    float tsd = 0.0f, tsdx = 0.0f;
    if (threadIdx.x < 8) { tsd = sda[threadIdx.x]; tsdx = sdxa[threadIdx.x]; }
    for (int o = 4; o > 0; o >>= 1) { tsd += __shfl_down_sync(0xffffffff, tsd, o); tsdx += __shfl_down_sync(0xffffffff, tsdx, o); }
    if (threadIdx.x == 0) { s_sum_dy = tsd; s_sum_dy_xhat = tsdx; } __syncthreads();
    float inv_d = 1.0f / d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xhat = (x[j] - smean) * sinv_std, dys = dy[j] * gamma[j];
        dx[j] = sinv_std * inv_d * (d * dys - s_sum_dy - xhat * s_sum_dy_xhat);
    }
}

__global__ void softmax_bwd_kernel(
    const float* __restrict__ attn_probs, const float* __restrict__ d_attn_probs,
    float* __restrict__ d_scores, int seq, float inv_sqrt_DH)
{
    int row = blockIdx.x;
    if (row >= seq) return;
    const float* p  = attn_probs   + (size_t)row * seq;
    const float* dp = d_attn_probs + (size_t)row * seq;
    float*       ds = d_scores     + (size_t)row * seq;

    // rowsum(P * dP): block reduction (previously every thread looped over the whole row)
    float part = 0.0f;
    for (int j = threadIdx.x; j < seq; j += blockDim.x) part += p[j] * dp[j];
    for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffff, part, o);
    __shared__ float warp_sums[32];
    __shared__ float s_dot;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp_sums[wid] = part;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < ((int)blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffff, t, o);
        if (lane == 0) s_dot = t;
    }
    __syncthreads();
    const float dot = s_dot;
    for (int j = threadIdx.x; j < seq; j += blockDim.x)
        ds[j] = p[j] * (dp[j] - dot) * inv_sqrt_DH;
}

void HeadBwdBuffers::ensure(int seq_, int DH_) {
    if (allocated && alloc_seq == seq_ && alloc_DH == DH_) return;
    // GPUTensor move-assignment frees the previous buffer (RAII) — no manual cudaFree.
    d_head_out       = gpu_alloc(seq_, DH_);
    d_attn_probs_buf = gpu_alloc(seq_, seq_);
    dV_s             = gpu_alloc(seq_, DH_);
    dV               = gpu_alloc(seq_, DH_);
    d_attn_probs     = gpu_alloc(seq_, seq_);
    d_scores         = gpu_alloc(seq_, seq_);
    dQ_adv           = gpu_alloc(seq_, DH_);
    dQ               = gpu_alloc(seq_, DH_);
    dK               = gpu_alloc(seq_, DH_);
    allocated = true; alloc_seq = seq_; alloc_DH = DH_;
}

void LayerBwdBuffers::ensure(int seq_, int D_) {
    if (allocated && alloc_seq == seq_ && alloc_D == D_) return;
    d_concat  = gpu_alloc(seq_, D_);
    d_dX_ln1  = gpu_alloc(seq_, D_);
    allocated = true; alloc_seq = seq_; alloc_D = D_;
}

HeadBwdBuffers  s_head_bwd;
LayerBwdBuffers s_layer_bwd;
HeadBwdBuffers  s_head_bwd_gpu1;
LayerBwdBuffers s_layer_bwd_gpu1;

void run_backward(
    ModelGPU& gpu_model, const ModelConfig& cfg,
    const std::vector<GPUTensor*>& gpu_grads,
    GPUTensor& d_logits_grad, GPUTensor& d_dX_out,
    GPUTensor& d_d_ffn_out, GPUTensor& d_d_ffn_A,
    GPUTensor& d_d_ffn_H,   GPUTensor& d_d_normed2,
    GPUTensor& d_dX_ln,     GPUTensor& d_dX_attn_in,
    int seq,
    HeadBwdBuffers* head_bwd_override,
    LayerBwdBuffers* layer_bwd_override)
{
    int D = cfg.d_model, V = cfg.vocab_size, D4 = 4 * D;
    int H = cfg.num_heads, DH = D / H;
    const int PPL = H * 4 + 9;
    const int wproj_off = H * 4, w1_off = H * 4 + 1, w2_off = H * 4 + 3;
    const int ln1g_off = H * 4 + 5, ln1b_off = H * 4 + 6;
    const int ln2g_off = H * 4 + 7, ln2b_off = H * 4 + 8;

    HeadBwdBuffers&  head_bwd  = head_bwd_override  ? *head_bwd_override  : s_head_bwd;
    LayerBwdBuffers& layer_bwd = layer_bwd_override ? *layer_bwd_override : s_layer_bwd;

    // dW_lm[D x V] += X^T[D x seq] . dY[seq x V]
    cuda_gemm_rm(true, false, D, V, seq,
                 1.f, gpu_model.last_hidden.data, D,
                      d_logits_grad.data,         V,
                 1.f, gpu_grads[2]->data,         V);

    CUDA_CHECK(cudaMemset(d_dX_out.data, 0, (size_t)seq * D * sizeof(float)));
    // dX[seq x D] = dY[seq x V] . W_lm^T   (W_lm: D x V)
    cuda_gemm_rm(false, true, seq, D, V,
                 1.f, d_logits_grad.data,            V,
                      gpu_model.gpu_lm_head.data,    V,
                 0.f, d_dX_out.data,                 D);

    // [FINAL-LN] d_dX_out is the grad wrt the final-LN OUTPUT; LN backward gives the residual-stream grad.
    {
        const int lnf_off = 3 + cfg.num_layers * PPL;   // ln_f_gamma grad slot; beta = +1
        layernorm_bwd_kernel<<<seq, 256>>>(
            gpu_model.final_in.data, gpu_model.ln_f_gamma.data,
            d_dX_out.data, d_dX_ln.data,
            gpu_grads[lnf_off]->data, gpu_grads[lnf_off + 1]->data,
            seq, D, 1e-5f);
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(d_dX_out.data, d_dX_ln.data,
                              (size_t)seq * D * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    for (int l = cfg.num_layers - 1; l >= 0; --l) {
        auto& blk = gpu_model.gpu_blocks[l];
        auto& cache = gpu_model.layer_cache[l];
        int base = 3 + l * PPL;

        CUDA_CHECK(cudaMemcpy(d_d_ffn_out.data, d_dX_out.data,
                   (size_t)seq * D * sizeof(float), cudaMemcpyDeviceToDevice));
        // dW2[D4 x D] += ffn_A^T[D4 x seq] . dOut[seq x D]
        cuda_gemm_rm(true, false, D4, D, seq,
                     1.f, cache.ffn_A.data,                D4,
                          d_d_ffn_out.data,                D,
                     1.f, gpu_grads[base + w2_off]->data,  D);
        // db2[D] += sum_seq dOut
        cuda_colsum_accumulate(d_d_ffn_out.data, gpu_grads[base + w2_off + 1]->data, seq, D);
        // dA[seq x D4] = dOut[seq x D] . W2^T   (W2: D4 x D)
        cuda_gemm_rm(false, true, seq, D4, D,
                     1.f, d_d_ffn_out.data,  D,
                          blk.W2.data,       D,
                     0.f, d_d_ffn_A.data,    D4);
        { int tot = seq * D4;
          gelu_bwd_kernel<<<(tot + 255) / 256, 256>>>(cache.ffn_H.data, d_d_ffn_A.data,
                                                      d_d_ffn_H.data, tot); CUDA_KERNEL_CHECK(); }

        if (gpu_model.phys.feynman_dropout && gpu_model.phys.training
            && cache.ffn_mask.valid()) {
            int tot = seq * D4;
            vec_mul_kernel<<<(tot + 255) / 256, 256>>>(d_d_ffn_H.data,
                                                       cache.ffn_mask.data, tot);
            CUDA_KERNEL_CHECK();
        }

        // dW1[D x D4] += normed2^T[D x seq] . dH[seq x D4]
        cuda_gemm_rm(true, false, D, D4, seq,
                     1.f, cache.normed2.data,               D,
                          d_d_ffn_H.data,                   D4,
                     1.f, gpu_grads[base + w1_off]->data,   D4);
        // db1[D4] += sum_seq dH
        cuda_colsum_accumulate(d_d_ffn_H.data, gpu_grads[base + w1_off + 1]->data, seq, D4);
        // dN2[seq x D] = dH[seq x D4] . W1^T   (W1: D x D4)
        cuda_gemm_rm(false, true, seq, D, D4,
                     1.f, d_d_ffn_H.data,   D4,
                          blk.W1.data,      D4,
                     0.f, d_d_normed2.data, D);

        if (gpu_model.phys.reynolds && cache.ln2_w.valid()) {
            reynolds_bwd_kernel<<<seq, 256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, cache.ln2_w.data,
                gpu_model.run_mean[2 * l + 1].data, gpu_model.run_var[2 * l + 1].data,
                d_dX_ln.data,
                gpu_grads[base + ln2g_off]->data, gpu_grads[base + ln2b_off]->data,
                seq, D, 1e-5f);
        } else {
            layernorm_bwd_kernel<<<seq, 256>>>(
                cache.post_attn.data, blk.ln2_gamma.data,
                d_d_normed2.data, d_dX_ln.data,
                gpu_grads[base + ln2g_off]->data, gpu_grads[base + ln2b_off]->data,
                seq, D, 1e-5f);
        }
        CUDA_KERNEL_CHECK();
        { int sz = seq * D;
          vec_add_kernel<<<(sz + 255) / 256, 256>>>(d_dX_out.data, d_dX_ln.data, sz); CUDA_KERNEL_CHECK(); }

        // ensure() must run BEFORE d_concat is used
        head_bwd.ensure(seq, DH);
        layer_bwd.ensure(seq, D);
        GPUTensor& d_concat = layer_bwd.d_concat;
        // dWproj[D x D] += concat^T[D x seq] . dMha[seq x D]
        cuda_gemm_rm(true, false, D, D, seq,
                     1.f, cache.concat.data,                 D,
                          d_dX_out.data,                     D,
                     1.f, gpu_grads[base + wproj_off]->data, D);
        // dConcat[seq x D] = dMha[seq x D] . Wproj^T
        cuda_gemm_rm(false, true, seq, D, D,
                     1.f, d_dX_out.data,      D,
                          blk.W_proj.data,    D,
                     0.f, d_concat.data,      D);

        CUDA_CHECK(cudaMemset(d_dX_attn_in.data, 0, (size_t)seq * D * sizeof(float)));
        for (int h = 0; h < H; ++h) {
            auto& hc = cache.heads[h];
            // dHeadOut[seq x DH] = dConcat[seq x D] . W_O^T   (W_O: DH x D)
            cuda_gemm_rm(false, true, seq, DH, D,
                         1.f, d_concat.data,                D,
                              blk.W_O[h].data,              D,
                         0.f, head_bwd.d_head_out.data,     DH);
            // dW_O[DH x D] += head_out^T[DH x seq] . dConcat[seq x D]
            cuda_gemm_rm(true, false, DH, D, seq,
                         1.f, hc.head_out.data,             DH,
                              d_concat.data,                D,
                         1.f, gpu_grads[base + h * 4 + 3]->data, D);

            // recompute attention probabilities (flash forward does not keep them)
            {
                float scale_attn = 1.0f / sqrtf((float)DH);
                cuda_gemm_rm(false, true, seq, seq, DH,
                             scale_attn, hc.Q_adv.data, DH,
                                         hc.K.data,     DH,
                             0.f, head_bwd.d_attn_probs_buf.data, seq);
                causal_mask_kernel<<<dim3(seq, (seq + 255) / 256), 256>>>(
                    head_bwd.d_attn_probs_buf.data, seq);
                CUDA_KERNEL_CHECK();
                boltzmann_softmax_kernel<<<seq, 256>>>(
                    head_bwd.d_attn_probs_buf.data,
                    head_bwd.d_attn_probs_buf.data,
                    seq, seq, 1.0f);
                CUDA_KERNEL_CHECK();
            }

            // dV_s[seq x DH] = P^T[seq x seq] . dHeadOut[seq x DH]
            cuda_gemm_rm(true, false, seq, DH, seq,
                         1.f, head_bwd.d_attn_probs_buf.data, seq,
                              head_bwd.d_head_out.data,       DH,
                         0.f, head_bwd.dV_s.data,             DH);

            { int qb = (seq * DH + 255) / 256;
              ns_diffuse_bwd_kernel<<<qb, 256>>>(head_bwd.dV_s.data, head_bwd.dV.data, seq, DH,
                                                 gpu_model.phys.ns_nu); CUDA_KERNEL_CHECK(); }

            // dP[seq x seq] = dHeadOut[seq x DH] . V_s^T   (V_s: seq x DH)
            cuda_gemm_rm(false, true, seq, seq, DH,
                         1.f, head_bwd.d_head_out.data,   DH,
                              hc.V_s.data,                DH,
                         0.f, head_bwd.d_attn_probs.data, seq);

            float inv_sqrt_DH = 1.0f / sqrtf((float)DH);
            softmax_bwd_kernel<<<seq, 256>>>(head_bwd.d_attn_probs_buf.data,
                                             head_bwd.d_attn_probs.data,
                                             head_bwd.d_scores.data,
                                             seq, inv_sqrt_DH); CUDA_KERNEL_CHECK();

            // dQ_adv[seq x DH] = dS[seq x seq] . K[seq x DH]
            cuda_gemm_rm(false, false, seq, DH, seq,
                         1.f, head_bwd.d_scores.data, seq,
                              hc.K.data,              DH,
                         0.f, head_bwd.dQ_adv.data,   DH);

            { int qb = (seq * DH + 255) / 256;
              ns_advect_bwd_kernel<<<qb, 256>>>(head_bwd.dQ_adv.data, head_bwd.dQ.data, seq, DH,
                                                gpu_model.phys.ns_eta); CUDA_KERNEL_CHECK(); }

            // dK[seq x DH] = dS^T[seq x seq] . Q_adv[seq x DH]
            cuda_gemm_rm(true, false, seq, DH, seq,
                         1.f, head_bwd.d_scores.data, seq,
                              hc.Q_adv.data,          DH,
                         0.f, head_bwd.dK.data,       DH);

            {
                // t = 0:Q  1:K  2:V  (grad slot base+h*4+t, same order as all_parameters())
                const float* dQKV[3] = { head_bwd.dQ.data, head_bwd.dK.data, head_bwd.dV.data };
                const float* Wqkv[3] = { blk.W_Q[h].data,  blk.W_K[h].data,  blk.W_V[h].data  };
                for (int t = 0; t < 3; ++t) {
                    // dW[D x DH] += normed1^T[D x seq] . dQKV[seq x DH]
                    cuda_gemm_rm(true, false, D, DH, seq,
                                 1.f, cache.normed1.data,           D,
                                      dQKV[t],                      DH,
                                 1.f, gpu_grads[base + h * 4 + t]->data, DH);
                    // dN1[seq x D] += dQKV[seq x DH] . W^T   (W: D x DH) — heads/QKV summed with beta=1
                    cuda_gemm_rm(false, true, seq, D, DH,
                                 1.f, dQKV[t],           DH,
                                      Wqkv[t],           DH,
                                 1.f, d_dX_attn_in.data, D);
                }
            }
        }

        GPUTensor& d_dX_ln1 = layer_bwd.d_dX_ln1;
        if (gpu_model.phys.reynolds && cache.ln1_w.valid()) {
            reynolds_bwd_kernel<<<seq, 256>>>(
                cache.block_input.data, blk.ln1_gamma.data,
                d_dX_attn_in.data, cache.ln1_w.data,
                gpu_model.run_mean[2 * l].data, gpu_model.run_var[2 * l].data,
                d_dX_ln1.data,
                gpu_grads[base + ln1g_off]->data, gpu_grads[base + ln1b_off]->data,
                seq, D, 1e-5f);
        } else {
            layernorm_bwd_kernel<<<seq, 256>>>(
                cache.block_input.data, blk.ln1_gamma.data,
                d_dX_attn_in.data, d_dX_ln1.data,
                gpu_grads[base + ln1g_off]->data, gpu_grads[base + ln1b_off]->data,
                seq, D, 1e-5f);
        }
        CUDA_KERNEL_CHECK();
        { int sz = seq * D;
          vec_add_kernel<<<(sz + 255) / 256, 256>>>(d_dX_out.data, d_dX_ln1.data, sz); CUDA_KERNEL_CHECK(); }
    }

    const bool hyper_chain = gpu_model.hyper_cfg.enabled && gpu_model.X_euclidean.valid();
    if (!hyper_chain) {
        dim3 blk(32), grd(seq, (D + 31) / 32);
        embedding_bwd_kernel<<<grd, blk>>>(
            gpu_model.d_token_ids, d_dX_out.data,
            gpu_grads[0]->data, gpu_grads[1]->data,
            seq, D, V); CUDA_KERNEL_CHECK();
    }

    if (hyper_chain) {
        GPUTensor d_dX_euc = gpu_alloc(seq, D);
        expmap0_bwd_kernel<<<seq, 256>>>(
            gpu_model.X_euclidean.data,
            d_dX_out.data,
            d_dX_euc.data,
            seq, D, gpu_model.hyper_cfg.curvature);
        CUDA_KERNEL_CHECK();
        { dim3 blk2(32), grd2(seq, (D + 31) / 32);
          embedding_bwd_kernel<<<grd2, blk2>>>(
              gpu_model.d_token_ids, d_dX_euc.data,
              gpu_grads[0]->data, gpu_grads[1]->data,
              seq, D, V); CUDA_KERNEL_CHECK(); }
    }
}

void run_backward_simple(ModelGPU& gpu_model, const std::vector<GPUTensor*>& gpu_grads,
                         GPUTensor& d_logits_grad, int seq, BackwardScratch& scratch,
                         HeadBwdBuffers* head_bwd_override, LayerBwdBuffers* layer_bwd_override)
{
    scratch.ensure(seq, gpu_model.cfg.d_model);
    run_backward(gpu_model, gpu_model.cfg, gpu_grads, d_logits_grad,
                 scratch.d_dX_out, scratch.d_d_ffn_out, scratch.d_d_ffn_A, scratch.d_d_ffn_H,
                 scratch.d_d_normed2, scratch.d_dX_ln, scratch.d_dX_attn_in, seq,
                 head_bwd_override, layer_bwd_override);
}
