// T6: finite-difference gradient check of the FULL GPU model backward pass (run_backward in BackwardGPU.cu).
//
// This is the standard correctness test for hand-written backprop: for sampled entries of EVERY parameter
// tensor, compare the analytic gradient with the central difference (F(w+e)-F(w-e))/(2e) of the mean
// free-energy loss F = CE - T*S. Also checks that gradients ACCUMULATE (beta=1 semantics) and that unused
// embedding rows / future positions receive exactly zero gradient.
//
// Scope notes (documented, not hidden):
//  * dropout OFF (stochastic) and Reynolds norm OFF: Reynolds backward treats the mixing weight w as a
//    constant (detached), so it is an approximation by design and not finite-difference exact. The plain
//    LayerNorm path is checked exactly. Navier-Stokes advect/diffuse is checked both ON and OFF.
//  * float32 forward => central difference with eps=1e-2; tolerance is per tensor (L2 over sampled entries).
#include "gpu_test_common.cuh"

struct Case { const char* name; bool ns; int seq; int heads; int layers; };

static bool run_case(const Case& c) {
    GT_SECTION(c.name);
    ModelConfig cfg;
    cfg.vocab_size = 32; cfg.d_model = 32; cfg.num_heads = c.heads;
    cfg.num_layers = c.layers; cfg.max_seq_len = c.seq + 3;

    LOGOSModel cpu(cfg);
    gt::randomize_model(cpu, 1234u);
    ModelGPU m(cfg);
    m.load_from_cpu(cpu);
    m.phys.training = false;
    m.phys.feynman_dropout = false;
    m.phys.reynolds = false;
    m.phys.navier_stokes = c.ns;

    const int seq = c.seq, V = cfg.vocab_size, D = cfg.d_model;
    gt::Lcg rng(777u);
    std::vector<int> in(seq), tg(seq);
    for (int i = 0; i < seq; ++i) { in[i] = rng.below(V); tg[i] = rng.below(V); }
    const float T = 0.05f;

    std::vector<GPUTensor*> grads = m.alloc_grad_buffers();
    GPUTensor dlog;
    const float F0 = gt::loss_F(m, in, tg, T, &dlog);
    GT_CHECKF(std::isfinite(F0) && F0 > 0.f, "forward loss finite and > 0 (F=%.5f)", F0);

    BackwardScratch scratch;
    run_backward_simple(m, grads, dlog, seq, scratch);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GPUTensor*> params = m.all_parameters();
    GT_CHECK(grads.size() == params.size(), "one gradient buffer per parameter");

    std::vector<std::vector<float>> ga(grads.size());
    bool grads_finite = true;
    for (size_t i = 0; i < grads.size(); ++i) {
        ga[i] = gt::to_host(*grads[i]);
        grads_finite = grads_finite && gt::all_finite(ga[i]);
    }
    GT_CHECK(grads_finite, "all analytic gradients finite");

    // ---- finite-difference comparison, every parameter tensor ----
    const float eps = 1e-2f;
    const int   K   = 6;
    int tensors_checked = 0, tensors_failed = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        GPUTensor& p = *params[i];
        std::vector<float> host = gt::to_host(p);
        const int n = p.size;
        const int ns_samples = std::min(n, K);
        double d2 = 0.0, a2 = 0.0, n2 = 0.0;
        for (int s = 0; s < ns_samples; ++s) {
            int idx;
            if (i == 0 && (s & 1))      idx = in[rng.below(seq)] * D + rng.below(D);     // rows that are actually used
            else if (i == 1)            idx = rng.below(seq) * D + rng.below(D);         // positions that are actually used
            else                        idx = rng.below(n);
            const float orig = host[idx];
            host[idx] = orig + eps; h2d(p, host.data(), n);
            const double Fp = gt::loss_F(m, in, tg, T);
            host[idx] = orig - eps; h2d(p, host.data(), n);
            const double Fm = gt::loss_F(m, in, tg, T);
            host[idx] = orig;       h2d(p, host.data(), n);
            const double num = (Fp - Fm) / (2.0 * eps);
            const double ana = ga[i][idx];
            d2 += (ana - num) * (ana - num); a2 += ana * ana; n2 += num * num;
        }
        const double err = std::sqrt(d2), scale = std::sqrt(a2) + std::sqrt(n2);
        const bool ok = err <= 1e-3 + 0.08 * scale;
        ++tensors_checked;
        if (!ok) {
            ++tensors_failed;
            std::fprintf(stderr, "    MISMATCH %-28s |ana|=%.3e |num|=%.3e |diff|=%.3e\n",
                         gt::param_name((int)i, cfg).c_str(), std::sqrt(a2), std::sqrt(n2), err);
        }
    }
    GT_CHECKF(tensors_failed == 0, "finite-difference grad check: %d/%d tensors match (eps=%.0e)",
              tensors_checked - tensors_failed, tensors_checked, eps);

    // ---- structural zero-gradient properties ----
    {
        std::vector<char> used(V, 0);
        for (int t : in) used[t] = 1;
        double unused_norm = 0.0;
        for (int v = 0; v < V; ++v)
            if (!used[v]) for (int d = 0; d < D; ++d) unused_norm += std::fabs(ga[0][v * D + d]);
        GT_CHECKF(unused_norm == 0.0, "embedding rows of unused tokens get exactly zero grad (%.3e)", unused_norm);

        double future_pos = 0.0;
        for (int r = seq; r < cfg.max_seq_len; ++r)
            for (int d = 0; d < D; ++d) future_pos += std::fabs(ga[1][r * D + d]);
        GT_CHECKF(future_pos == 0.0, "positions >= seq get exactly zero pos-embedding grad (%.3e)", future_pos);
    }

    // ---- accumulation: a 2nd backward pass on the same cache must double every gradient ----
    run_backward_simple(m, grads, dlog, seq, scratch);
    CUDA_CHECK(cudaDeviceSynchronize());
    double worst = 0.0;
    for (size_t i = 0; i < grads.size(); ++i) {
        std::vector<float> g2 = gt::to_host(*grads[i]);
        for (size_t k = 0; k < g2.size(); ++k) {
            const double want = 2.0 * ga[i][k];
            worst = std::max(worst, std::fabs(g2[k] - want) / (1e-6 + std::fabs(want)));
        }
    }
    GT_CHECKF(worst < 1e-3, "gradients accumulate (2 passes == 2x), worst rel dev %.2e", worst);

    for (auto* g : grads) delete g;
    return true;
}

int main() {
    std::printf("LOGOS T6: finite-difference gradient check of the full model backward\n");
    try {
        const Case cases[] = {
            {"NS off, 9 tokens, H=2, L=2",            false,  9, 2, 2},
            {"NS on, 21 tokens (multi flash tile), H=2, L=2", true, 21, 2, 2},
            {"NS on, 10 tokens, H=4, L=1",            true,  10, 4, 1},
        };
        for (const Case& c : cases) run_case(c);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        ++gt::g_fail;
    }
    return gt::finish("T6_GradCheck");
}
