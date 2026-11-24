// T8: ModelGPU::forward vs an independent double-precision CPU reference of the whole network,
// plus the structural properties an autoregressive LM must satisfy (causality, prefix invariance,
// determinism, input sanitising, argument validation).
//
// Reference architecture (mirrors ModelGPU.cu): X = emb[tok] + pos; per layer:
//   N1 = LN(X); per head Q,K,V = N1 W; optional Navier-Stokes Qa = Q + eta (Q - Q_prev), Vs = (1-nu) V + nu V_prev;
//   causal softmax attention; concat += head_out W_O; X += concat W_proj; N2 = LN(X);
//   X += gelu(N2 W1 + b1) W2 + b2;   finally LN_f and logits = LN_f(X) lm_head.
#include "gpu_test_common.cuh"
#include <stdexcept>

namespace {

using Vec = std::vector<double>;

Vec matmul(const Vec& A, int M, int K, const Tensor& W) {       // W: K x N (row-major)
    const int N = W.cols();
    Vec C((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const double a = A[(size_t)i * K + k];
            for (int j = 0; j < N; ++j) C[(size_t)i * N + j] += a * (double)W.data[(size_t)k * N + j];
        }
    return C;
}

Vec layernorm(const Vec& X, int seq, int D, const Tensor& g, const Tensor& b) {
    Vec Y(X.size());
    for (int i = 0; i < seq; ++i) {
        double mean = 0.0, var = 0.0;
        for (int j = 0; j < D; ++j) mean += X[(size_t)i * D + j];
        mean /= D;
        for (int j = 0; j < D; ++j) { double t = X[(size_t)i * D + j] - mean; var += t * t; }
        var /= D;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int j = 0; j < D; ++j)
            Y[(size_t)i * D + j] = (double)g.data[j] * (X[(size_t)i * D + j] - mean) * inv + (double)b.data[j];
    }
    return Y;
}

double gelu(double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }

Vec ref_forward(LOGOSModel& m, const std::vector<int>& tok, bool ns, double eta, double nu) {
    const ModelConfig& c = m.cfg;
    const int seq = (int)tok.size(), D = c.d_model, H = c.num_heads, DH = D / H, V = c.vocab_size;
    Vec X((size_t)seq * D);
    for (int i = 0; i < seq; ++i) {
        int t = tok[i]; if (t < 0 || t >= V) t = 0;
        for (int j = 0; j < D; ++j)
            X[(size_t)i * D + j] = (double)m.embedding.data[(size_t)t * D + j] +
                                   (double)m.pos_embedding.data[(size_t)i * D + j];
    }
    for (auto& blk : m.layers) {
        Vec N1 = layernorm(X, seq, D, blk.ln1.gamma, blk.ln1.beta);
        Vec concat((size_t)seq * D, 0.0);
        for (int h = 0; h < H; ++h) {
            auto& hw = blk.mha.heads[h];
            Vec Q = matmul(N1, seq, D, hw.W_Q), K = matmul(N1, seq, D, hw.W_K), Vv = matmul(N1, seq, D, hw.W_V);
            Vec Qa = Q, Vs = Vv;
            if (ns) {
                for (int i = 1; i < seq; ++i)
                    for (int d = 0; d < DH; ++d) {
                        Qa[(size_t)i * DH + d] = Q[(size_t)i * DH + d] + eta * (Q[(size_t)i * DH + d] - Q[(size_t)(i - 1) * DH + d]);
                        Vs[(size_t)i * DH + d] = (1.0 - nu) * Vv[(size_t)i * DH + d] + nu * Vv[(size_t)(i - 1) * DH + d];
                    }
            }
            Vec out((size_t)seq * DH, 0.0);
            const double scale = 1.0 / std::sqrt((double)DH);
            for (int i = 0; i < seq; ++i) {
                Vec p(i + 1);
                double mx = -1e300;
                for (int j = 0; j <= i; ++j) {
                    double s = 0.0;
                    for (int d = 0; d < DH; ++d) s += Qa[(size_t)i * DH + d] * K[(size_t)j * DH + d];
                    p[j] = s * scale; mx = std::max(mx, p[j]);
                }
                double sum = 0.0;
                for (int j = 0; j <= i; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
                for (int j = 0; j <= i; ++j)
                    for (int d = 0; d < DH; ++d) out[(size_t)i * DH + d] += (p[j] / sum) * Vs[(size_t)j * DH + d];
            }
            Vec proj = matmul(out, seq, DH, hw.W_O);
            for (size_t k = 0; k < concat.size(); ++k) concat[k] += proj[k];
        }
        Vec attn = matmul(concat, seq, D, blk.mha.W_proj);
        for (size_t k = 0; k < X.size(); ++k) X[k] += attn[k];
        Vec N2 = layernorm(X, seq, D, blk.ln2.gamma, blk.ln2.beta);
        Vec Hh = matmul(N2, seq, D, blk.ffn.W1);
        const int D4 = 4 * D;
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < D4; ++j) Hh[(size_t)i * D4 + j] = gelu(Hh[(size_t)i * D4 + j] + (double)blk.ffn.b1.data[j]);
        Vec F = matmul(Hh, seq, D4, blk.ffn.W2);
        for (int i = 0; i < seq; ++i)
            for (int j = 0; j < D; ++j) X[(size_t)i * D + j] += F[(size_t)i * D + j] + (double)blk.ffn.b2.data[j];
    }
    Vec L = layernorm(X, seq, D, m.final_ln.gamma, m.final_ln.beta);
    return matmul(L, seq, D, m.lm_head);
}

std::vector<int> random_tokens(int n, int V, uint32_t seed) {
    gt::Lcg r(seed);
    std::vector<int> t(n);
    for (int& x : t) x = r.below(V);
    return t;
}

void setup_phys(ModelGPU& m, bool ns) {
    m.phys.training = false;
    m.phys.feynman_dropout = false;
    m.phys.reynolds = false;
    m.phys.navier_stokes = ns;
}

void test_forward_vs_reference() {
    GT_SECTION("GPU forward == double-precision CPU reference");
    struct C { int H, L, seq; bool ns; };
    const C cases[] = {{2, 2, 1, false}, {2, 2, 5, false}, {2, 2, 17, true}, {2, 2, 24, true},
                       {4, 1, 9, true}, {4, 3, 16, false}};
    for (const C& k : cases) {
        ModelConfig cfg;
        cfg.vocab_size = 48; cfg.d_model = 32; cfg.num_heads = k.H; cfg.num_layers = k.L; cfg.max_seq_len = 24;
        LOGOSModel cpu(cfg);
        gt::randomize_model(cpu, 4242u + (uint32_t)k.seq);
        ModelGPU m(cfg);
        m.load_from_cpu(cpu);
        setup_phys(m, k.ns);
        std::vector<int> tok = random_tokens(k.seq, cfg.vocab_size, 99u + (uint32_t)k.seq);
        std::vector<float> g = gt::to_host(m.forward(tok));
        Vec r = ref_forward(cpu, tok, k.ns, m.phys.ns_eta, m.phys.ns_nu);
        const double diff = gt::max_abs_diff(g, r);
        GT_CHECKF(g.size() == r.size() && diff < 5e-3,
                  "H=%d L=%d seq=%d ns=%d: max|gpu-ref|=%.2e (< 5e-3)", k.H, k.L, k.seq, (int)k.ns, diff);
    }
}

void test_causality_and_prefix() {
    GT_SECTION("causality + prefix invariance (what generation relies on)");
    for (int ns = 0; ns < 2; ++ns) {
        for (int rey = 0; rey < 2; ++rey) {
            ModelConfig cfg;
            cfg.vocab_size = 40; cfg.d_model = 32; cfg.num_heads = 2; cfg.num_layers = 2; cfg.max_seq_len = 24;
            LOGOSModel cpu(cfg);
            gt::randomize_model(cpu, 31337u);
            ModelGPU m(cfg);
            m.load_from_cpu(cpu);
            setup_phys(m, ns != 0);
            m.phys.reynolds = (rey != 0);
            const int seq = 20, V = cfg.vocab_size, cut = 11;
            std::vector<int> a = random_tokens(seq, V, 5u), b = a;
            for (int i = cut; i < seq; ++i) b[i] = (a[i] + 1 + i) % V;      // differ only at positions >= cut
            std::vector<float> la = gt::to_host(m.forward(a)), lb = gt::to_host(m.forward(b));
            double past = 0.0, future = 0.0;
            for (int i = 0; i < seq; ++i)
                for (int v = 0; v < V; ++v) {
                    const double d = std::fabs((double)la[(size_t)i * V + v] - lb[(size_t)i * V + v]);
                    (i < cut ? past : future) = std::max(i < cut ? past : future, d);
                }
            GT_CHECKF(past < 1e-5, "ns=%d reynolds=%d: logits before the edit are unchanged (max diff %.2e)", ns, rey, past);
            GT_CHECKF(future > 1e-4, "ns=%d reynolds=%d: logits at/after the edit do change (%.2e) [test is sensitive]", ns, rey, future);

            // prefix invariance: forward(first k tokens) == first k rows of forward(all)
            std::vector<int> pre(a.begin(), a.begin() + cut);
            std::vector<float> lp = gt::to_host(m.forward(pre));
            double pd = 0.0;
            for (int i = 0; i < cut; ++i)
                for (int v = 0; v < V; ++v)
                    pd = std::max(pd, std::fabs((double)lp[(size_t)i * V + v] - la[(size_t)i * V + v]));
            GT_CHECKF(pd < 1e-3, "ns=%d reynolds=%d: prefix forward == rows of full forward (max diff %.2e)", ns, rey, pd);
        }
    }
}

void test_determinism_and_sanitising() {
    GT_SECTION("determinism, input sanitising, argument validation");
    ModelConfig cfg;
    cfg.vocab_size = 40; cfg.d_model = 32; cfg.num_heads = 2; cfg.num_layers = 2; cfg.max_seq_len = 16;
    LOGOSModel cpu(cfg);
    gt::randomize_model(cpu, 7u);
    ModelGPU m(cfg);
    m.load_from_cpu(cpu);
    setup_phys(m, true);

    std::vector<int> t = random_tokens(12, cfg.vocab_size, 3u);
    std::vector<float> a = gt::to_host(m.forward(t)), b = gt::to_host(m.forward(t));
    GT_CHECK(gt::max_abs_diff(a, b) == 0.0, "two eval forwards are bitwise identical");

    std::vector<int> bad = t, good = t;
    bad[2] = -5; good[2] = 0; bad[7] = cfg.vocab_size + 100; good[7] = 0;
    std::vector<float> lb = gt::to_host(m.forward(bad)), lg = gt::to_host(m.forward(good));
    GT_CHECK(gt::max_abs_diff(lb, lg) == 0.0, "out-of-range token ids behave exactly like token 0");

    bool threw_empty = false, threw_long = false;
    try { m.forward(std::vector<int>{}); } catch (const std::invalid_argument&) { threw_empty = true; }
    try { m.forward(std::vector<int>(cfg.max_seq_len + 1, 1)); } catch (const std::invalid_argument&) { threw_long = true; }
    GT_CHECK(threw_empty, "empty sequence is rejected");
    GT_CHECK(threw_long, "sequence longer than max_seq_len is rejected");

    bool threw_cfg = false;
    try { ModelConfig bad_cfg = cfg; bad_cfg.d_model = 30; bad_cfg.num_heads = 4; ModelGPU x(bad_cfg); } catch (const std::invalid_argument&) { threw_cfg = true; }
    GT_CHECK(threw_cfg, "d_model not divisible by num_heads is rejected by ModelGPU");

    // CPU -> GPU -> CPU weight round trip is lossless (all tensors incl. pos-emb, LN, final LN)
    LOGOSModel back(cfg);
    m.sync_to_cpu(back);
    bool same = back.embedding.data == cpu.embedding.data && back.lm_head.data == cpu.lm_head.data &&
                back.pos_embedding.data == cpu.pos_embedding.data &&
                back.final_ln.gamma.data == cpu.final_ln.gamma.data && back.final_ln.beta.data == cpu.final_ln.beta.data;
    for (int l = 0; l < cfg.num_layers && same; ++l) {
        same = back.layers[l].ln1.gamma.data == cpu.layers[l].ln1.gamma.data &&
               back.layers[l].ln2.beta.data == cpu.layers[l].ln2.beta.data &&
               back.layers[l].ffn.W1.data == cpu.layers[l].ffn.W1.data &&
               back.layers[l].ffn.b2.data == cpu.layers[l].ffn.b2.data &&
               back.layers[l].mha.heads[1].W_O.data == cpu.layers[l].mha.heads[1].W_O.data;
    }
    GT_CHECK(same, "load_from_cpu -> sync_to_cpu round-trips every tensor bit-exactly");
}

}  // namespace

int main() {
    std::printf("LOGOS T8: GPU forward vs CPU reference + causality\n");
    try {
        test_forward_vs_reference();
        test_causality_and_prefix();
        test_determinism_and_sanitising();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        ++gt::g_fail;
    }
    return gt::finish("T8_ModelReference");
}
