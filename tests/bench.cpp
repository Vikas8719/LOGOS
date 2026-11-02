#include "../include/VedicGEMM.hpp"
#include "../include/Tensor.hpp"
#include "../include/Attention.hpp"
#include "../include/LayerNorm.hpp"
#include "../include/FeedForward.hpp"
#include "../include/PhysicsOpt.hpp"
#include "../include/Model.hpp"
#include "../include/Backprop.hpp"
#include "../include/GradientClip.hpp"
#include "../include/Tokenizer.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <cmath>
#include <sstream>

using Clock     = std::chrono::high_resolution_clock;
using FpSeconds = std::chrono::duration<double>;

struct BenchResult {
    std::string name;
    double      median_ms;
    double      mean_ms;
    double      min_ms;
    double      max_ms;
    double      stddev_ms;
    double      throughput;
    std::string unit;
    bool        passed;
    std::string fail_reason;
};

static std::vector<BenchResult> g_results;
static std::atomic<float> g_benchmark_sink{0.0f};

static BenchResult run_bench(
    const std::string& name,
    int warmup,
    int iters,
    std::function<void()> fn,
    double throughput_num = 0.0,
    const std::string& throughput_unit = "")
{
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[i] = FpSeconds(t1 - t0).count() * 1e3;
    }

    std::sort(times.begin(), times.end());
    double median = times[iters / 2];
    double mean   = std::accumulate(times.begin(), times.end(), 0.0) / iters;
    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / iters);

    double tp = (throughput_num > 0.0 && median > 0.0)
                ? throughput_num / (median * 1e-3)
                : 0.0;

    BenchResult r;
    r.name         = name;
    r.median_ms    = median;
    r.mean_ms      = mean;
    r.min_ms       = times.front();
    r.max_ms       = times.back();
    r.stddev_ms    = stddev;
    r.throughput   = tp;
    r.unit         = throughput_unit;
    r.passed       = true;
    return r;
}

static void print_result(const BenchResult& r) {
    std::cout << std::left  << std::setw(42) << r.name
              << std::right << std::setw(9)  << std::fixed << std::setprecision(3)
              << r.median_ms << " ms"
              << "  (min=" << std::setw(7) << r.min_ms
              << "  max=" << std::setw(7) << r.max_ms
              << "  σ=" << std::setw(6) << r.stddev_ms << ")";
    if (r.throughput > 0.0) {
        if (r.throughput >= 1e9)
            std::cout << "  " << std::fixed << std::setprecision(2)
                      << r.throughput * 1e-9 << " G" << r.unit;
        else if (r.throughput >= 1e6)
            std::cout << "  " << std::setprecision(2)
                      << r.throughput * 1e-6 << " M" << r.unit;
        else
            std::cout << "  " << std::setprecision(2)
                      << r.throughput * 1e-3 << " K" << r.unit;
    }
    if (!r.passed)
        std::cout << "  [FAIL: " << r.fail_reason << "]";
    std::cout << "\n";
}

static BenchResult& add(BenchResult r) {
    g_results.push_back(r);
    print_result(g_results.back());
    return g_results.back();
}

static void section(const std::string& title) {
    int fill = static_cast<int>(60 - title.size() - 4);
    if (fill < 0) fill = 0;
    std::cout << "\n-- " << title << " " << std::string(fill, '-') << "\n";
}

static void threshold_check(BenchResult& r,
                             double max_allowed_ms,
                             const std::string& context = "")
{
    if (r.median_ms > max_allowed_ms) {
        r.passed      = false;
        r.fail_reason = "median " + std::to_string(r.median_ms)
                      + " ms > threshold " + std::to_string(max_allowed_ms) + " ms"
                      + (context.empty() ? "" : " (" + context + ")");
    }
}


static void bench_b1_vedic_gemm() {
    section("B1 · VedicGEMM  (block-tiled GEMM correctness + speed)");

    struct Shape { int M, K, N; };
    std::vector<Shape> shapes = {
        {64,  64,  64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {128, 256, 128},
        {64,  512, 64},
    };

    for (auto [M, K, N] : shapes) {
        Tensor A({M, K}); A.fill_random(-1.f, 1.f, 42);
        Tensor B({K, N}); B.fill_random(-1.f, 1.f, 99);

        double flops = 2.0 * M * K * N;
        std::ostringstream nm;
        nm << "vedic_gemm [" << M << "x" << K << "x" << N << "]";

        auto r = run_bench(nm.str(), 3, 20,
            [&]{ auto C = vedic_gemm(A, B); (void)C; },
            flops, "FLOP/s");

        if (M <= 256) threshold_check(r, 200.0, "small/medium matrix");
        add(r);
    }

    {
        Tensor A({128, 128}); A.fill_random(-1.f, 1.f, 1);
        Tensor B({128, 128}); B.fill_random(-1.f, 1.f, 2);
        Tensor b({128});       b.fill_random(-0.1f, 0.1f, 3);
        auto r = run_bench("vedic_gemm_bias [128x128+b]", 3, 20,
            [&]{ auto C = vedic_gemm_bias(A, B, b); (void)C; },
            2.0 * 128 * 128 * 128, "FLOP/s");
        threshold_check(r, 300.0);
        add(r);
    }

    {
        Tensor A({64, 64}); A.fill_random(-1.f, 1.f, 7);
        Tensor B({64, 64}); B.fill_random(-1.f, 1.f, 8);
        auto C_ved = vedic_gemm(A, B);
        auto C_ref = reference_gemm(A, B);
        float max_err = 0.f;
        for (int i = 0; i < 64 * 64; ++i)
            max_err = std::max(max_err, std::abs(C_ved.data[i] - C_ref.data[i]));

        BenchResult corr;
        corr.name       = "vedic_gemm vs reference_gemm: max_err";
        corr.median_ms  = 0;
        corr.passed     = (max_err < 1e-2f);
        corr.fail_reason = corr.passed ? "" : "max_err=" + std::to_string(max_err);
        corr.throughput = 0;
        std::cout << "  " << (corr.passed ? "[OK]  " : "[FAIL] ")
                  << corr.name << " = " << max_err << "\n";
        g_results.push_back(corr);
    }
}


static void bench_b2_attention() {
    section("B2 · Attention  (boltzmann_softmax, masks, full head)");

    for (int seq : {32, 64, 128, 256}) {
        int d_k = 64;
        Tensor scores({seq, seq}); scores.fill_random(-2.f, 2.f, 11);
        std::ostringstream nm;
        nm << "boltzmann_softmax [" << seq << "x" << seq << "] T=8";
        auto r = run_bench(nm.str(), 3, 30,
            [&]{ auto p = boltzmann_softmax(scores, 8.f); (void)p; });
        threshold_check(r, 100.0, "softmax seq=" + std::to_string(seq));
        add(r);
    }

    for (int seq : {32, 64, 128}) {
        std::ostringstream nm;
        nm << "causal_mask_cached [" << seq << "]";
        auto r = run_bench(nm.str(), 2, 50,
            [&]{ auto& m = causal_mask_cached(seq); (void)m; });
        threshold_check(r, 1.0, "cache hit after first call");
        add(r);
    }

    {
        int seq = 64, d_model = 128;
        AttentionHead head(d_model, 32);
        head.use_sparse = false;
        Tensor X({seq, d_model}); X.fill_random(-0.5f, 0.5f, 55);
        auto r = run_bench("AttentionHead::forward [seq=64,d=128,dk=32]", 3, 10,
            [&]{ auto o = head.forward(X); (void)o; });
        threshold_check(r, 500.0);
        add(r);
    }

    {
        int seq = 32, d_model = 64;
        MultiHeadAttention mha(d_model, 4);
        Tensor X({seq, d_model}); X.fill_random(-0.5f, 0.5f, 77);
        auto r = run_bench("MultiHeadAttention [seq=32,d=64,h=4]", 3, 10,
            [&]{ auto o = mha.forward(X); (void)o; });
        threshold_check(r, 500.0);
        add(r);
    }

    {
        int seq = 32, d_k = 16, d_v = 16;
        Tensor Q({seq, d_k}); Q.fill_random(-0.5f, 0.5f, 1);
        Tensor K({seq, d_k}); K.fill_random(-0.5f, 0.5f, 2);
        Tensor V({seq, d_v}); V.fill_random(-0.5f, 0.5f, 3);
        Tensor mask({seq, seq}, 0.f);
        for (int i = 0; i < seq; ++i)
            for (int j = i + 1; j < seq; ++j) mask.at(i, j) = -1e9f;
        auto r = run_bench("navier_stokes_attention [seq=32,eta=0.1,nu=0.05]", 3, 10,
            [&]{ auto o = navier_stokes_attention(Q, K, V, mask, 4.f, 0.1f, 0.05f); (void)o; });
        threshold_check(r, 500.0);
        add(r);
    }
}


static void bench_b3_layernorm() {
    section("B3 · LayerNorm + ReynoldsBatchNorm");

    for (auto [seq, dim] : std::vector<std::pair<int,int>>{{32,64},{128,256},{256,512}}) {
        LayerNorm ln(dim);
        Tensor X({seq, dim}); X.fill_random(-2.f, 2.f, 33);
        std::ostringstream nm;
        nm << "LayerNorm::forward [seq=" << seq << ",d=" << dim << "]";
        auto r = run_bench(nm.str(), 3, 30,
            [&]{ auto y = ln.forward(X); (void)y; });
        threshold_check(r, 50.0);
        add(r);
    }

    {
        int seq = 64, dim = 128;
        ReynoldsBatchNorm rbn(dim);
        Tensor X({seq, dim}); X.fill_random(-2.f, 2.f, 44);
        auto r = run_bench("ReynoldsBatchNorm::forward [seq=64,d=128,train]", 3, 20,
            [&]{ auto y = rbn.forward(X, true); (void)y; });
        threshold_check(r, 200.0);
        add(r);
    }

    {
        int seq = 64, dim = 128;
        ReynoldsBatchNorm rbn(dim);
        Tensor X({seq, dim}); X.fill_random(-2.f, 2.f, 55);
        auto r = run_bench("ReynoldsBatchNorm::forward [seq=64,d=128,eval]", 3, 20,
            [&]{ auto y = rbn.forward(X, false); (void)y; });
        threshold_check(r, 200.0);
        add(r);
    }

    {
        int seq = 64, dim = 128;
        ReynoldsBatchNorm rbn(dim);
        Tensor X({seq, dim}); X.fill_random(-2.f, 2.f, 66);
        auto r = run_bench("ReynoldsBatchNorm::reynolds_number [seq=64,d=128]", 3, 30,
            [&]{ float re = rbn.reynolds_number(X); (void)re; });
        threshold_check(r, 50.0);
        add(r);
    }
}


static void bench_b4_feedforward() {
    section("B4 · FeedForward + GELU");

    for (auto [seq, d_model] : std::vector<std::pair<int,int>>{{32,64},{64,128},{128,256}}) {
        FeedForward ff(d_model, 0);
        Tensor X({seq, d_model}); X.fill_random(-0.5f, 0.5f, 77);
        std::ostringstream nm;
        nm << "FeedForward::forward [seq=" << seq << ",d=" << d_model << "]";
        auto r = run_bench(nm.str(), 3, 20,
            [&]{ auto y = ff.forward(X, false); (void)y; });
        threshold_check(r, 500.0);
        add(r);
    }

    {
        int N = 1 << 20;
        Tensor input({N, 1}); input.fill_random(-3.f, 3.f, 88);
        Tensor X = input;
        auto r = run_bench("gelu in-place [1M elements]", 3, 10, [&]{
            X.data = input.data;
            for (int i = 0; i < N; ++i) X.data[i] = gelu(X.data[i]);
            g_benchmark_sink.store(X.data[N / 2], std::memory_order_relaxed);
        }, (double)N, "elem/s");
        threshold_check(r, 100.0);
        add(r);
    }
}


static void bench_b5_optimizers() {
    section("B5 · Physics Optimizers (Langevin / HybridSHM / NaturalGrad)");

    auto make_tensors = [](int n_params, int param_size) {
        std::vector<Tensor> Ws, Gs;
        for (int i = 0; i < n_params; ++i) {
            Tensor W({1, param_size}); W.fill_random(-0.1f, 0.1f, i);
            Tensor G({1, param_size}); G.fill(0.01f);
            Ws.push_back(std::move(W));
            Gs.push_back(std::move(G));
        }
        return std::make_pair(std::move(Ws), std::move(Gs));
    };

    auto ptrs = [](std::vector<Tensor>& v) {
        std::vector<Tensor*> p;
        for (auto& t : v) p.push_back(&t);
        return p;
    };

    {
        auto [Ws, Gs] = make_tensors(16, 256);
        auto ps = ptrs(Ws), gs = ptrs(Gs);
        LangevinOptimizer opt(1e-4f, 0.9f, 1e-5f, 1e-8f, 10000);
        opt.init(ps);
        auto r = run_bench("LangevinOptimizer::step [16×256 params]", 3, 50,
            [&]{ opt.step(ps, gs); });
        threshold_check(r, 10.0);
        add(r);
    }

    {
        auto [Ws, Gs] = make_tensors(16, 256);
        auto ps = ptrs(Ws), gs = ptrs(Gs);
        HybridSHMOptimizer opt(1e-4f, 0.1f, 0.9f, 0.5f, 1e-3f, 0.3f, 0.9f, 10000);
        opt.init(ps);
        auto r = run_bench("HybridSHMOptimizer::step [16×256 params]", 3, 50,
            [&]{ opt.step(ps, gs); });
        threshold_check(r, 10.0);
        add(r);
    }

    {
        auto [Ws, Gs] = make_tensors(16, 256);
        auto ps = ptrs(Ws), gs = ptrs(Gs);
        NaturalGradientOptimizer opt(1e-4f, 0.99f, 1e-8f, 0.9f, 0.01f, 1e-3f, 10000);
        opt.init(ps);
        auto r = run_bench("NaturalGradOptimizer::step [16×256 params]", 3, 50,
            [&]{ opt.step(ps, gs); });
        threshold_check(r, 15.0);
        add(r);
    }

    {
        WeightPathIntegral wpi(1.0f, 1000);
        std::vector<float> delta(256 * 16, 0.01f);
        auto r = run_bench("WeightPathIntegral::record_step [4096-dim]", 2, 100,
            [&]{ wpi.record_step(0.5f, delta); });
        threshold_check(r, 5.0);
        add(r);
    }
}


static void bench_b6_gradient_clip() {
    section("B6 · GradientClip");

    auto make_grads = [](int n, int sz) {
        std::vector<Tensor> Gs;
        for (int i = 0; i < n; ++i) {
            Tensor G({1, sz}); G.fill_random(-1.f, 1.f, i + 100);
            Gs.push_back(std::move(G));
        }
        return Gs;
    };

    auto ptrs = [](std::vector<Tensor>& v) {
        std::vector<Tensor*> p;
        for (auto& t : v) p.push_back(&t);
        return p;
    };

    for (auto [n, sz] : std::vector<std::pair<int,int>>{{8,256},{32,1024},{128,512}}) {
        auto Gs = make_grads(n, sz);
        auto gp = ptrs(Gs);
        std::ostringstream nm;
        nm << "clip_gradients [" << n << "×" << sz << " params]";
        auto r = run_bench(nm.str(), 2, 50,
            [&]{ clip_gradients(gp, 1.0f, false); });
        threshold_check(r, 10.0);
        add(r);
    }

    {
        int N = 1 << 17;
        Tensor G({1, N}); G.fill_random(-5.f, 5.f, 42);
        G.data[12345] = std::numeric_limits<float>::quiet_NaN();
        std::vector<Tensor*> gp = {&G};
        auto r = run_bench("clip_gradients NaN path [128K params]", 2, 50,
            [&]{ clip_gradients(gp, 1.0f, false); });
        threshold_check(r, 5.0);
        add(r);
    }
}


static void bench_b7_model_forward() {
    section("B7 · LOGOSModel forward + backward (small config)");

    ModelConfig cfg;
    cfg.vocab_size  = 512;
    cfg.d_model     = 64;
    cfg.num_heads   = 4;
    cfg.num_layers  = 2;
    cfg.max_seq_len = 64;
    LOGOSModel model(cfg);

    std::vector<int> inp(32, 5), tgt(32, 6);

    {
        auto r = run_bench("LOGOSModel::forward [seq=32, L=2, d=64]", 3, 20,
            [&]{ auto lo = model.forward(inp, false); g_benchmark_sink.store(lo.data.back(), std::memory_order_relaxed); });
        threshold_check(r, 500.0);
        add(r);
    }

    {
        auto params = model.parameters();
        std::vector<Tensor> grads;
        for (auto* p : params) grads.emplace_back(p->shape, 0.0f);

        auto r = run_bench("model_backward [seq=32, L=2, d=64]", 2, 10, [&]{
            for (auto& g : grads) std::fill(g.data.begin(), g.data.end(), 0.f);
            float loss = model_backward(model, inp, tgt, grads, true);
            (void)loss;
        });
        threshold_check(r, 2000.0);
        add(r);
    }

    {
        ModelConfig big;
        big.vocab_size  = 1024;
        big.d_model     = 128;
        big.num_heads   = 4;
        big.num_layers  = 4;
        big.max_seq_len = 64;
        LOGOSModel big_model(big);
        std::vector<int> inp2(48, 3), tgt2(48, 7);
        auto r = run_bench("LOGOSModel::forward [seq=48, L=4, d=128]", 2, 5,
            [&]{ auto lo = big_model.forward(inp2, false); g_benchmark_sink.store(lo.data.back(), std::memory_order_relaxed); });
        threshold_check(r, 5000.0);
        add(r);
    }
}


static void bench_b8_tokenizer_encode() {
    section("B8 · Tokenizer encode throughput");

    Tokenizer tok;
    std::string corpus;
    {
        std::vector<std::string> words = {
            "hello", "world", "logos", "vedic", "math", "physics", "model",
            "training", "gradient", "attention", "layer", "norm", "batch",
            "optimizer", "learning", "rate", "epoch", "loss", "entropy",
        };
        for (int i = 0; i < 5000; ++i)
            corpus += words[i % words.size()] + " ";
    }
    tok.build(corpus, 512);

    std::string test_text = corpus.substr(0, 4096);
    int char_count = (int)test_text.size();

    {
        auto r = run_bench("Tokenizer::encode [4KB text]", 3, 50,
            [&]{ auto ids = tok.encode(test_text, char_count * 4); g_benchmark_sink.store(static_cast<float>(ids.back()), std::memory_order_relaxed); },
            (double)char_count, "char/s");
        threshold_check(r, 200.0);
        add(r);
    }

    {
        std::string big = corpus;
        while ((int)big.size() < 65536) big += corpus;
        big.resize(65536);
        auto r = run_bench("Tokenizer::encode [64KB text]", 2, 10,
            [&]{ auto ids = tok.encode(big, (int)big.size() * 4); g_benchmark_sink.store(static_cast<float>(ids.back()), std::memory_order_relaxed); },
            (double)big.size(), "char/s");
        threshold_check(r, 2000.0);
        add(r);
    }
}


static void bench_b9_tensor_ops() {
    section("B9 · Tensor primitives (transpose, fill, add, scalar)");

    for (int N : {128, 512, 1024}) {
        Tensor A({N, N}); A.fill_random(-1.f, 1.f, 1);
        std::ostringstream nm;
        nm << "Tensor::transpose [" << N << "x" << N << "]";
        auto r = run_bench(nm.str(), 3, 20,
            [&]{ auto T = A.transpose(); (void)T; });
        threshold_check(r, 200.0);
        add(r);
    }

    {
        int N = 1 << 20;
        Tensor A({1, N}); A.fill(1.f);
        Tensor B({1, N}); B.fill(2.f);
        auto r = run_bench("Tensor::operator+ [1M elements]", 3, 30,
            [&]{ auto C = A + B; g_benchmark_sink.store(C.data.back(), std::memory_order_relaxed); },
            (double)N, "elem/s");
        threshold_check(r, 50.0);
        add(r);
    }

    {
        int N = 1 << 20;
        Tensor A({1, N}); A.fill_random(-1.f, 1.f, 5);
        auto r = run_bench("Tensor::has_nan [1M elements]", 3, 20,
            [&]{ g_benchmark_sink.store(A.has_nan() ? 1.0f : 0.0f, std::memory_order_relaxed); },
            (double)N, "elem/s");
        threshold_check(r, 20.0);
        add(r);
    }
}

static void bench_production_inference() {
    section("Production CPU inference profile");

    ModelConfig cfg;
    cfg.vocab_size  = 8192;
    cfg.d_model     = 512;
    cfg.num_heads   = 8;
    cfg.num_layers  = 8;
    cfg.max_seq_len = 128;

    LOGOSModel model(cfg);
    std::vector<int> input(cfg.max_seq_len);
    for (int i = 0; i < cfg.max_seq_len; ++i)
        input[i] = (i * 97) % cfg.vocab_size;

    std::cout << "  Parameters: " << model.count_parameters() << "\n";
    auto r = run_bench("LOGOSModel::forward [seq=128, L=8, d=512, vocab=8192]", 1, 3,
        [&]{ auto logits = model.forward(input, false); g_benchmark_sink.store(logits.data.back(), std::memory_order_relaxed); },
        static_cast<double>(cfg.max_seq_len), "token/s");
    add(r);

    Tokenizer tok;
    std::string corpus;
    corpus.reserve(1 << 20);
    for (int i = 0; i < 65536; ++i) {
        corpus += "record_" + std::to_string((i * 8191) % 4093)
               + " feature_" + std::to_string((i * 811) % 1021)
               + " value_" + std::to_string((i * 313) % 257) + " ";
    }
    tok.build(corpus, 4096);
    corpus.resize(1 << 20);
    auto tokenizer_r = run_bench("Tokenizer::encode production [1MiB, 4093 terms]", 1, 3,
        [&]{ auto ids = tok.encode(corpus, static_cast<int>(corpus.size()) * 2); g_benchmark_sink.store(static_cast<float>(ids.back()), std::memory_order_relaxed); },
        static_cast<double>(corpus.size()), "char/s");
    add(tokenizer_r);
}


int main(int argc, char* argv[]) {
    bool run_production = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--all" || std::string(argv[i]) == "--production") run_production = true;

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║  LOGOS Benchmark Suite                                       ║\n"
              << "║  median | min | max | σ per benchmark group                 ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";

    logos_rng::set_global_seed(42);

    bench_b1_vedic_gemm();
    bench_b2_attention();
    bench_b3_layernorm();
    bench_b4_feedforward();
    bench_b5_optimizers();
    bench_b6_gradient_clip();
    bench_b7_model_forward();
    bench_b8_tokenizer_encode();
    bench_b9_tensor_ops();
    if (run_production) bench_production_inference();

    int total  = (int)g_results.size();
    int failed = 0;
    for (auto& r : g_results) if (!r.passed) ++failed;

    std::cout << "\n════════════════════════════════════════════════════════════════\n";
    std::cout << "  Benchmarks: " << total
              << "  |  Above threshold (FAIL): " << failed << "\n";

    if (failed > 0) {
        std::cout << "\n  Threshold violations:\n";
        for (auto& r : g_results)
            if (!r.passed)
                std::cout << "    [FAIL] " << r.name << " — " << r.fail_reason << "\n";
    }

    std::cout << "════════════════════════════════════════════════════════════════\n";
    return failed > 0 ? 1 : 0;
}
