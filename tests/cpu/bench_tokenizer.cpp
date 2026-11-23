// T24: Micro-benchmarks + coarse regression floors (CPU only).
// Floors are deliberately generous: they catch catastrophic regressions (O(n^2), accidental
// per-call rebuilds), not 10% noise. Results go to LOGOS_BENCH_JSON (default bench_results.json).
//   LOGOS_BENCH_MIN_MBPS  (default 0.5)   encode / pretokenize floor
//   LOGOS_BENCH_MAX_SCALE (default 8.0)   time(4x input) / time(1x input) ceiling
#include "test_common.hpp"
#include "TokenBin.hpp"
#include "PathIntegral.hpp"
#include <chrono>
#include <map>

using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static std::map<std::string, double> g_results;

static void record(const char* name, double v, const char* unit) {
    g_results[name] = v;
    std::printf("BENCH %-34s %14.3f %s\n", name, v, unit);
    std::fflush(stdout);
}

int main() {
    const double min_mbps  = tc::env_double("LOGOS_BENCH_MIN_MBPS", 0.5);
    const double max_scale = tc::env_double("LOGOS_BENCH_MAX_SCALE", 8.0);
    std::printf("LOGOS T24: benchmarks (min_mbps=%.2f max_scale=%.1f)\n", min_mbps, max_scale);

    // ---- BPE build ----
    std::string train = tc::make_corpus(1 * 1024 * 1024, 1);
    Tokenizer tok;
    double dt_build;
    {
        tc::Silence s;
        auto tb = Clock::now();
        tok.build(train, 1000);
        dt_build = since(tb);
    }
    record("bpe_build_1MB_v1000_seconds", dt_build, "s");
    record("bpe_merges", (double)tok.merges.size(), "merges");
    TC_CHECK(dt_build < 120.0, "BPE build of 1MB corpus < 120 s");

    // ---- encode throughput (cold cache, then warm) ----
    std::string text1 = tc::make_corpus(1 * 1024 * 1024, 2);
    std::string text4 = tc::make_corpus(4 * 1024 * 1024, 3);
    Tokenizer cold(tok);
    auto t0 = Clock::now();
    auto ids1 = cold.encode_raw(text1);
    double dt_cold = since(t0);
    record("encode_cold_MBps", 1.0 / std::max(dt_cold, 1e-9), "MB/s");
    TC_CHECKF(1.0 / std::max(dt_cold, 1e-9) >= min_mbps, "encode cold %.3f MB/s < floor %.3f", 1.0 / dt_cold, min_mbps);

    t0 = Clock::now();
    auto ids1w = cold.encode_raw(text1);
    double dt_warm = since(t0);
    record("encode_warm_MBps", 1.0 / std::max(dt_warm, 1e-9), "MB/s");
    TC_CHECK(ids1w == ids1, "warm == cold output");

    // ---- scaling: 4x input must not cost >> 4x time (quadratic-behaviour guard) ----
    t0 = Clock::now();
    auto ids4 = cold.encode_raw(text4);
    double dt4 = since(t0);
    double ratio = dt4 / std::max(dt_warm, 1e-9);
    record("encode_scaling_ratio_4x", ratio, "x");
    TC_CHECKF(ratio < max_scale, "encode scaling ratio %.2f >= %.2f (superlinear?)", ratio, max_scale);
    record("tokens_per_byte", (double)ids1.size() / (double)text1.size(), "tok/B");

    // ---- decode throughput ----
    t0 = Clock::now();
    std::string back = cold.decode(ids4);
    double dtd = since(t0);
    record("decode_MBps", 4.0 / std::max(dtd, 1e-9), "MB/s");
    TC_CHECK(back == text4, "decode(encode(x)) == x on 4MB");

    // ---- pretokenize_file throughput ----
    std::string path = tc::tmp_file("bench_text.txt");
    std::string bin  = tc::tmp_file("bench_text.bin");
    tc::write_all(path, text4);
    bool ok;
    t0 = Clock::now();
    { tc::Silence s; ok = pretokenize_file(path, cold, bin, 1 << 20); }
    double dtp = since(t0);
    TC_CHECK(ok, "pretokenize ok");
    record("pretokenize_MBps", 4.0 / std::max(dtp, 1e-9), "MB/s");
    TC_CHECKF(4.0 / std::max(dtp, 1e-9) >= min_mbps, "pretokenize %.3f MB/s < floor %.3f", 4.0 / dtp, min_mbps);

    // ---- TokenBinLoader batches/s ----
    {
        TokenBinLoader ld(bin, 256, 1);
        std::vector<int> in, tg;
        int64_t n = 0;
        t0 = Clock::now();
        while (ld.next_batch(in, tg)) ++n;
        double dtl = since(t0);
        record("tokenbin_batches_per_s", (double)n / std::max(dtl, 1e-9), "batch/s");
        TC_CHECK(n > 100, "loader produced batches");
    }

    // ---- physics: WeightPathIntegral step rate ----
    {
        WeightPathIntegral p(1.0f);
        const int N = 5000000;
        float sink = 0.0f;
        t0 = Clock::now();
        for (int i = 0; i < N; ++i) {
            p.record_step_norm(1.0f + (i & 7) * 0.1f, 0.01f);
            sink += p.lr_scale_ema();
        }
        double dtpi = since(t0);
        record("path_integral_Msteps_per_s", (N / 1e6) / std::max(dtpi, 1e-9), "Mstep/s");
        TC_CHECK(std::isfinite(sink), "path integral finite after 5M steps");
        TC_CHECKF((N / 1e6) / std::max(dtpi, 1e-9) > 0.5, "path integral rate too slow");
    }

    // ---- JSON report ----
    const char* jp = std::getenv("LOGOS_BENCH_JSON");
    std::string json_path = (jp && *jp) ? jp : "bench_results.json";
    {
        std::ofstream j(json_path);
        j << "{\n";
        size_t i = 0;
        for (auto& kv : g_results) {
            j << "  \"" << kv.first << "\": " << kv.second << (++i < g_results.size() ? "," : "") << "\n";
        }
        j << "}\n";
    }
    std::printf("bench json: %s\n", json_path.c_str());
    return tc::finish("T24_Bench");
}
