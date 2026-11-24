// T24: Micro-benchmarks + regression gates (CPU only).
//
// Corpus: by default a Zipfian pseudo-text with a LARGE lexicon (tc::make_corpus has only 30 words, so the
// tokenizer's piece cache is always hot and "cold" == "warm" - it measured nothing). Point LOGOS_BENCH_CORPUS at a
// real text file (e.g. 50-100 MB of English/Hindi) for production-grade numbers.
//
// Every timed section is repeated LOGOS_BENCH_REPS times and the MEDIAN is reported (the BPE build is one sample).
// Results go to LOGOS_BENCH_JSON (default bench_results.json) as a flat {"metric": number} object.
//
// Gates (all env-tunable):
//   LOGOS_BENCH_MIN_MBPS     (default 2.0)    absolute floor for encode / pretokenize throughput (MB/s)
//   LOGOS_BENCH_MAX_SCALE    (default 6.0)    time(4x input) / time(1x input) ceiling (linear == 4)
//   LOGOS_BENCH_MAX_BUILD_S  (default 120)    BPE build ceiling (seconds)
//   LOGOS_BENCH_BASELINE     (path)           JSON of a previous green run; rates must stay >= (1-TOL) * baseline,
//                                             *_seconds must stay <= baseline / (1-TOL)
//   LOGOS_BENCH_TOL          (default 0.5)    tolerance for the baseline gate (shared CI runners are noisy)
// Sizes: LOGOS_BENCH_MB (default 4) = unit size, LOGOS_BENCH_VOCAB (4000), LOGOS_BENCH_LEXICON (50000),
//        LOGOS_BENCH_REPS (5).
#include "test_common.hpp"
#include "TokenBin.hpp"
#include "PathIntegral.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
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

static double median_of(int reps, const std::function<double()>& timed) {
    std::vector<double> t;
    t.reserve(static_cast<size_t>(reps));
    for (int i = 0; i < reps; ++i) t.push_back(timed());
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static double MB(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

static bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Zipfian pseudo-text: 1-4 syllable words (some with a Devanagari letter), rank-frequency ~ 1/rank.
static std::string make_zipf_corpus(size_t approx_bytes, uint32_t seed, size_t lexicon_size) {
    static const char* SYL[] = {"ka", "ra", "mi", "to", "ne", "su", "la", "vo", "di", "an", "el", "or",
                                "ti", "ma", "ri", "sa", "no", "pe", "lu", "ga", "he", "wi", "za", "ko",
                                "fa", "by", "ch", "th", "sh", "qu", "st", "pr"};
    constexpr size_t NS = sizeof(SYL) / sizeof(SYL[0]);
    std::mt19937_64 g(seed);
    std::vector<std::string> lex;
    lex.reserve(lexicon_size);
    for (size_t i = 0; i < lexicon_size; ++i) {
        std::string w;
        const size_t syl = 1 + static_cast<size_t>(g() % 4);
        for (size_t k = 0; k < syl; ++k) w += SYL[g() % NS];
        if (g() % 9 == 0) w += (g() % 2) ? "\xE0\xA4\x95" : "\xE0\xA4\xAE";
        lex.push_back(std::move(w));
    }
    std::vector<double> cdf(lexicon_size);
    double acc = 0.0;
    for (size_t i = 0; i < lexicon_size; ++i) { acc += 1.0 / static_cast<double>(i + 1); cdf[i] = acc; }
    std::uniform_real_distribution<double> U(0.0, acc);
    std::string s;
    s.reserve(approx_bytes + 64);
    while (s.size() < approx_bytes) {
        size_t idx = static_cast<size_t>(std::lower_bound(cdf.begin(), cdf.end(), U(g)) - cdf.begin());
        if (idx >= lexicon_size) idx = lexicon_size - 1;
        s += lex[idx];
        const unsigned r = static_cast<unsigned>(g() % 100);
        if      (r < 5)  s += "\n";
        else if (r < 7)  s += "\n\n";
        else if (r < 9)  s += "  ";
        else if (r < 10) s += "\t";
        else if (r < 12) s += ", ";
        else if (r < 14) s += ". ";
        else             s += " ";
    }
    return s;
}

// Flat {"key": number, ...} reader for our own bench_results.json (one pair per line).
static std::map<std::string, double> load_flat_json(const std::string& path) {
    std::map<std::string, double> m;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t q1 = line.find('"');
        if (q1 == std::string::npos) continue;
        const size_t q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        const size_t colon = line.find(':', q2);
        if (colon == std::string::npos) continue;
        const char* start = line.c_str() + colon + 1;
        char* end = nullptr;
        const double v = std::strtod(start, &end);
        if (end != start) m[line.substr(q1 + 1, q2 - q1 - 1)] = v;
    }
    return m;
}

// higher-is-better: *MBps, *_per_s     lower-is-better: *_seconds     everything else: informational only
static void compare_with_baseline(const std::map<std::string, double>& base, double tol) {
    TC_SECTION("regression gate vs baseline");
    int compared = 0;
    for (const auto& kv : g_results) {
        const std::string& k = kv.first;
        const auto it = base.find(k);
        if (it == base.end() || !(it->second > 0.0)) continue;
        const bool rate = ends_with(k, "MBps") || k.find("_per_s") != std::string::npos;
        const bool secs = ends_with(k, "_seconds");
        if (!rate && !secs) continue;
        if (secs && it->second < 0.05) continue;       // too short to time reliably
        ++compared;
        if (rate) {
            TC_CHECKF(kv.second >= it->second * (1.0 - tol),
                      "%s regressed: %.3f vs baseline %.3f (must stay >= %.0f%% of it)",
                      k.c_str(), kv.second, it->second, 100.0 * (1.0 - tol));
        } else {
            TC_CHECKF(kv.second <= it->second / (1.0 - tol),
                      "%s regressed: %.3f s vs baseline %.3f s (must stay <= %.1fx)",
                      k.c_str(), kv.second, it->second, 1.0 / (1.0 - tol));
        }
    }
    std::printf("baseline gate: %d metrics compared (tolerance %.0f%%)\n", compared, tol * 100.0);
}

int main() {
    const double min_mbps  = tc::env_double("LOGOS_BENCH_MIN_MBPS", 2.0);
    const double max_scale = tc::env_double("LOGOS_BENCH_MAX_SCALE", 6.0);
    const double max_build = tc::env_double("LOGOS_BENCH_MAX_BUILD_S", 120.0);
    const double tol       = std::min(0.95, std::max(0.05, tc::env_double("LOGOS_BENCH_TOL", 0.5)));
    const int    reps      = static_cast<int>(std::max(1L, tc::env_long("LOGOS_BENCH_REPS", 5)));
    const int    io_reps   = std::min(reps, 3);
    const int    vocab     = static_cast<int>(std::max(300L, tc::env_long("LOGOS_BENCH_VOCAB", 4000)));
    const size_t lexicon   = static_cast<size_t>(std::max(1000L, tc::env_long("LOGOS_BENCH_LEXICON", 50000)));
    size_t unit = static_cast<size_t>(std::max(0.05, tc::env_double("LOGOS_BENCH_MB", 4.0)) * 1024.0 * 1024.0);
    unit &= ~static_cast<size_t>(1);

    // corpus layout: [ train: unit/2 | text1: unit | text4: 4*unit ]  (disjoint, so encode sees unseen words)
    std::string all;
    const char* source = "zipf";
    if (const char* p = std::getenv("LOGOS_BENCH_CORPUS"); p && *p) {
        all = tc::read_all(p);
        if (all.empty()) std::fprintf(stderr, "LOGOS_BENCH_CORPUS=%s unreadable or empty -> using Zipf corpus\n", p);
        else { source = "file"; unit = std::min(unit, all.size() * 2 / 11); unit &= ~static_cast<size_t>(1); }
    }
    TC_CHECK(unit >= 4096, "bench corpus large enough");
    if (std::strcmp(source, "zipf") == 0) all = make_zipf_corpus(unit / 2 + unit + 4 * unit, 1, lexicon);
    const std::string train = all.substr(0, unit / 2);
    const std::string text1 = all.substr(unit / 2, unit);
    const std::string text4 = all.substr(unit / 2 + unit, 4 * unit);
    all.clear();
    all.shrink_to_fit();

    std::printf("LOGOS T24: benchmarks  corpus=%s unit=%.2f MB reps=%d vocab=%d (min_mbps=%.2f max_scale=%.1f tol=%.0f%%)\n",
                source, MB(unit), reps, vocab, min_mbps, max_scale, tol * 100.0);
    record("corpus_unit_MB", MB(unit), "MB");
    record("reps", static_cast<double>(reps), "runs");

    // ---- BPE build (single sample) ----
    Tokenizer tok;
    double dt_build;
    {
        tc::Silence s;
        const auto t0 = Clock::now();
        tok.build(train, vocab);
        dt_build = since(t0);
    }
    record("bpe_build_seconds", dt_build, "s");
    record("bpe_merges", static_cast<double>(tok.merges.size()), "merges");
    TC_CHECKF(dt_build < max_build, "BPE build %.2f s >= ceiling %.1f s", dt_build, max_build);
    TC_CHECK(tok.merges.size() > 100, "BPE learned a non-trivial number of merges");

    // ---- encode throughput: cold (fresh piece cache every rep) ----
    std::vector<int> ids1;
    const double dt_cold = median_of(reps, [&]() {
        Tokenizer c(tok);                         // copy ctor => fresh, empty piece cache
        const auto t0 = Clock::now();
        ids1 = c.encode_raw(text1);
        return since(t0);
    });
    const double cold_mbps = MB(text1.size()) / std::max(dt_cold, 1e-9);
    record("encode_cold_MBps", cold_mbps, "MB/s");
    TC_CHECKF(cold_mbps >= min_mbps, "encode cold %.3f MB/s < floor %.3f", cold_mbps, min_mbps);

    // ---- encode throughput: warm (cache primed on the 4x text, so 1x and 4x are both ~all hits) ----
    Tokenizer warm(tok);
    std::vector<int> ids4 = warm.encode_raw(text4);
    const double dt_w1 = median_of(reps, [&]() {
        const auto t0 = Clock::now();
        std::vector<int> v = warm.encode_raw(text1);
        const double dt = since(t0);
        TC_CHECK(v == ids1, "warm encode == cold encode");
        return dt;
    });
    const double dt_w4 = median_of(reps, [&]() {
        const auto t0 = Clock::now();
        std::vector<int> v = warm.encode_raw(text4);
        const double dt = since(t0);
        TC_CHECK(v.size() == ids4.size(), "warm 4x encode is stable");
        return dt;
    });
    record("encode_warm_MBps", MB(text1.size()) / std::max(dt_w1, 1e-9), "MB/s");
    const double ratio = dt_w4 / std::max(dt_w1, 1e-9);
    record("encode_scaling_ratio_4x", ratio, "x");
    TC_CHECKF(ratio < max_scale, "encode scaling ratio %.2f >= %.2f (superlinear?)", ratio, max_scale);
    record("tokens_per_byte", static_cast<double>(ids1.size()) / static_cast<double>(text1.size()), "tok/B");

    // ---- decode throughput ----
    std::string back;
    const double dtd = median_of(reps, [&]() {
        const auto t0 = Clock::now();
        back = warm.decode(ids4);
        return since(t0);
    });
    record("decode_MBps", MB(text4.size()) / std::max(dtd, 1e-9), "MB/s");
    TC_CHECK(back == text4, "decode(encode(x)) == x on the 4x text");

    // ---- pretokenize_file throughput ----
    const std::string path = tc::tmp_file("bench_text.txt");
    const std::string bin  = tc::tmp_file("bench_text.bin");
    tc::write_all(path, text4);
    bool pre_ok = true;
    const double dtp = median_of(io_reps, [&]() {
        tc::Silence s;
        const auto t0 = Clock::now();
        pre_ok = pretokenize_file(path, warm, bin, 1 << 20) && pre_ok;
        return since(t0);
    });
    TC_CHECK(pre_ok, "pretokenize ok");
    const double pre_mbps = MB(text4.size()) / std::max(dtp, 1e-9);
    record("pretokenize_MBps", pre_mbps, "MB/s");
    TC_CHECKF(pre_mbps >= min_mbps, "pretokenize %.3f MB/s < floor %.3f", pre_mbps, min_mbps);

    // ---- TokenBinLoader batches/s ----
    {
        int64_t n = 0;
        const double dtl = median_of(io_reps, [&]() {
            TokenBinLoader ld(bin, 256, 1);
            std::vector<int> in, tg;
            n = 0;
            const auto t0 = Clock::now();
            while (ld.next_batch(in, tg)) ++n;
            return since(t0);
        });
        record("tokenbin_batches_per_s", static_cast<double>(n) / std::max(dtl, 1e-9), "batch/s");
        TC_CHECK(n > 100, "tokenbin loader produced batches");
    }

    // ---- StreamingDataLoader batches/s (tokenizes on the fly: what training uses without a .bin) ----
    {
        const int64_t max_batches = 2000;
        int64_t n = 0;
        const double dts = median_of(io_reps, [&]() {
            StreamingDataLoader ld(path, warm, 256, 1, 1 << 20);
            std::vector<int> in, tg;
            n = 0;
            const auto t0 = Clock::now();
            while (n < max_batches && ld.next_batch(in, tg)) ++n;
            return since(t0);
        });
        record("streaming_batches_per_s", static_cast<double>(n) / std::max(dts, 1e-9), "batch/s");
        TC_CHECK(n > 100, "streaming loader produced batches");
    }

    // ---- physics: WeightPathIntegral step rate ----
    {
        const int N = 5000000;
        float sink = 0.0f;
        const double dtpi = median_of(io_reps, [&]() {
            WeightPathIntegral p(1.0f);
            const auto t0 = Clock::now();
            for (int i = 0; i < N; ++i) {
                p.record_step_norm(1.0f + (i & 7) * 0.1f, 0.01f);
                sink += p.lr_scale_ema();
            }
            return since(t0);
        });
        const double rate = (N / 1e6) / std::max(dtpi, 1e-9);
        record("path_integral_Msteps_per_s", rate, "Mstep/s");
        TC_CHECK(std::isfinite(sink), "path integral finite after 5M steps x reps");
        TC_CHECKF(rate > 0.5, "path integral rate %.3f Mstep/s too slow", rate);
    }

    // ---- JSON report ----
    const char* jp = std::getenv("LOGOS_BENCH_JSON");
    const std::string json_path = (jp && *jp) ? jp : "bench_results.json";
    {
        std::ofstream j(json_path);
        j.precision(10);
        j << "{\n";
        size_t i = 0;
        for (const auto& kv : g_results)
            j << "  \"" << kv.first << "\": " << kv.second << (++i < g_results.size() ? "," : "") << "\n";
        j << "}\n";
    }
    std::printf("bench json: %s\n", json_path.c_str());

    // ---- regression gate vs a previous green run ----
    if (const char* bp = std::getenv("LOGOS_BENCH_BASELINE"); bp && *bp) {
        const auto base = load_flat_json(bp);
        if (base.empty()) std::printf("baseline %s missing/empty -> first run, nothing to compare\n", bp);
        else              compare_with_baseline(base, tol);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(bin, ec);
    return tc::finish("T24_Bench");
}
