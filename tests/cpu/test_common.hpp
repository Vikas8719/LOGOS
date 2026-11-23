#pragma once
// Chhota header-only test harness (CPU tests ke liye). Koi external dependency nahi.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace tc {

inline int& passes() { static int p = 0; return p; }
inline int& fails()  { static int f = 0; return f; }

inline long env_long(const char* name, long def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    return (end && *end == '\0') ? x : def;
}

inline double env_double(const char* name, double def) {
    const char* v = std::getenv(name);
    if (!v || !*v) return def;
    char* end = nullptr;
    double x = std::strtod(v, &end);
    return (end && *end == '\0') ? x : def;
}

inline std::string tmp_file(const std::string& name) {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "logos_cpu_tests";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return (dir / name).string();
}

inline void write_all(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
}

inline std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Deterministic corpus: English + Hindi (UTF-8) words, newlines, double spaces, tabs.
inline std::string make_corpus(size_t approx_bytes, uint32_t seed) {
    static const char* W[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "physics",
        "energy", "entropy", "gradient", "momentum", "tensor", "logos", "model",
        "transformer", "attention", "hamiltonian", "langevin", "free", "token",
        "hello", "world", "data", "नमस्ते", "दुनिया", "भौतिकी", "ऊर्जा", "क्वांटम"
    };
    constexpr size_t NW = sizeof(W) / sizeof(W[0]);
    std::mt19937 g(seed);
    std::string s;
    s.reserve(approx_bytes + 64);
    while (s.size() < approx_bytes) {
        s += W[g() % NW];
        unsigned r = g() % 100;
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

// cout/cerr ko chup karata hai (load() bahut print karta hai).
struct Silence {
    std::streambuf* o_; std::streambuf* e_;
    std::ostringstream sink_;
    Silence()  { o_ = std::cout.rdbuf(sink_.rdbuf()); e_ = std::cerr.rdbuf(sink_.rdbuf()); }
    ~Silence() { std::cout.rdbuf(o_); std::cerr.rdbuf(e_); }
};

inline int finish(const char* suite) {
    std::printf("\n== %s: %d passed, %d failed ==\n", suite, passes(), fails());
    std::fflush(stdout);
    return fails() == 0 ? 0 : 1;
}

} // namespace tc

#define TC_SECTION(name) std::printf("\n-- %s --\n", (name))

#define TC_CHECK(cond, msg) \
    do { \
        if (cond) { ++tc::passes(); } \
        else { ++tc::fails(); std::fprintf(stderr, "  FAIL [%s:%d] %s\n", __FILE__, __LINE__, (msg)); } \
    } while (0)

#define TC_CHECKF(cond, ...) \
    do { \
        if (cond) { ++tc::passes(); } \
        else { ++tc::fails(); std::fprintf(stderr, "  FAIL [%s:%d] ", __FILE__, __LINE__); \
               std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
    } while (0)

#define TC_NEAR(val, ref, tol, msg) \
    do { \
        double _v = (double)(val), _r = (double)(ref), _t = (double)(tol); \
        if (std::fabs(_v - _r) <= _t) { ++tc::passes(); } \
        else { ++tc::fails(); \
               std::fprintf(stderr, "  FAIL [%s:%d] %s | got %.9g expected %.9g (tol %.3g)\n", \
                            __FILE__, __LINE__, (msg), _v, _r, _t); } \
    } while (0)
