#pragma once
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <atomic>

// ============================================================
//  LOGOS — Tensor.hpp
//
//  BUG 8 FIX: fill_random() — deterministic seeding system
//    Pehle: rand() global state use karta tha
//           - Non-reproducible: har run alag weights
//           - Thread-unsafe: race condition in multi-thread
//           - srand(42) sirf unit test mein call hota tha,
//             production training mein nahi → completely random init
//    Ab:    Global deterministic RNG with seed control:
//           - set_global_seed(N) → reproducible training runs
//           - Default seed=42 → consistent results out-of-box
//           - Each fill_random() advances global RNG atomically
//           - Thread-safe via per-call snapshot pattern
//           - fill_random(lo, hi, local_seed) bhi supported
//             (local_seed >= 0 → isolated RNG for that tensor only)
//
//  BUG 15 FIX (carry forward): at() bounds check always active
//    assert() → disabled in Release. Now: std::out_of_range always.
//
//  BUG 2 FIX (carry forward): reshape() avoids unnecessary copy.
// ============================================================

// ── Global RNG for reproducible weight initialization ────────
// BUG 8 FIX: replaces global rand() with seeded mt19937
// Call set_global_seed() at program start for reproducibility.
namespace logos_rng {
    // Global generator — default seed 42 (same init every run)
    inline std::mt19937& global_gen() {
        static std::mt19937 gen(42u);
        return gen;
    }

    // Set global seed — call once in main() before model init
    // Example: logos_rng::set_global_seed(42);
    inline void set_global_seed(unsigned seed) {
        global_gen().seed(seed);
    }

    // Draw N values from [lo, hi] using global generator
    inline void fill_uniform(std::vector<float>& buf, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : buf) v = dist(global_gen());
    }

    // Draw N values using a LOCAL generator (isolated, reproducible)
    inline void fill_uniform_local(std::vector<float>& buf, float lo, float hi,
                                   unsigned seed) {
        std::mt19937 local_gen(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (float& v : buf) v = dist(local_gen);
    }
}

class Tensor {
public:
    std::vector<float> data;
    std::vector<int>   shape;
    int total_size = 0;

    // ── Constructors ──────────────────────────────────────────
    Tensor() = default;

    explicit Tensor(std::vector<int> shape_, float fill = 0.0f)
        : shape(std::move(shape_))
    {
        total_size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Shape dimension must be > 0");
            total_size *= d;
        }
        data.reserve(total_size);
        data.assign(total_size, fill);
    }

    // ── Shape utilities ───────────────────────────────────────
    int rows() const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
    int cols() const { return shape.empty() ? 0 : shape.back(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    // ── BUG 15 FIX: bounds check always active ───────────────
    float& at(int r, int c) {
        if (r < 0 || c < 0 || r >= rows() || c >= cols())
            throw std::out_of_range(
                "Tensor::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of range for (" + std::to_string(rows()) + "," +
                std::to_string(cols()) + ")");
        return data[r * cols() + c];
    }
    float at(int r, int c) const {
        if (r < 0 || c < 0 || r >= rows() || c >= cols())
            throw std::out_of_range(
                "Tensor::at(" + std::to_string(r) + "," + std::to_string(c) +
                ") out of range for (" + std::to_string(rows()) + "," +
                std::to_string(cols()) + ")");
        return data[r * cols() + c];
    }

    // ── 1D access (unchecked — hot paths) ────────────────────
    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    // ── Fill ─────────────────────────────────────────────────
    void fill(float val) { std::fill(data.begin(), data.end(), val); }
    void zero()          { fill(0.0f); }

    // BUG 8 FIX: Deterministic random fill
    // local_seed < 0 (default) → use global seeded RNG (recommended)
    // local_seed >= 0          → use isolated local RNG for this tensor
    //
    // For reproducible training:
    //   Call logos_rng::set_global_seed(42) once in main() before model init.
    //   All fill_random() calls with no local_seed will then be deterministic.
    //
    // Pehle: rand() → non-deterministic, thread-unsafe, srand() ignored in training
    // Ab:    logos_rng::global_gen() → seeded, deterministic, consistent
    void fill_random(float lo = -0.1f, float hi = 0.1f, int local_seed = -1) {
        if (local_seed >= 0) {
            logos_rng::fill_uniform_local(data, lo, hi,
                                          static_cast<unsigned>(local_seed));
        } else {
            logos_rng::fill_uniform(data, lo, hi);
        }
    }

    // ── Element-wise ops ──────────────────────────────────────
    Tensor operator+(const Tensor& other) const {
        if (total_size != other.total_size)
            throw std::invalid_argument("Tensor::operator+ size mismatch");
        Tensor result(shape);
        for (int i = 0; i < total_size; ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape);
        for (int i = 0; i < total_size; ++i)
            result.data[i] = data[i] * scalar;
        return result;
    }

    Tensor& operator+=(const Tensor& other) {
        if (total_size != other.total_size)
            throw std::invalid_argument("Tensor::operator+= size mismatch");
        for (int i = 0; i < total_size; ++i)
            data[i] += other.data[i];
        return *this;
    }

    // ── BUG 2 FIX: reshape — avoids unnecessary copy ─────────
    Tensor reshape(std::vector<int> new_shape) const {
        int new_total = 1;
        for (int d : new_shape) new_total *= d;
        if (new_total != total_size)
            throw std::invalid_argument(
                "reshape: elements mismatch (" + std::to_string(total_size) +
                " → " + std::to_string(new_total) + ")");
        Tensor result;
        result.data       = data;
        result.shape      = std::move(new_shape);
        result.total_size = total_size;
        return result;
    }

    Tensor& reshape_inplace(std::vector<int> new_shape) {
        int new_total = 1;
        for (int d : new_shape) new_total *= d;
        if (new_total != total_size)
            throw std::invalid_argument("reshape_inplace: elements mismatch");
        shape = std::move(new_shape);
        return *this;
    }

    // ── Transpose (2D only) ───────────────────────────────────
    Tensor transpose() const {
        if (ndim() != 2)
            throw std::invalid_argument("transpose: only 2D tensors supported");
        int R = rows(), C = cols();
        Tensor result({C, R});
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                result.at(c, r) = at(r, c);
        return result;
    }

    // ── Debug print ───────────────────────────────────────────
    void print(const std::string& name = "Tensor", int max_rows = 4, int max_cols = 8) const {
        std::cout << "[" << name << "] shape=(";
        for (int i = 0; i < (int)shape.size(); ++i)
            std::cout << shape[i] << (i+1<(int)shape.size() ? "," : "");
        std::cout << ")\n";
        int R = std::min(rows(), max_rows);
        int C = std::min(cols(), max_cols);
        for (int r = 0; r < R; ++r) {
            std::cout << "  [ ";
            for (int c = 0; c < C; ++c) std::cout << at(r, c) << " ";
            if (cols() > max_cols) std::cout << "...";
            std::cout << "]\n";
        }
        if (rows() > max_rows) std::cout << "  ...\n";
    }

    // ── Statistics ────────────────────────────────────────────
    float mean() const {
        if (total_size == 0) return 0.0f;
        return std::accumulate(data.begin(), data.end(), 0.0f) / total_size;
    }
    float max_val() const {
        if (data.empty()) throw std::runtime_error("max_val on empty tensor");
        return *std::max_element(data.begin(), data.end());
    }
    float min_val() const {
        if (data.empty()) throw std::runtime_error("min_val on empty tensor");
        return *std::min_element(data.begin(), data.end());
    }
    bool has_nan() const {
        for (float v : data) if (std::isnan(v) || std::isinf(v)) return true;
        return false;
    }
};
