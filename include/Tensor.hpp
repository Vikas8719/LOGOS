#pragma once
#include <vector>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cmath>

// ============================================================
//  LOGOS — Tensor.hpp
//  Sabka base: N-dimensional data store (Row-Major format)
//  CPU cache-friendly layout — Matrix ops ke liye optimized
// ============================================================

class Tensor {
public:
    std::vector<float> data;   // Flat 1D buffer — Row-Major
    std::vector<int>   shape;  // Dimensions (e.g., {batch, seq, dim})
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
        data.reserve(total_size);   // Memory leak prevention
        data.assign(total_size, fill);
    }

    // ── Shape utilities ───────────────────────────────────────
    int rows() const { return shape.size() >= 2 ? shape[shape.size()-2] : 1; }
    int cols() const { return shape.empty() ? 0 : shape.back(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    // ── Element access (2D shorthand) ─────────────────────────
    float& at(int r, int c) {
        assert(r >= 0 && c >= 0 && r < rows() && c < cols());
        return data[r * cols() + c];
    }
    float at(int r, int c) const {
        assert(r >= 0 && c >= 0 && r < rows() && c < cols());
        return data[r * cols() + c];
    }

    // ── 1D access ─────────────────────────────────────────────
    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    // ── Fill operations ───────────────────────────────────────
    void fill(float val) { std::fill(data.begin(), data.end(), val); }
    void zero()          { fill(0.0f); }

    void fill_random(float lo = -0.1f, float hi = 0.1f) {
        // Xavier-style small init — training stable rehta hai
        for (float& v : data)
            v = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
    }

    // ── Element-wise ops ──────────────────────────────────────
    Tensor operator+(const Tensor& other) const {
        assert(total_size == other.total_size);
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
        assert(total_size == other.total_size);
        for (int i = 0; i < total_size; ++i)
            data[i] += other.data[i];
        return *this;
    }

    // ── Reshape (no copy — same data, new view) ───────────────
    Tensor reshape(std::vector<int> new_shape) const {
        int new_total = 1;
        for (int d : new_shape) new_total *= d;
        assert(new_total == total_size);
        Tensor result;
        result.data  = data;          // shared copy (value semantics)
        result.shape = std::move(new_shape);
        result.total_size = total_size;
        return result;
    }

    // ── Transpose (2D only) ───────────────────────────────────
    Tensor transpose() const {
        assert(ndim() == 2);
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
            for (int c = 0; c < C; ++c)
                std::cout << at(r, c) << " ";
            if (cols() > max_cols) std::cout << "...";
            std::cout << "]\n";
        }
        if (rows() > max_rows) std::cout << "  ...\n";
    }

    // ── Statistics ────────────────────────────────────────────
    float mean() const {
        return std::accumulate(data.begin(), data.end(), 0.0f) / total_size;
    }
    float max_val() const { return *std::max_element(data.begin(), data.end()); }
    float min_val() const { return *std::min_element(data.begin(), data.end()); }

    bool has_nan() const {
        for (float v : data) if (std::isnan(v) || std::isinf(v)) return true;
        return false;
    }
};
