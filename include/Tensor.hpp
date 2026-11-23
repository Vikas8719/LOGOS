#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace logos_rng {

inline std::mt19937& global_gen() {
    static std::mt19937 gen(42u);
    return gen;
}

inline void set_global_seed(unsigned seed) { global_gen().seed(seed); }

inline void fill_uniform(std::vector<float>& buf, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : buf) v = dist(global_gen());
}

inline void fill_uniform_local(std::vector<float>& buf, float lo, float hi, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : buf) v = dist(gen);
}

}  // namespace logos_rng

// Host-side row-major float tensor; weight container for checkpoints.
class Tensor {
public:
    std::vector<float> data;
    std::vector<int>   shape;
    int total_size = 0;

    Tensor() = default;

    explicit Tensor(std::vector<int> shape_, float fill_value = 0.0f)
        : shape(std::move(shape_))
    {
        total_size = 1;
        for (int d : shape) {
            if (d <= 0) throw std::invalid_argument("Shape dimension must be > 0");
            total_size *= d;
        }
        data.assign(total_size, fill_value);
    }

    int rows() const { return shape.size() >= 2 ? shape[shape.size() - 2] : 1; }
    int cols() const { return shape.empty() ? 0 : shape.back(); }
    int ndim() const { return static_cast<int>(shape.size()); }

    float& at(int r, int c) { check_index(r, c); return data[r * cols() + c]; }
    float  at(int r, int c) const { check_index(r, c); return data[r * cols() + c]; }

    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void zero() { fill(0.0f); }

    void fill_random(float lo = -0.1f, float hi = 0.1f, int local_seed = -1) {
        if (local_seed >= 0) logos_rng::fill_uniform_local(data, lo, hi, static_cast<unsigned>(local_seed));
        else                 logos_rng::fill_uniform(data, lo, hi);
    }

    bool has_nan() const {
        for (float v : data) if (!std::isfinite(v)) return true;
        return false;
    }

private:
    void check_index(int r, int c) const {
        if (r < 0 || c < 0 || r >= rows() || c >= cols())
            throw std::out_of_range("Tensor::at(" + std::to_string(r) + "," + std::to_string(c) + ") out of range");
    }
};
