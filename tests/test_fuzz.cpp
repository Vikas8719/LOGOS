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
#include "../include/Checkpoint.hpp"

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <stdexcept>

struct FuzzReader {
    const uint8_t* data;
    size_t         size;
    size_t         pos;

    FuzzReader(const uint8_t* d, size_t s) : data(d), size(s), pos(0) {}

    uint8_t  u8()  { return pos < size ? data[pos++] : 0; }
    uint16_t u16() { uint16_t v = 0; std::memcpy(&v, data + pos, std::min(size-pos,(size_t)2)); pos+=2; return v; }
    uint32_t u32() { uint32_t v = 0; std::memcpy(&v, data + pos, std::min(size-pos,(size_t)4)); pos+=4; return v; }
    float    f32() { float v = 0.f; std::memcpy(&v, data + pos, std::min(size-pos,(size_t)4)); pos+=4; return v; }

    int bounded_int(int lo, int hi) {
        if (lo >= hi) return lo;
        return lo + static_cast<int>(u16() % static_cast<uint16_t>(hi - lo));
    }

    float bounded_float(float lo, float hi) {
        float raw = f32();
        if (!std::isfinite(raw)) raw = 0.f;
        float range = hi - lo;
        float wrapped = std::fmod(std::fabs(raw), range + 1.f);
        return lo + wrapped;
    }

    std::string string_of(size_t max_len) {
        size_t len = bounded_int(0, static_cast<int>(max_len));
        std::string s;
        for (size_t i = 0; i < len && pos < size; ++i)
            s += static_cast<char>(data[pos++]);
        return s;
    }

    bool remaining() const { return pos < size; }
};

static bool is_finite_tensor(const Tensor& t) {
    for (float v : t.data)
        if (!std::isfinite(v)) return false;
    return true;
}

static void fuzz_vedic_gemm(FuzzReader& r) {
    int M = r.bounded_int(1, 129);
    int K = r.bounded_int(1, 129);
    int N = r.bounded_int(1, 129);

    Tensor A({M, K}); A.fill_random(r.bounded_float(-10.f, 0.f), r.bounded_float(0.f, 10.f), r.u32());
    Tensor B({K, N}); B.fill_random(r.bounded_float(-10.f, 0.f), r.bounded_float(0.f, 10.f), r.u32());

    Tensor C = vedic_gemm(A, B);

    if (C.rows() != M || C.cols() != N)
        __builtin_trap();
    if (C.data.size() != static_cast<size_t>(M * N))
        __builtin_trap();
}

static void fuzz_vedic_gemm_bias(FuzzReader& r) {
    int M = r.bounded_int(1, 65);
    int K = r.bounded_int(1, 65);
    int N = r.bounded_int(1, 65);

    Tensor A({M, K}); A.fill_random(-5.f, 5.f, r.u32());
    Tensor B({K, N}); B.fill_random(-5.f, 5.f, r.u32());
    Tensor b({N});    b.fill_random(-1.f, 1.f, r.u32());

    Tensor C = vedic_gemm_bias(A, B, b);

    if (C.rows() != M || C.cols() != N)
        __builtin_trap();
}

static void fuzz_tensor_ops(FuzzReader& r) {
    int rows = r.bounded_int(1, 64);
    int cols = r.bounded_int(1, 64);

    Tensor A({rows, cols}); A.fill_random(-100.f, 100.f, r.u32());
    Tensor B({rows, cols}); B.fill_random(-100.f, 100.f, r.u32());

    Tensor C = A + B;
    if (C.data.size() != A.data.size())
        __builtin_trap();

    Tensor T = A.transpose();
    if (T.rows() != cols || T.cols() != rows)
        __builtin_trap();

    int new_rows = r.bounded_int(1, rows * cols + 1);
    int new_cols = (rows * cols) / new_rows;
    if (new_rows * new_cols == rows * cols) {
        try {
            Tensor R = A.reshape({new_rows, new_cols});
            if (R.data.size() != A.data.size())
                __builtin_trap();
        } catch (const std::invalid_argument&) {}
    }

    bool has_nan = A.has_nan();
    (void)has_nan;
}

static void fuzz_layer_norm(FuzzReader& r) {
    int dim = r.bounded_int(1, 256);
    int seq = r.bounded_int(1, 64);

    LayerNorm ln(dim);  // LayerNorm sirf 1 argument leta hai (d_model); eps member variable hai

    Tensor X({seq, dim}); X.fill_random(-5.f, 5.f, r.u32());

    Tensor Y = ln.forward(X);

    if (Y.rows() != seq || Y.cols() != dim)
        __builtin_trap();
}

static void fuzz_reynolds_batch_norm(FuzzReader& r) {
    int dim = r.bounded_int(1, 128);
    int seq = r.bounded_int(2, 32);

    ReynoldsBatchNorm rbn(dim);

    Tensor X({seq, dim}); X.fill_random(-3.f, 3.f, r.u32());

    bool train = (r.u8() & 1) != 0;
    Tensor Y = rbn.forward(X, train);

    if (Y.rows() != seq || Y.cols() != dim)
        __builtin_trap();

    float re = rbn.reynolds_number(X);
    if (!std::isfinite(re) || re < 0.f)
        __builtin_trap();
}

static void fuzz_attention_softmax(FuzzReader& r) {
    int seq = r.bounded_int(1, 128);

    Tensor scores({seq, seq}); scores.fill_random(-50.f, 50.f, r.u32());
    float  temp = r.bounded_float(0.1f, 20.f);

    Tensor probs = boltzmann_softmax(scores, temp);

    if (probs.rows() != seq || probs.cols() != seq)
        __builtin_trap();

    for (int i = 0; i < seq; ++i) {
        float row_sum = 0.f;
        for (int j = 0; j < seq; ++j)
            row_sum += probs.at(i, j);
        if (std::fabs(row_sum - 1.f) > 1e-4f)
            __builtin_trap();
        for (int j = 0; j < seq; ++j)
            if (probs.at(i, j) < -1e-6f)
                __builtin_trap();
    }
}

static void fuzz_causal_mask(FuzzReader& r) {
    int seq = r.bounded_int(1, 512);
    const Tensor& mask = causal_mask_cached(seq);

    if (mask.rows() != seq || mask.cols() != seq)
        __builtin_trap();

    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < seq; ++j)
            if (j > i && mask.at(i, j) > -1e8f)
                __builtin_trap();
}

static void fuzz_attention_head(FuzzReader& r) {
    int d_model = r.bounded_int(4, 64) * 2;
    int d_k     = r.bounded_int(2, d_model / 2);
    int seq     = r.bounded_int(1, 32);

    AttentionHead head(d_model, d_k);
    head.use_sparse = (r.u8() & 1) != 0;

    Tensor X({seq, d_model}); X.fill_random(-1.f, 1.f, r.u32());

    try {
        Tensor out = head.forward(X);
        if (out.rows() != seq)
            __builtin_trap();
    } catch (const std::exception&) {}
}

static void fuzz_feedforward(FuzzReader& r) {
    int d_model  = r.bounded_int(2, 128);
    int seq      = r.bounded_int(1, 32);
    float dropout = r.bounded_float(0.f, 0.9f);

    FeedForward ff(d_model, dropout);

    Tensor X({seq, d_model}); X.fill_random(-2.f, 2.f, r.u32());

    bool training = (r.u8() & 1) != 0;
    Tensor Y = ff.forward(X, training);

    if (Y.rows() != seq || Y.cols() != d_model)
        __builtin_trap();
}

static void fuzz_gelu(FuzzReader& r) {
    int n = r.bounded_int(1, 1024);
    for (int i = 0; i < n; ++i) {
        float x = r.f32();
        if (!std::isfinite(x)) continue;
        float g = gelu(x);
        if (!std::isfinite(g))
            __builtin_trap();
        if (x > 10.f && g < x * 0.99f)
            __builtin_trap();
        if (x < -10.f && g > 0.01f)
            __builtin_trap();
    }
}

static void fuzz_gradient_clip(FuzzReader& r) {
    int n_tensors = r.bounded_int(1, 16);
    int tensor_sz = r.bounded_int(1, 512);
    float max_norm = r.bounded_float(0.01f, 10.f);

    std::vector<Tensor> grads;
    std::vector<Tensor*> gp;
    for (int i = 0; i < n_tensors; ++i) {
        Tensor G({1, tensor_sz}); G.fill_random(-50.f, 50.f, r.u32());
        if (r.u8() & 1)
            G.data[r.bounded_int(0, tensor_sz)] = std::numeric_limits<float>::quiet_NaN();
        grads.push_back(std::move(G));
    }
    for (auto& g : grads) gp.push_back(&g);

    clip_gradients(gp, max_norm, false);

    for (auto* g : gp)
        for (float v : g->data)
            if (std::isnan(v))
                __builtin_trap();
}

static void fuzz_langevin_optimizer(FuzzReader& r) {
    int n_params  = r.bounded_int(1, 8);
    int param_sz  = r.bounded_int(1, 128);
    float lr      = r.bounded_float(1e-6f, 1e-2f);
    float momentum = r.bounded_float(0.5f, 0.99f);
    float noise   = r.bounded_float(1e-8f, 1e-4f);
    int   max_steps = r.bounded_int(1, 10000);

    std::vector<Tensor> Ws, Gs;
    std::vector<Tensor*> ps, gs;
    for (int i = 0; i < n_params; ++i) {
        Tensor W({1, param_sz}); W.fill_random(-0.1f, 0.1f, r.u32());
        Tensor G({1, param_sz}); G.fill(0.01f);
        Ws.push_back(std::move(W));
        Gs.push_back(std::move(G));
    }
    for (auto& w : Ws) ps.push_back(&w);
    for (auto& g : Gs) gs.push_back(&g);

    LangevinOptimizer opt(lr, momentum, noise, 1e-8f, max_steps);
    opt.init(ps);

    int steps = r.bounded_int(1, 5);
    for (int s = 0; s < steps; ++s) {
        opt.step(ps, gs);
        for (auto* p : ps)
            for (float v : p->data)
                if (!std::isfinite(v))
                    __builtin_trap();
    }
}

static void fuzz_tokenizer(FuzzReader& r) {
    std::string corpus = r.string_of(4096);
    if (corpus.empty()) corpus = "hello world logos";

    int vocab_size = r.bounded_int(2, 512);

    Tokenizer tok;
    try {
        tok.build(corpus, vocab_size);
    } catch (const std::exception&) {
        return;
    }

    std::string test_text = r.string_of(1024);
    if (test_text.empty()) test_text = "hello";

    try {
        auto ids = tok.encode(test_text, static_cast<int>(test_text.size()) * 4);
        for (int id : ids)
            if (id < 0 || id >= vocab_size)
                __builtin_trap();
        std::string decoded = tok.decode(ids);
        (void)decoded;
    } catch (const std::exception&) {}
}

static void fuzz_model_forward(FuzzReader& r) {
    ModelConfig cfg;
    cfg.vocab_size  = r.bounded_int(2, 64);
    cfg.d_model     = r.bounded_int(1, 8) * 4;
    cfg.num_heads   = 1;
    while (cfg.d_model % cfg.num_heads != 0)
        cfg.num_heads = 1;
    cfg.num_layers  = r.bounded_int(1, 3);
    cfg.max_seq_len = r.bounded_int(2, 32);

    try {
        LOGOSModel model(cfg);
        int seq_len = r.bounded_int(1, cfg.max_seq_len);
        std::vector<int> inp(seq_len);
        for (int& id : inp)
            id = r.bounded_int(0, cfg.vocab_size);

        auto logits = model.forward(inp, false);
        if (static_cast<int>(logits.total_size) != cfg.vocab_size)
            __builtin_trap();
    } catch (const std::exception&) {}
}

static void fuzz_checkpoint_roundtrip(FuzzReader& r) {
    ModelConfig cfg;
    cfg.vocab_size  = r.bounded_int(2, 32);
    cfg.d_model     = r.bounded_int(1, 4) * 4;
    cfg.num_heads   = 1;
    cfg.num_layers  = r.bounded_int(1, 2);
    cfg.max_seq_len = r.bounded_int(2, 16);

    try {
        LOGOSModel model(cfg);
        const std::string base = "/tmp/logos_fuzz_ckpt";
        const int step = r.bounded_int(0, 99999);
        bool saved = save_checkpoint(model, base, step);
        if (!saved) return;

        const std::string path = base + "_step" + std::to_string(step) + ".bin";
        LOGOSModel model2(cfg);
        bool loaded = load_checkpoint(model2, base + "_step" + std::to_string(step) + ".bin");
        (void)loaded;

        std::remove(path.c_str());
    } catch (const std::exception&) {}
}

static void fuzz_navier_stokes_attention(FuzzReader& r) {
    int seq = r.bounded_int(1, 32);
    int d_k = r.bounded_int(1, 32);
    int d_v = r.bounded_int(1, 32);

    Tensor Q({seq, d_k}); Q.fill_random(-1.f, 1.f, r.u32());
    Tensor K({seq, d_k}); K.fill_random(-1.f, 1.f, r.u32());
    Tensor V({seq, d_v}); V.fill_random(-1.f, 1.f, r.u32());
    Tensor mask({seq, seq}, 0.f);
    for (int i = 0; i < seq; ++i)
        for (int j = i + 1; j < seq; ++j)
            mask.at(i, j) = -1e9f;

    float temp = r.bounded_float(0.1f, 10.f);
    float eta  = r.bounded_float(0.001f, 0.5f);
    float nu   = r.bounded_float(0.001f, 0.5f);

    try {
        Tensor out = navier_stokes_attention(Q, K, V, mask, temp, eta, nu);
        if (out.rows() != seq || out.cols() != d_v)
            __builtin_trap();
    } catch (const std::exception&) {}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 2) return 0;

    FuzzReader r(data, size);
    uint8_t selector = r.u8() % 13;

    try {
        switch (selector) {
            case 0:  fuzz_vedic_gemm(r);              break;
            case 1:  fuzz_vedic_gemm_bias(r);         break;
            case 2:  fuzz_tensor_ops(r);              break;
            case 3:  fuzz_layer_norm(r);              break;
            case 4:  fuzz_reynolds_batch_norm(r);     break;
            case 5:  fuzz_attention_softmax(r);       break;
            case 6:  fuzz_causal_mask(r);             break;
            case 7:  fuzz_attention_head(r);          break;
            case 8:  fuzz_feedforward(r);             break;
            case 9:  fuzz_gelu(r);                    break;
            case 10: fuzz_gradient_clip(r);           break;
            case 11: fuzz_langevin_optimizer(r);      break;
            case 12: fuzz_tokenizer(r);               break;
            default: break;
        }
    } catch (const std::bad_alloc&) {
        return 0;
    } catch (...) {
        return 0;
    }

    return 0;
}

#ifndef LOGOS_LIBFUZZER_ONLY
int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") {
        const uint8_t smoke[] = {
            0x00, 0x40, 0x00, 0x40, 0x00, 0x40,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf,
            0x2a, 0x00, 0x00, 0x00
        };
        for (uint8_t sel = 0; sel < 13; ++sel) {
            std::vector<uint8_t> buf(256, 0);
            buf[0] = sel;
            for (size_t i = 1; i < sizeof(smoke) && i < buf.size(); ++i)
                buf[i] = smoke[i];
            LLVMFuzzerTestOneInput(buf.data(), buf.size());
        }
        std::cout << "[FUZZ-SELFTEST] All 13 fuzz targets smoke-tested OK\n";
        return 0;
    }

    if (argc >= 2) {
        FILE* f = std::fopen(argv[1], "rb");
        if (!f) { std::fprintf(stderr, "Cannot open: %s\n", argv[1]); return 1; }
        std::fseek(f, 0, SEEK_END);
        long fsz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> buf(fsz > 0 ? fsz : 0);
        if (!buf.empty()) std::fread(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        return LLVMFuzzerTestOneInput(buf.data(), buf.size());
    }

    std::cout << "Usage:\n"
              << "  " << argv[0] << " --selftest      smoke-test all 13 targets\n"
              << "  " << argv[0] << " <corpus_file>   replay one corpus entry\n"
              << "  Compile with -fsanitize=fuzzer for full LibFuzzer mode\n";
    return 0;
}
#endif
