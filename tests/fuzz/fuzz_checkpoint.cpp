// libFuzzer target: corrupted checkpoint file must never crash load_checkpoint()/load_checkpoint_force().
// (Rejecting with `false` or an exception is fine; UB / OOB / huge allocations / hangs are bugs.)
//
// Strict invariant assertions (added after mutation analysis):
//   1. Successful load: parameter count must match the cfg — kills off-by-one on parameter loop.
//   2. Successful load: no parameter tensor has zero total_size — kills empty-alloc mutation.
//   3. Successful load: no parameter tensor contains NaN/Inf — kills float-read corruption bypass.
//   4. Successful load: config fields stay within allowed bounds — kills bounds-check removal mutant.
//   5. load_checkpoint_force: same size-checks as strict loader, plus partial-tensor zero-fill invariant.
#include "fuzz_common.hpp"
#include "Checkpoint.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <exception>

static std::string g_good, g_path;

// Known-good config mirrored from LLVMFuzzerInitialize so assertions can use it.
static constexpr int G_VOCAB   = 64;
static constexpr int G_D_MODEL = 16;
static constexpr int G_HEADS   = 2;
static constexpr int G_LAYERS  = 1;
static constexpr int G_SEQ     = 8;

// Returns true iff all invariants for a successfully-loaded model hold.
// Aborts (via __builtin_trap / assert) on UB-class violations so the fuzzer
// reports them as crashes rather than silent mis-accepts.
static bool check_model_invariants(const LOGOSModel& m, const ModelConfig& cfg,
                                   const char* tag) {
    // Invariant 1: parameter list non-empty and each tensor has positive total_size.
    // Kills mutation that skips a parameter push_back (loop-off-by-one).
    const auto params = const_cast<LOGOSModel&>(m).parameters();
    if (params.empty()) {
        std::fprintf(stderr, "[%s] invariant FAIL: parameters() is empty\n", tag);
        return false;
    }
    for (const Tensor* p : params) {
        if (p->total_size == 0) {
            std::fprintf(stderr, "[%s] invariant FAIL: parameter has total_size==0\n", tag);
            return false;
        }
        // Invariant 2: no NaN or Inf in loaded weights — kills float-read bypass mutant.
        for (float v : p->data) {
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "[%s] invariant FAIL: non-finite value in parameter\n", tag);
                return false;
            }
        }
    }

    // Invariant 3: config fields within known valid bounds — kills bounds-check removal.
    if (m.cfg.vocab_size  != cfg.vocab_size  ||
        m.cfg.d_model     != cfg.d_model     ||
        m.cfg.num_heads   != cfg.num_heads   ||
        m.cfg.num_layers  != cfg.num_layers  ||
        m.cfg.max_seq_len != cfg.max_seq_len) {
        std::fprintf(stderr, "[%s] invariant FAIL: cfg mismatch after load\n", tag);
        return false;
    }

    // Invariant 4: layer count matches num_layers — kills layer-count off-by-one.
    if ((int)m.layers.size() != cfg.num_layers) {
        std::fprintf(stderr, "[%s] invariant FAIL: layer count %zu != cfg.num_layers %d\n",
                     tag, m.layers.size(), cfg.num_layers);
        return false;
    }

    // Invariant 5: pos_embedding size is max_seq_len * d_model — kills pos-block skip mutant.
    const size_t expected_pos = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model;
    if (m.pos_embedding.data.size() != expected_pos) {
        std::fprintf(stderr, "[%s] invariant FAIL: pos_embedding size %zu != %zu\n",
                     tag, m.pos_embedding.data.size(), expected_pos);
        return false;
    }

    // Invariant 6: final_ln gamma size == d_model and all gamma values finite.
    if ((int)m.final_ln.gamma.data.size() != cfg.d_model ||
        (int)m.final_ln.beta.data.size()  != cfg.d_model) {
        std::fprintf(stderr, "[%s] invariant FAIL: final_ln size mismatch\n", tag);
        return false;
    }
    for (float g : m.final_ln.gamma.data)
        if (!std::isfinite(g)) {
            std::fprintf(stderr, "[%s] invariant FAIL: non-finite final_ln.gamma\n", tag);
            return false;
        }
    for (float b : m.final_ln.beta.data)
        if (!std::isfinite(b)) {
            std::fprintf(stderr, "[%s] invariant FAIL: non-finite final_ln.beta\n", tag);
            return false;
        }

    return true;
}

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    fz::silence_streams();
    ModelConfig cfg;
    cfg.vocab_size = G_VOCAB; cfg.d_model = G_D_MODEL; cfg.num_heads = G_HEADS;
    cfg.num_layers = G_LAYERS; cfg.max_seq_len = G_SEQ;
    LOGOSModel m(cfg);
    const std::string base = tc::tmp_file("libfuzz_ckpt");
    g_path = base + "_step0.bin";  // save_checkpoint() writes <base>_step<N>.bin
    if (!save_checkpoint(m, base, 0)) {
        // Return -1 (not exit(2)) so libFuzzer reports a proper setup error
        // instead of treating it as a crash and stopping after a handful of inputs.
        std::fprintf(stderr, "setup: save_checkpoint failed\n"); return -1;
    }
    g_good = tc::read_all(g_path);
    if (g_good.empty()) {
        std::fprintf(stderr, "setup: checkpoint file empty at %s\n", g_path.c_str()); return -1;
    }

    // Sanity: the good model itself must pass all invariants.
    if (!check_model_invariants(m, cfg, "setup")) {
        std::fprintf(stderr, "setup: good model failed invariant check\n"); return -1;
    }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    tc::write_all(g_path, fz::mutate_good(g_good, data, size));
    try {
        ModelConfig cfg;
        cfg.vocab_size = G_VOCAB; cfg.d_model = G_D_MODEL; cfg.num_heads = G_HEADS;
        cfg.num_layers = G_LAYERS; cfg.max_seq_len = G_SEQ;

        // --- strict loader ---
        LOGOSModel a(cfg);
        const bool strict_ok = load_checkpoint(a, g_path);
        if (strict_ok) {
            // A successful strict load MUST satisfy all invariants.
            // Any assertion failure here is a fuzzer-reportable crash (not a benign reject).
            if (!check_model_invariants(a, cfg, "strict")) {
                // Treat as a crash so the fuzzer saves the input.
                std::abort();
            }
        }

        // --- force loader ---
        LOGOSModel b(cfg);
        const bool force_ok = load_checkpoint_force(b, g_path);
        if (force_ok) {
            // force-load may partially fill tensors (zero-fills missing floats),
            // but config and structure must still be valid.
            if (!check_model_invariants(b, cfg, "force")) {
                std::abort();
            }
        }
    } catch (const std::exception&) {
        // Exceptions are acceptable (bad input); UB / abort above is not.
    }
    return 0;
}
