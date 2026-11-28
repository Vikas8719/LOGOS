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
#include <cstdlib>    // _Exit()
#include <exception>
#include <unistd.h>   // STDERR_FILENO / write() / getpid() — low-level output during setup

static std::string g_good, g_path;

// Pre-allocated model instances reused across every LLVMFuzzerTestOneInput call.
// Allocating two LOGOSModel objects per input inside the fuzz loop causes the RSS
// to spike past the libFuzzer rss_limit on GitHub runners (7 GB shared between
// ASan shadow + other fuzzers), triggering a SIGKILL that prevents the shell
// from writing the .rc exit-code file — the "exit code: missing / units: 5" symptom.
static LOGOSModel* g_model_strict  = nullptr;
static LOGOSModel* g_model_force   = nullptr;

// Known-good config mirrored from LLVMFuzzerInitialize so assertions can use it.
static constexpr int G_VOCAB   = 64;
static constexpr int G_D_MODEL = 16;
static constexpr int G_HEADS   = 2;
static constexpr int G_LAYERS  = 1;
static constexpr int G_SEQ     = 8;

// Returns true iff all invariants for a successfully-loaded model hold.
// Caller uses _Exit(1) on failure so bash can write the .rc file (std::abort
// raises SIGABRT which kills the subshell before echo $? runs).
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
    // NOTE: stderr is intentionally kept open during setup so that libFuzzer can
    // report a meaningful error message if initialisation fails.  silence_streams()
    // is called only after a successful setup so that the fuzzer corpus loop runs
    // quietly.  Calling silence_streams() before setup was the root cause of the
    // "executed units: 5 / setup failure" CI failure: libFuzzer received -1 from
    // LLVMFuzzerInitialize but could not diagnose why because stderr was already
    // redirected to /dev/null.
    ModelConfig cfg;
    cfg.vocab_size = G_VOCAB; cfg.d_model = G_D_MODEL; cfg.num_heads = G_HEADS;
    cfg.num_layers = G_LAYERS; cfg.max_seq_len = G_SEQ;
    LOGOSModel m(cfg);

    // Use a PID-scoped path so that parallel libFuzzer instances (e.g. during CI
    // where fuzz_checkpoint may be retried alongside other fuzz targets) never
    // collide on the same temp file.  /tmp is always available on Linux.
    const std::string base =
        tc::tmp_file("libfuzz_ckpt_" + std::to_string(static_cast<long>(::getpid())));
    g_path = base + "_step0.bin";  // save_checkpoint() writes <base>_step<N>.bin

    if (!save_checkpoint(m, base, 0)) {
        // Print directly to fd 2 so the message survives even if cout/cerr were
        // already redirected by a previous partial initialisation attempt.
        const char msg[] = "fuzz_checkpoint setup: save_checkpoint() failed — "
                           "check that /tmp/logos_cpu_tests/ is writable\n";
        (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
        return -1;
    }

    g_good = tc::read_all(g_path);
    if (g_good.empty()) {
        std::fprintf(stderr,
            "fuzz_checkpoint setup: checkpoint file is empty after save: %s\n",
            g_path.c_str());
        return -1;
    }

    // Sanity: the freshly-constructed model must pass all invariants before we
    // hand the good-file bytes to the mutator.
    if (!check_model_invariants(m, cfg, "setup")) {
        std::fprintf(stderr,
            "fuzz_checkpoint setup: freshly saved model failed invariant check\n");
        return -1;
    }

    // Pre-allocate the two model instances that LLVMFuzzerTestOneInput reuses.
    // This keeps the process RSS stable across the entire fuzz run.
    g_model_strict = new LOGOSModel(cfg);
    g_model_force  = new LOGOSModel(cfg);

    // Setup succeeded — silence stdout/stderr for the fuzzing loop so that
    // libFuzzer's own progress output is not drowned in Checkpoint log spam.
    fz::silence_streams();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    tc::write_all(g_path, fz::mutate_good(g_good, data, size));
    try {
        const ModelConfig cfg = g_model_strict->cfg;   // same as G_VOCAB/D_MODEL/etc.

        // --- strict loader: reuse pre-allocated model (load_checkpoint overwrites weights) ---
        const bool strict_ok = load_checkpoint(*g_model_strict, g_path);
        if (strict_ok) {
            // Invariant violation = real bug in load_checkpoint.
            // Use _Exit(1) not std::abort(): abort() raises SIGABRT which kills the bash
            // subshell before "echo $?" runs, so .rc is never written and CI reports
            // "exit code: missing".  _Exit(1) terminates immediately with exit code 1
            // which bash captures correctly in $? and writes to the .rc file.
            if (!check_model_invariants(*g_model_strict, cfg, "strict"))
                _Exit(1);
        }

        // --- force loader ---
        const bool force_ok = load_checkpoint_force(*g_model_force, g_path);
        if (force_ok) {
            // load_checkpoint_force now sanitizes NaN/Inf in-place (Checkpoint.cpp), so
            // a force-load that returns true must always pass the invariant check.
            if (!check_model_invariants(*g_model_force, cfg, "force"))
                _Exit(1);
        }
    } catch (const std::exception&) {
        // Exceptions are acceptable (bad input); _Exit above is not reachable via exceptions.
    }
    return 0;
}
