// libFuzzer target: corrupted checkpoint file must never crash load_checkpoint()/load_checkpoint_force().
// (Rejecting with `false` or an exception is fine; UB / OOB / huge allocations / hangs are bugs.)
#include "fuzz_common.hpp"
#include "Checkpoint.hpp"

#include <exception>

static std::string g_good, g_path;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    fz::silence_streams();
    ModelConfig cfg;
    cfg.vocab_size = 64; cfg.d_model = 16; cfg.num_heads = 2; cfg.num_layers = 1; cfg.max_seq_len = 8;
    LOGOSModel m(cfg);
    g_path = tc::tmp_file("libfuzz_ckpt.bin");
    if (!save_checkpoint(m, g_path, 0)) { std::fprintf(stderr, "setup: save_checkpoint failed\n"); std::exit(2); }
    g_good = tc::read_all(g_path);
    if (g_good.empty()) { std::fprintf(stderr, "setup: checkpoint file empty at %s\n", g_path.c_str()); std::exit(2); }
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    tc::write_all(g_path, fz::mutate_good(g_good, data, size));
    try {
        ModelConfig cfg;
        cfg.vocab_size = 64; cfg.d_model = 16; cfg.num_heads = 2; cfg.num_layers = 1; cfg.max_seq_len = 8;
        LOGOSModel a(cfg);
        (void)load_checkpoint(a, g_path);
        LOGOSModel b(cfg);
        (void)load_checkpoint_force(b, g_path);
    } catch (const std::exception&) {
    }
    return 0;
}
