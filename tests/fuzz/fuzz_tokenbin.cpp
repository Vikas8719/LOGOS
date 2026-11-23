// libFuzzer target: corrupted pre-tokenized .bin (header, truncation, payload).
// Header parsing, token_bin_matches() and TokenBinLoader must never crash / overflow / hang.
#include "fuzz_common.hpp"
#include "Tokenizer.hpp"
#include "TokenBin.hpp"

#include <exception>

static Tokenizer* g_tok = nullptr;
static std::string g_good, g_bad_path;
static int64_t g_src_size = 0;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    fz::silence_streams();
    g_tok = new Tokenizer();
    g_tok->build(tc::make_corpus(60 * 1024, 5), 600);

    const std::string src = tc::make_corpus(20 * 1024, 8);
    g_src_size = static_cast<int64_t>(src.size());
    const std::string txt = tc::tmp_file("libfuzz_bin_src.txt");
    const std::string bin = tc::tmp_file("libfuzz_bin_good.bin");
    tc::write_all(txt, src);
    if (!pretokenize_file(txt, *g_tok, bin)) { std::fprintf(stderr, "setup: pretokenize failed\n"); std::exit(2); }
    g_good = tc::read_all(bin);
    g_bad_path = tc::tmp_file("libfuzz_bin_bad.bin");
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    tc::write_all(g_bad_path, fz::mutate_good(g_good, data, size));

    (void)read_token_bin_header(g_bad_path);
    (void)token_bin_matches(g_bad_path, *g_tok, g_src_size);
    try {
        TokenBinLoader ld(g_bad_path, 8, 1);
        std::vector<int> in, tg;
        for (int j = 0; j < 8; ++j) (void)ld.next_batch(in, tg);
    } catch (const std::exception&) {
        // rejecting a corrupt file with an exception is the correct behaviour
    }
    return 0;
}
