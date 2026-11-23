// libFuzzer target: corrupted vocab.bin must never crash Tokenizer::load()/encode()/decode().
#include "fuzz_common.hpp"
#include "Tokenizer.hpp"

static std::string g_good, g_path, g_sample;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    fz::silence_streams();
    Tokenizer t;
    t.build(tc::make_corpus(60 * 1024, 3), 600);
    g_path = tc::tmp_file("libfuzz_vocab.bin");
    t.save(g_path);
    g_good = tc::read_all(g_path);
    g_sample = tc::make_corpus(2000, 4);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // half of the inputs: raw bytes; other half: one known-good vocab with targeted edits
    const std::string d = (size && (data[0] & 0x80))
        ? std::string(reinterpret_cast<const char*>(data), size)
        : fz::mutate_good(g_good, data, size);
    tc::write_all(g_path, d);
    Tokenizer t;
    if (t.load(g_path)) {
        const std::vector<int> ids = t.encode_raw(g_sample);
        (void)t.decode(ids);
        (void)t.fingerprint();
    }
    return 0;
}
