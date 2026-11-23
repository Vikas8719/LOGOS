// libFuzzer target: Tokenizer encode/decode/pre-tokenizer on arbitrary bytes.
#include "fuzz_common.hpp"
#include "Tokenizer.hpp"

#include <algorithm>
#include <vector>

static Tokenizer* g_tok = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    fz::silence_streams();
    g_tok = new Tokenizer();
    g_tok->build(tc::make_corpus(120 * 1024, 1), 700);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const Tokenizer& t = *g_tok;
    const std::string s(reinterpret_cast<const char*>(data), size);

    // round trip + id range
    const std::vector<int> ids = t.encode_raw(s);
    FZ_ASSERT(t.decode(ids) == s);
    for (int id : ids) FZ_ASSERT(id >= 4 && id < t.vocab_size);

    // pre-tokenizer pieces: concat == input, each 1..MAX_PIECE_BYTES
    std::vector<size_t> offsets{0};
    std::string cat;
    bool sizes_ok = true;
    Tokenizer::for_each_piece(s.data(), s.size(), [&](const char* p, size_t l) {
        if (l == 0 || l > Tokenizer::MAX_PIECE_BYTES) sizes_ok = false;
        cat.append(p, l);
        offsets.push_back(cat.size());
        return true;
    });
    FZ_ASSERT(sizes_ok);
    FZ_ASSERT(cat == s);

    // additivity at a piece boundary chosen from the input itself
    const size_t cut = offsets[(size ? data[0] : 0) % offsets.size()];
    std::vector<int> left = t.encode_raw(s.substr(0, cut));
    const std::vector<int> right = t.encode_raw(s.substr(cut));
    left.insert(left.end(), right.begin(), right.end());
    FZ_ASSERT(left == ids);

    // max_tokens prefix property
    const int64_t m = static_cast<int64_t>((size > 1 ? data[1] : 0) % (ids.size() + 3));
    const std::vector<int> pre = t.encode_raw(s, m);
    const size_t want = std::min<size_t>(static_cast<size_t>(m), ids.size());
    FZ_ASSERT(pre.size() == want);
    FZ_ASSERT(std::equal(pre.begin(), pre.end(), ids.begin()));

    // legacy encode(): BOS..EOS framing and length cap
    const int L = static_cast<int>((size > 2 ? data[2] : 0) % 41);
    const std::vector<int> e = t.encode(s, L);
    FZ_ASSERT(static_cast<int>(e.size()) <= L);
    if (L >= 2) FZ_ASSERT(e.size() >= 2 && e.front() == TOKEN_BOS && e.back() == TOKEN_EOS);
    return 0;
}
