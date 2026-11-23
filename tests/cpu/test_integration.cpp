// T23: Integration tests — pre-tokenization (.bin) + TokenBinLoader + StreamingDataLoader.
#include "test_common.hpp"
#include "TokenBin.hpp"          // includes StreamingDataLoader.hpp + Tokenizer.hpp
#include <filesystem>

static Tokenizer& tok() {
    static Tokenizer t;
    static bool built = false;
    if (!built) { t.build(tc::make_corpus(150 * 1024, 5), 700); built = true; }
    return t;
}

static std::vector<uint16_t> read_bin_tokens(const std::string& path, int64_t* n_out = nullptr) {
    TokenBinInfo info = read_token_bin_header(path);
    std::vector<uint16_t> v;
    if (!info.valid) return v;
    std::ifstream f(path, std::ios::binary);
    f.seekg(TOKBIN_HEADER_BYTES, std::ios::beg);
    v.resize(static_cast<size_t>(info.n_tokens));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 2));
    if (n_out) *n_out = info.n_tokens;
    return v;
}

static void test_pretokenize_chunk_equivalence(const std::string& corpus, const std::string& text_path) {
    TC_SECTION("pretokenize_file: chunked == whole-text tokenization");
    Tokenizer& t = tok();
    std::vector<int> whole = t.encode_raw(corpus);
    for (int64_t chunk : {4096LL, 10007LL, 65536LL, 1LL << 24}) {
        std::string bin = tc::tmp_file("pretok_" + std::to_string(chunk) + ".bin");
        bool ok;
        { tc::Silence s; ok = pretokenize_file(text_path, t, bin, chunk); }
        TC_CHECKF(ok, "pretokenize_file chunk=%lld", (long long)chunk);
        TokenBinInfo info = read_token_bin_header(bin);
        TC_CHECK(info.valid, "header valid");
        TC_CHECK(info.n_tokens == (int64_t)whole.size(), "n_tokens == whole-text token count");
        TC_CHECK(info.src_size == (int64_t)corpus.size(), "src_size recorded");
        TC_CHECK(info.vocab_size == (uint32_t)t.vocab_size, "vocab_size recorded");
        TC_CHECK(token_bin_matches(bin, t, (int64_t)corpus.size()), "token_bin_matches == true");
        auto toks = read_bin_tokens(bin);
        bool same = toks.size() == whole.size();
        for (size_t i = 0; same && i < toks.size(); ++i) same = (toks[i] == (uint16_t)whole[i]);
        TC_CHECKF(same, "chunk=%lld: bin tokens identical to whole-text encode_raw", (long long)chunk);
        TC_CHECK(!std::filesystem::exists(bin + ".tmp"), "no leftover .tmp file");
    }
}

static void test_bin_invalidation(const std::string& corpus, const std::string& text_path) {
    TC_SECTION("token_bin_matches invalidation");
    Tokenizer& t = tok();
    std::string bin = tc::tmp_file("inval.bin");
    bool made;
    { tc::Silence s; made = pretokenize_file(text_path, t, bin); }
    TC_CHECK(made, "pretokenize ok");
    int64_t src = (int64_t)corpus.size();
    TC_CHECK(token_bin_matches(bin, t, src), "baseline matches");
    TC_CHECK(!token_bin_matches(bin, t, src + 1), "different src size -> no match");
    TC_CHECK(!token_bin_matches(tc::tmp_file("nope.bin"), t, src), "missing file -> no match");

    Tokenizer other;
    { tc::Silence s; other.build(tc::make_corpus(100 * 1024, 999), 400); }
    TC_CHECK(!token_bin_matches(bin, other, src), "different tokenizer -> no match");

    std::string bytes = tc::read_all(bin);
    std::string cut = tc::tmp_file("inval_cut.bin");
    tc::write_all(cut, bytes.substr(0, bytes.size() - 2));
    TC_CHECK(!token_bin_matches(cut, t, src), "truncated by one token -> no match");
    {
        bool threw = false;
        try { TokenBinLoader x(cut, 16, 1); } catch (const std::runtime_error&) { threw = true; }
        TC_CHECK(threw, "TokenBinLoader rejects a truncated .bin");
    }
    tc::write_all(cut, bytes + std::string(2, '\0'));
    TC_CHECK(!token_bin_matches(cut, t, src), "extra trailing bytes -> no match");
}

// Synthetic .bin with known token values (loader does not check the fingerprint).
static std::string make_synth_bin(const std::string& name, int64_t n) {
    std::string path = tc::tmp_file(name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    tokbin_detail::write_header(out, tok(), 1, n);
    for (int64_t i = 0; i < n; ++i) {
        uint16_t v = static_cast<uint16_t>(i % 1000);
        out.write(reinterpret_cast<const char*>(&v), 2);
    }
    return path;
}

static void test_tokenbin_loader() {
    TC_SECTION("TokenBinLoader windows / state / epoch");
    const int L = 16;
    {
        // range = 2L+1 tokens -> exactly 2 full windows (off-by-one guard)
        std::string p = make_synth_bin("synth_exact.bin", 2 * L + 1);
        TokenBinLoader ld(p, L, 1);
        std::vector<int> in, tg;
        int n = 0;
        while (ld.next_batch(in, tg)) {
            for (int i = 0; i < L; ++i) {
                TC_CHECK(in[i] == (n * L + i) % 1000, "input = token[pos+i]");
                TC_CHECK(tg[i] == (n * L + i + 1) % 1000, "target = token[pos+i+1]");
            }
            ++n;
        }
        TC_CHECKF(n == 2, "windows in 2L+1 tokens = %d (want 2)", n);
        TC_CHECK(ld.current_epoch == 1, "epoch incremented at end");
        TC_CHECK(ld.next_batch(in, tg) && in[0] == 0, "loader rewound after epoch end");
    }
    {
        std::string p = make_synth_bin("synth_big.bin", 10000);
        TokenBinLoader ld(p, L, 4, 100, 1100);                // range of 1000 tokens
        TC_CHECK(ld.total_batches_per_epoch() == 1000 / L, "total_batches_per_epoch");
        std::vector<int> in, tg;
        TC_CHECK(ld.next_batch(in, tg) && in[0] == 100 % 1000, "start_tok respected");

        ld.reset();
        for (int i = 0; i < 5; ++i) ld.next_batch(in, tg);
        auto st = ld.get_state();
        std::vector<std::vector<int>> a, b;
        for (int i = 0; i < 3; ++i) { ld.next_batch(in, tg); a.push_back(in); }
        ld.restore_state(st);
        for (int i = 0; i < 3; ++i) { ld.next_batch(in, tg); b.push_back(in); }
        TC_CHECK(a == b, "restore_state replays identical batches");

        ld.reset();
        std::vector<std::pair<std::vector<int>, std::vector<int>>> mb;
        TC_CHECK(ld.next_accum_batch(mb) && mb.size() == 4, "next_accum_batch returns grad_accum micro-batches");
    }
    {
        bool t1 = false, t2 = false, t3 = false, t4 = false;
        std::string p = make_synth_bin("synth_small.bin", 100);
        try { TokenBinLoader x(p, 0, 1); } catch (const std::invalid_argument&) { t1 = true; }
        try { TokenBinLoader x(p, 8, 0); } catch (const std::invalid_argument&) { t2 = true; }
        try { TokenBinLoader x(tc::tmp_file("missing.bin"), 8, 1); } catch (const std::runtime_error&) { t3 = true; }
        std::string tiny = make_synth_bin("synth_tiny.bin", L + 1);   // range < L+2
        try { TokenBinLoader x(tiny, L, 1); } catch (const std::invalid_argument&) { t4 = true; }
        TC_CHECK(t1, "seq_len=0 throws");
        TC_CHECK(t2, "grad_accum=0 throws");
        TC_CHECK(t3, "missing file throws");
        TC_CHECK(t4, "range too small throws");
    }
}

static void test_ws_boundary_regression() {
    TC_SECTION("chunk cut inside a mixed-whitespace run (\\v / \\f) must not change tokens");
    Tokenizer& t = tok();
    // pehle chunk (4096 B) ka end ' \v ' run ke beech me padta hai
    std::string text = std::string(4090, 'a') + "\v \v" + std::string(300, 'b') + " tail\f \n end\n";
    std::string path = tc::tmp_file("ws_boundary.txt");
    tc::write_all(path, text);
    std::string bin = tc::tmp_file("ws_boundary.bin");
    bool ok;
    { tc::Silence s; ok = pretokenize_file(path, t, bin, 4096); }
    TC_CHECK(ok, "pretokenize ok");
    auto toks = read_bin_tokens(bin);
    auto whole = t.encode_raw(text);
    bool same = toks.size() == whole.size();
    for (size_t i = 0; same && i < toks.size(); ++i) same = (toks[i] == (uint16_t)whole[i]);
    TC_CHECK(same, "chunked tokens == whole-text tokens across a \\v/space run");
}

static void test_streaming_loader(const std::string& corpus, const std::string& text_path) {
    TC_SECTION("StreamingDataLoader");
    Tokenizer& t = tok();
    const int L = 16;
    std::vector<int> whole = t.encode_raw(corpus.substr(0, 20000));
    {
        StreamingDataLoader ld(text_path, t, L, 2, 8192);
        std::vector<int> in, tg;
        TC_CHECK(ld.next_batch(in, tg), "first batch");
        bool first_ok = in.size() == (size_t)L;
        for (int i = 0; first_ok && i < L; ++i) first_ok = (in[i] == whole[i]) && (tg[i] == whole[i + 1]);
        TC_CHECK(first_ok, "first batch == first tokens of the text");

        int64_t batches = 1;
        bool shifted = true, valid = true;
        while (ld.next_batch(in, tg)) {
            ++batches;
            for (int i = 0; i + 1 < L; ++i) if (tg[i] != in[i + 1]) shifted = false;
            for (int i = 0; i < L; ++i)
                if (in[i] < 0 || in[i] >= t.vocab_size || tg[i] < 0 || tg[i] >= t.vocab_size) valid = false;
        }
        TC_CHECK(shifted, "target[i] == input[i+1] for every batch");
        TC_CHECK(valid, "all ids within [0,vocab)");
        TC_CHECK(ld.current_epoch == 1, "epoch counter incremented at end of data");
        TC_CHECKF(batches > 100, "enough batches in an epoch (%lld)", (long long)batches);
        std::vector<std::pair<std::vector<int>, std::vector<int>>> mb;
        TC_CHECK(ld.next_accum_batch(mb) && mb.size() == 2, "next_accum_batch after rewind");
    }
    {
        // vs TokenBinLoader: same first window
        std::string bin = tc::tmp_file("stream_vs_bin.bin");
        { tc::Silence s; pretokenize_file(text_path, t, bin); }
        TokenBinLoader bl(bin, L, 1);
        StreamingDataLoader sl(text_path, t, L, 1, 8192);
        std::vector<int> i1, t1, i2, t2;
        TC_CHECK(bl.next_batch(i1, t1) && sl.next_batch(i2, t2) && i1 == i2 && t1 == t2,
                 "TokenBinLoader and StreamingDataLoader agree on batch 0");
    }
    {
        std::string empty = tc::tmp_file("empty.txt");
        tc::write_all(empty, "");
        bool e1 = false, e2 = false, e3 = false, e4 = false;
        try { StreamingDataLoader x(empty, t, L, 1); } catch (const std::runtime_error&) { e1 = true; }
        try { StreamingDataLoader x(tc::tmp_file("missing.txt"), t, L, 1); } catch (const std::runtime_error&) { e2 = true; }
        try { StreamingDataLoader x(text_path, t, 0, 1); } catch (const std::invalid_argument&) { e3 = true; }
        try { StreamingDataLoader x(text_path, t, L, 0); } catch (const std::invalid_argument&) { e4 = true; }
        TC_CHECK(e1, "empty file throws");
        TC_CHECK(e2, "missing file throws");
        TC_CHECK(e3, "seq_len=0 throws");
        TC_CHECK(e4, "grad_accum=0 throws");
    }
    {
        // construct/destroy while prefetch thread is busy: must neither hang nor crash
        for (int i = 0; i < 25; ++i) {
            StreamingDataLoader x(text_path, t, L, 1, 8192);
            std::vector<int> in, tg;
            x.next_batch(in, tg);
        }
        TC_CHECK(true, "25x construct/destroy with live prefetch thread");
    }
}

int main() {
    std::printf("LOGOS T23: integration tests (pretokenize + loaders)\n");
    std::string corpus = tc::make_corpus(400 * 1024, 31);
    // \v, \f aur CRLF bhi chunk boundaries par aayen (whitespace-consistency regression guard)
    for (size_t pos = 3000; pos + 8 < corpus.size(); pos += 7919) {
        if (corpus[pos] == ' ') corpus.replace(pos, 1, (pos / 7919) % 2 ? "\v " : "\r\n");
    }
    std::string text_path = tc::tmp_file("integration_corpus.txt");
    tc::write_all(text_path, corpus);

    test_pretokenize_chunk_equivalence(corpus, text_path);
    test_bin_invalidation(corpus, text_path);
    test_ws_boundary_regression();
    test_tokenbin_loader();
    test_streaming_loader(corpus, text_path);

    std::error_code ec;
    std::filesystem::remove(text_path, ec);
    return tc::finish("T23_Integration");
}
