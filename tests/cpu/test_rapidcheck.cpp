// T25: RapidCheck property-based tests.
//   RC_PARAMS="max_success=1000"  -> 1000 examples per property (set below if RC_PARAMS is unset)
//   RC_PARAMS="seed=<n>"          -> reproduce a failure (RapidCheck prints the seed it used)
// Every failing property is shrunk to a minimal counterexample by RapidCheck.
#include <rapidcheck.h>

#include "test_common.hpp"
#include "TokenBin.hpp"
#include "PathIntegral.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// ---- shared fixtures ---------------------------------------------------------------------
static const Tokenizer& tok() {
    static const Tokenizer t = [] {
        Tokenizer x;
        tc::Silence quiet;
        x.build(tc::make_corpus(120 * 1024, 1), 700);
        return x;
    }();
    return t;
}

// Same tokenizer, but passed through save() -> load().
static const Tokenizer& tok_reloaded() {
    static const Tokenizer t = [] {
        const std::string path = tc::tmp_file("rc_vocab.bin");
        { tc::Silence quiet; tok().save(path); }
        Tokenizer x;
        bool ok = false;
        { tc::Silence quiet; ok = x.load(path); }
        if (!ok) { std::fprintf(stderr, "tok_reloaded: load() failed\n"); std::exit(2); }
        return x;
    }();
    return t;
}

// ---- generators --------------------------------------------------------------------------
// Whitespace-heavy ASCII (exercises the pre-tokenizer's whitespace rules).
static rc::Gen<std::string> genWsText() {
    return rc::gen::container<std::string>(
        rc::gen::element<char>(' ', '\n', '\t', '\r', '\v', '\f', 'a', 'b', 'c'));
}

// English + Hindi/CJK UTF-8 fragments glued together.
static rc::Gen<std::string> genWordsText() {
    static const std::vector<std::string> W = {
        "the", "quick", "fox", " ", "  ", "\n", "\n\n", "\t", "ab", ".", ",",
        "\xE0\xA4\x85", "\xE0\xA4\x95", "\xC3\xA9", "\xE5\xAD\x97"};
    return rc::gen::map(
        rc::gen::container<std::vector<std::string>>(rc::gen::elementOf(W)),
        [](const std::vector<std::string>& v) {
            std::string s;
            for (const auto& x : v) s += x;
            return s;
        });
}

// Normal values plus NaN / Inf / negatives / zero (bounded so sums stay finite).
static rc::Gen<float> genFloatNasty() {
    return rc::gen::oneOf(
        rc::gen::map(rc::gen::inRange<int>(0, 100001),
                     [](int x) { return static_cast<float>(x) * 0.01f; }),
        rc::gen::just(std::numeric_limits<float>::quiet_NaN()),
        rc::gen::just(std::numeric_limits<float>::infinity()),
        rc::gen::just(-1.0f),
        rc::gen::just(0.0f),
        rc::gen::just(-1000.0f));
}

// ---- shared property bodies ---------------------------------------------------------------
static void prop_roundtrip(const std::string& s) {
    const Tokenizer& t = tok();
    const std::vector<int> ids = t.encode_raw(s);
    RC_ASSERT(t.decode(ids) == s);
    for (int id : ids) RC_ASSERT(id >= 4 && id < t.vocab_size);
}

static void prop_pieces(const std::string& s) {
    std::string cat;
    bool sizes_ok = true;
    Tokenizer::for_each_piece(s.data(), s.size(), [&](const char* p, size_t l) {
        if (l == 0 || l > Tokenizer::MAX_PIECE_BYTES) sizes_ok = false;
        cat.append(p, l);
        return true;
    });
    RC_ASSERT(sizes_ok);
    RC_ASSERT(cat == s);
}

int main() {
#ifdef _WIN32
    if (!std::getenv("RC_PARAMS")) _putenv_s("RC_PARAMS", "max_success=1000");
#else
    setenv("RC_PARAMS", "max_success=1000", 0);   // 0 = keep a value that is already set
#endif
    std::printf("LOGOS T25: RapidCheck (RC_PARAMS=%s)\n", std::getenv("RC_PARAMS"));
    bool ok = true;

    // ===== tokenizer =====
    ok = rc::check("tok: decode(encode_raw(x)) == x, ids in [4,vocab)  [arbitrary bytes]",
                   [](const std::string& s) { prop_roundtrip(s); }) && ok;
    ok = rc::check("tok: decode(encode_raw(x)) == x, ids in [4,vocab)  [whitespace-heavy]",
                   [] { prop_roundtrip(*genWsText()); }) && ok;
    ok = rc::check("tok: decode(encode_raw(x)) == x, ids in [4,vocab)  [UTF-8 words]",
                   [] { prop_roundtrip(*genWordsText()); }) && ok;

    ok = rc::check("tok: pre-tokenizer pieces concat == input, each 1..128 bytes  [arbitrary bytes]",
                   [](const std::string& s) { prop_pieces(s); }) && ok;
    ok = rc::check("tok: pre-tokenizer pieces concat == input, each 1..128 bytes  [UTF-8 words]",
                   [] { prop_pieces(*genWordsText()); }) && ok;

    ok = rc::check("tok: encoding is additive at piece boundaries", [] {
        const Tokenizer& t = tok();
        const std::string s = *genWordsText();
        std::vector<size_t> offsets{0};
        size_t pos = 0;
        Tokenizer::for_each_piece(s.data(), s.size(), [&](const char*, size_t l) {
            pos += l;
            offsets.push_back(pos);
            return true;
        });
        const size_t cut = offsets[*rc::gen::inRange<size_t>(0, offsets.size())];
        std::vector<int> left  = t.encode_raw(s.substr(0, cut));
        const std::vector<int> right = t.encode_raw(s.substr(cut));
        left.insert(left.end(), right.begin(), right.end());
        RC_ASSERT(left == t.encode_raw(s));
    }) && ok;

    ok = rc::check("tok: encode_raw(x, m) is the first min(m, n) ids of encode_raw(x)", [] {
        const Tokenizer& t = tok();
        const std::string s = *genWordsText();
        const std::vector<int> ids = t.encode_raw(s);
        const int64_t m = *rc::gen::inRange<int64_t>(0, static_cast<int64_t>(ids.size()) + 3);
        const std::vector<int> pre = t.encode_raw(s, m);
        const size_t want = std::min<size_t>(static_cast<size_t>(m), ids.size());
        RC_ASSERT(pre.size() == want);
        RC_ASSERT(std::equal(pre.begin(), pre.end(), ids.begin()));
    }) && ok;

    ok = rc::check("tok: encode(x, L) has <= L ids and BOS..EOS framing when L >= 2",
                   [](const std::string& s) {
        const Tokenizer& t = tok();
        const int L = *rc::gen::inRange<int>(0, 41);
        const std::vector<int> e = t.encode(s, L);
        RC_ASSERT(static_cast<int>(e.size()) <= L);
        if (L >= 2) {
            RC_ASSERT(e.size() >= 2);
            RC_ASSERT(e.front() == TOKEN_BOS);
            RC_ASSERT(e.back() == TOKEN_EOS);
        }
    }) && ok;

    ok = rc::check("tok: save()/load() keeps fingerprint and encoding", [](const std::string& s) {
        RC_ASSERT(tok().fingerprint() == tok_reloaded().fingerprint());
        RC_ASSERT(tok().encode_raw(s) == tok_reloaded().encode_raw(s));
    }) && ok;

    // ===== physics =====
    ok = rc::check("physics: WeightPathIntegral invariants (incl. NaN/Inf/negative inputs)", [] {
        WeightPathIntegral p(*genFloatNasty());
        const auto steps = *rc::gen::container<std::vector<std::pair<float, float>>>(
            rc::gen::pair(genFloatNasty(), genFloatNasty()));
        float prev_log = p.log_amplitude;
        int i = 0;
        for (const auto& st : steps) {
            p.record_step_norm(st.first, st.second);
            RC_ASSERT(!std::isnan(p.log_amplitude));
            RC_ASSERT(!std::isnan(p.ema_action));
            RC_ASSERT(std::isfinite(p.cumulative_action));
            RC_ASSERT(p.log_amplitude <= prev_log);          // log amplitude never increases
            prev_log = p.log_amplitude;
            const float a = p.amplitude();
            RC_ASSERT(a > 0.0f && a <= 1.0f);
            const float sc = p.lr_scale_ema(0.5f);
            RC_ASSERT(sc >= 0.5f - 1e-6f && sc <= 1.0f + 1e-6f);
            if (i % 7 == 3) p.reset_best();
            const float ra = p.relative_amplitude();
            RC_ASSERT(ra > 0.0f && ra <= 1.0f + 1e-6f);
            ++i;
        }
    }) && ok;

    // ===== pre-tokenized .bin + loader =====
    ok = rc::check("tokenbin: pretokenize_file + TokenBinLoader reproduce encode_raw (shift by 1)", [] {
        const Tokenizer& t = tok();
        const std::string filler = "the quick brown fox jumps over the lazy dog hello world data ";
        const std::string text = filler + *genWordsText();
        const int seq = *rc::gen::inRange<int>(2, 9);
        const std::vector<int> ids = t.encode_raw(text);

        const std::string txt = tc::tmp_file("rc_bin_src.txt");
        const std::string bin = tc::tmp_file("rc_bin.bin");
        tc::write_all(txt, text);
        bool made = false;
        { tc::Silence quiet; made = pretokenize_file(txt, t, bin); }
        RC_ASSERT(made);
        RC_ASSERT(token_bin_matches(bin, t, static_cast<int64_t>(text.size())));
        RC_ASSERT(!token_bin_matches(bin, t, static_cast<int64_t>(text.size()) + 1));

        TokenBinLoader ld(bin, seq, 1);
        std::vector<int> in, tg;
        size_t pos = 0;
        while (ld.next_batch(in, tg)) {
            RC_ASSERT(pos + static_cast<size_t>(seq) + 1 <= ids.size());
            for (int i = 0; i < seq; ++i) {
                RC_ASSERT(in[static_cast<size_t>(i)] == ids[pos + static_cast<size_t>(i)]);
                RC_ASSERT(tg[static_cast<size_t>(i)] == ids[pos + static_cast<size_t>(i) + 1]);
            }
            pos += static_cast<size_t>(seq);
        }
        RC_ASSERT(pos > 0);
    }) && ok;

    std::printf("\n== T25_RapidCheck: %s ==\n", ok ? "all properties passed" : "FAILED");
    return ok ? 0 : 1;
}
