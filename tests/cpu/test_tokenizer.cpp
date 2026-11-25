// T20: Tokenizer unit tests (CPU only, no CUDA).
#include "test_common.hpp"
#include "Tokenizer.hpp"
#include <atomic>
#include <thread>

static std::vector<std::string> pieces_of(const std::string& s) {
    std::vector<std::string> v;
    Tokenizer::for_each_piece(s.data(), s.size(), [&](const char* p, size_t l) {
        v.emplace_back(p, l);
        return true;
    });
    return v;
}

static std::string join(const std::vector<std::string>& v) {
    std::string r;
    for (const auto& x : v) r += x;
    return r;
}

static Tokenizer& shared_tok() {
    static Tokenizer t;
    static bool built = false;
    if (!built) {
        t.build(tc::make_corpus(200 * 1024, 7), 600);
        built = true;
    }
    return t;
}

static void test_pieces_golden() {
    TC_SECTION("pre-tokenizer golden pieces");
    using V = std::vector<std::string>;
    const std::vector<std::pair<std::string, V>> cases = {
        {"",            V{}},
        {"hello world", V{"hello", " world"}},
        {"a\n\nb",      V{"a", "\n\n", "b"}},
        {"a  b",        V{"a", " ", " b"}},
        {" a",          V{" a"}},
        {"a ",          V{"a", " "}},
        {"\n a",        V{"\n", " a"}},
        {"a\tb",        V{"a", "\t", "b"}},
        {"a\r\nb",      V{"a", "\r\n", "b"}},
        {"   ",         V{"   "}},
    };
    for (const auto& c : cases) {
        auto got = pieces_of(c.first);
        TC_CHECKF(got == c.second, "pieces mismatch for input of %zu bytes (got %zu pieces, want %zu)",
                  c.first.size(), got.size(), c.second.size());
    }
}

static void test_piece_cap() {
    TC_SECTION("piece cap (128 bytes) + UTF-8 safety");
    for (size_t len : {1u, 127u, 128u, 129u, 255u, 256u, 257u, 1000u}) {
        std::string w(len, 'x');
        auto ps = pieces_of(w);
        TC_CHECKF(join(ps) == w, "concat mismatch len=%zu", len);
        bool ok = true;
        for (auto& p : ps) if (p.empty() || p.size() > Tokenizer::MAX_PIECE_BYTES) ok = false;
        TC_CHECKF(ok, "empty or oversize piece for len=%zu", len);
        size_t want = (len + 127) / 128;
        TC_CHECKF(ps.size() == want, "len=%zu pieces=%zu want=%zu", len, ps.size(), want);
    }
    {   // leading space + 128 x  => 129 bytes => 2 pieces
        std::string w = " " + std::string(128, 'x');
        auto ps = pieces_of(w);
        TC_CHECK(ps.size() == 2 && ps[0].size() == 128 && ps[1].size() == 1, "space+128x splits 128+1");
    }
    std::string hi;
    for (int i = 0; i < 100; ++i) hi += "\xE0\xA4\x85";   // 'अ' x100 = 300 bytes
    auto ps = pieces_of(hi);
    bool utf8_ok = (join(ps) == hi);
    for (auto& p : ps) {
        if (p.empty() || p.size() > 128) utf8_ok = false;
        if ((static_cast<unsigned char>(p[0]) & 0xC0) == 0x80) utf8_ok = false;   // continuation byte se start
    }
    TC_CHECK(utf8_ok, "Hindi long word: pieces never split a UTF-8 sequence");
}

static void test_golden_bpe() {
    TC_SECTION("BPE golden merges + deterministic tie-break");
    {
        Tokenizer t;
        t.build("ab ab ab ab", 300);
        TC_CHECKF(t.merges.size() == 2, "merges=%zu want 2", t.merges.size());
        TC_CHECKF(t.vocab_size == 262, "vocab_size=%d want 262", t.vocab_size);
        if (t.merges.size() == 2) {
            TC_CHECK(t.merges[0].first == "a" && t.merges[0].second == "b", "merge0 = (a,b)");
            TC_CHECK(t.merges[1].first == " " && t.merges[1].second == "ab", "merge1 = ( ,ab)");
        }
        auto ids = t.encode_raw("ab ab");
        TC_CHECK(ids.size() == 2 && ids[0] == 260 && ids[1] == 261, "encode_raw('ab ab') == [260,261]");
        TC_CHECK(t.decode(ids) == "ab ab", "decode golden");
    }
    {
        // 3 pairs tie at freq 2: smallest (id_a,id_b) key must win => (' ','c')
        Tokenizer t;
        t.build("ab cd ab cd", 300);
        TC_CHECK(!t.merges.empty(), "tie corpus produced merges");
        if (!t.merges.empty())
            TC_CHECK(t.merges[0].first == " " && t.merges[0].second == "c", "tie-break picks smallest key (' ','c')");
    }
}

static void test_roundtrip_cases() {
    TC_SECTION("exact round-trip decode(encode_raw(x)) == x");
    Tokenizer& tok = shared_tok();
    std::vector<std::string> cases = {
        "", " ", "  ", "\n", "\n\n\n", "a", " a", "  a", "a ", "hello world", "\n\n  x", "\t\tindent",
        "\r\nwindows\r\n", "नमस्ते दुनिया", "mixed हिन्दी and English\n", "tab\tseparated\tvalues",
        "trailing spaces   ", "   leading spaces", "a\vb\fc",
    };
    std::string all;
    for (int c = 0; c < 256; ++c) all.push_back(static_cast<char>(c));
    cases.push_back(all);
    cases.push_back(std::string(1000, 'z'));
    cases.push_back(tc::make_corpus(5000, 99));
    for (const auto& s : cases) {
        auto ids = tok.encode_raw(s);
        TC_CHECKF(tok.decode(ids) == s, "round-trip failed for input of %zu bytes", s.size());
    }
}

static void test_specials_and_encode() {
    TC_SECTION("encode() BOS/EOS + max_len, decode() specials");
    Tokenizer& tok = shared_tok();

    auto raw = tok.encode_raw("hello world");
    bool no_special = true;
    for (int id : raw) if (id < 4) no_special = false;
    TC_CHECK(no_special, "encode_raw never emits BOS/EOS/UNK/PAD for normal text");

    auto e = tok.encode("hello", 512);
    TC_CHECK(e.size() >= 3 && e.front() == TOKEN_BOS && e.back() == TOKEN_EOS, "encode: BOS ... EOS");

    std::string longtext = tc::make_corpus(2000, 5);
    auto e10 = tok.encode(longtext, 10);
    TC_CHECKF(e10.size() == 10, "encode(max_len=10) size=%zu", e10.size());
    TC_CHECK(!e10.empty() && e10.front() == TOKEN_BOS && e10.back() == TOKEN_EOS, "truncated encode keeps BOS and EOS");

    auto e3 = tok.encode(longtext, 3);
    TC_CHECK(e3.size() == 3 && e3[0] == TOKEN_BOS && e3[2] == TOKEN_EOS, "max_len=3 -> [BOS,tok,EOS]");
    auto e2 = tok.encode(longtext, 2);
    TC_CHECK(e2.size() == 2 && e2[0] == TOKEN_BOS && e2[1] == TOKEN_EOS, "max_len=2 -> [BOS,EOS]");
    auto e1 = tok.encode(longtext, 1);
    TC_CHECK(e1.size() == 1 && e1[0] == TOKEN_BOS, "max_len=1 -> [BOS]");
    TC_CHECK(tok.encode(longtext, 0).empty(), "max_len=0 -> empty");
    TC_CHECK(tok.encode(longtext, -5).empty(), "max_len<0 -> empty");
    auto ee = tok.encode("", 8);
    TC_CHECK(ee.size() == 2, "empty text -> [BOS,EOS]");

    TC_CHECK(tok.decode({TOKEN_BOS, TOKEN_PAD, TOKEN_EOS}).empty(), "decode skips BOS/PAD/EOS");
    TC_CHECK(tok.decode({tok.vocab_size, -1, 999999}) == "<UNK><UNK><UNK>", "decode: out-of-range id -> <UNK>");
}

static void test_truncation_prefix() {
    TC_SECTION("encode_raw(max_tokens) is a prefix of the full encoding");
    Tokenizer& tok = shared_tok();
    std::string text = tc::make_corpus(3000, 11);
    auto full = tok.encode_raw(text);
    TC_CHECK(full.size() > 50, "full encoding long enough");
    for (int64_t m : {1, 2, 5, 17, 100}) {
        auto part = tok.encode_raw(text, m);
        bool ok = part.size() == std::min<size_t>(static_cast<size_t>(m), full.size());
        for (size_t i = 0; ok && i < part.size(); ++i) ok = (part[i] == full[i]);
        TC_CHECKF(ok, "prefix property max_tokens=%lld", (long long)m);
    }
    TC_CHECK(tok.encode_raw(text, 0).empty(), "max_tokens=0 -> empty");
    TC_CHECK(tok.encode_raw(text, -3).empty(), "max_tokens<0 -> empty");
}

static void test_determinism_and_compression() {
    TC_SECTION("determinism + compression");
    std::string corpus = tc::make_corpus(150 * 1024, 21);
    Tokenizer a, b;
    a.build(corpus, 500);
    b.build(corpus, 500);
    TC_CHECK(a.fingerprint() == b.fingerprint(), "two builds -> same fingerprint");
    TC_CHECK(a.merges == b.merges, "two builds -> same merges");
    TC_CHECK(a.vocab_size >= 260 && a.vocab_size <= 500, "vocab_size within [260,target]");
    auto ids = a.encode_raw(corpus);
    TC_CHECKF(ids.size() * 10 < corpus.size() * 6, "BPE compresses: %zu tokens for %zu bytes", ids.size(), corpus.size());
    TC_CHECK(ids == a.encode_raw(corpus), "encode twice (cold then cached) identical");
    Tokenizer c;
    c.build(corpus + " extra words appended", 500);
    TC_CHECK(c.fingerprint() != a.fingerprint(), "different corpus -> different fingerprint");
}

static void test_empty_and_clamp() {
    TC_SECTION("empty corpus + target_vocab clamp");
    Tokenizer e;
    { tc::Silence s; e.build("", 1000); }
    TC_CHECKF(e.vocab_size == 260, "empty corpus vocab=%d want 260", e.vocab_size);
    TC_CHECK(e.merges.empty(), "empty corpus -> no merges");
    TC_CHECK(e.decode(e.encode_raw("hello")) == "hello", "empty-corpus tokenizer still round-trips bytes");

    Tokenizer big;
    big.build(tc::make_corpus(100 * 1024, 3), 1000000);
    TC_CHECKF(big.vocab_size <= MAX_VOCAB_SIZE, "vocab_size=%d <= MAX_VOCAB_SIZE", big.vocab_size);
    std::string p = tc::tmp_file("clamp_vocab.bin");
    TC_CHECK(big.save(p), "save ok");
    Tokenizer back;
    TC_CHECK(back.load(p), "load of huge-target build ok (save/load symmetric)");
    TC_CHECK(back.fingerprint() == big.fingerprint(), "fingerprint survives save/load");
}

static void test_persistence() {
    TC_SECTION("save/load round trip + corrupt file rejection");
    Tokenizer& tok = shared_tok();
    std::string p = tc::tmp_file("persist_vocab.bin");
    TC_CHECK(tok.save(p), "save ok");

    Tokenizer t2;
    TC_CHECK(t2.load(p), "load ok");
    TC_CHECK(t2.fingerprint() == tok.fingerprint(), "fingerprint equal after load");
    TC_CHECK(t2.vocab_size == tok.vocab_size && t2.merges == tok.merges, "vocab/merges equal after load");
    std::string sample = tc::make_corpus(4000, 77);
    TC_CHECK(t2.encode_raw(sample) == tok.encode_raw(sample), "loaded tokenizer encodes identically");

    std::string bytes = tc::read_all(p);
    TC_CHECK(bytes.size() > 100, "vocab file non-trivial");
    std::string q = tc::tmp_file("persist_bad.bin");
    // load() cout/cerr me bahut print karta hai; checks Silence ke BAHAR rakhe hain (fail msg dikhe)
    auto quiet_load = [](const std::string& path) {
        tc::Silence s;
        Tokenizer t;
        return t.load(path);
    };
    {
        TC_CHECK(!quiet_load(tc::tmp_file("does_not_exist.bin")), "missing file -> false");

        std::string bad = bytes; bad[0] = 'X';
        tc::write_all(q, bad);
        TC_CHECK(!quiet_load(q), "bad magic -> false");

        bad = bytes; bad[4] = static_cast<char>(bad[4] + 1);   // version field
        tc::write_all(q, bad);
        TC_CHECK(!quiet_load(q), "wrong version -> false");

        bad = bytes; bad[6] = bad[7] = bad[8] = bad[9] = 0;     // n = 0
        tc::write_all(q, bad);
        TC_CHECK(!quiet_load(q), "n=0 -> false");

        bad = bytes; bad[6] = bad[7] = bad[8] = '\x7f';          // n huge
        tc::write_all(q, bad);
        TC_CHECK(!quiet_load(q), "n huge -> false");

        for (size_t len : {size_t(0), size_t(1), size_t(3), size_t(4), size_t(5), size_t(6), size_t(9),
                           size_t(13), size_t(17), bytes.size() / 2, bytes.size() - 1}) {
            tc::write_all(q, bytes.substr(0, len));
            bool ok = quiet_load(q);
            TC_CHECKF(!ok, "truncated to %zu bytes -> load must fail", len);
        }
    }
}

static void test_copy_semantics() {
    TC_SECTION("copy ctor / assignment");
    Tokenizer& a = shared_tok();
    std::string sample = tc::make_corpus(3000, 13);
    auto ref = a.encode_raw(sample);
    Tokenizer b(a);
    TC_CHECK(b.encode_raw(sample) == ref, "copy-constructed encodes identically");
    Tokenizer c;
    c = a;
    TC_CHECK(c.encode_raw(sample) == ref, "copy-assigned encodes identically");
    TC_CHECK(b.fingerprint() == a.fingerprint(), "copy fingerprint equal");
    TC_CHECK(a.encode_raw(sample) == ref, "original unchanged after copies");
}

static void test_thread_safety() {
    TC_SECTION("concurrent encode (shared piece cache)");
    Tokenizer tok(shared_tok());   // fresh cache
    std::vector<std::string> texts;
    for (int i = 0; i < 4; ++i) texts.push_back(tc::make_corpus(60 * 1024, 100 + i));
    std::vector<std::vector<int>> expect;
    {
        Tokenizer solo(shared_tok());
        for (auto& t : texts) expect.push_back(solo.encode_raw(t));
    }
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int w = 0; w < 4; ++w) {
        th.emplace_back([&, w]() {
            for (int rep = 0; rep < 3; ++rep)
                for (int i = 0; i < 4; ++i) {
                    int idx = (i + w) % 4;
                    if (tok.encode_raw(texts[idx]) != expect[idx]) ++bad;
                }
        });
    }
    for (auto& t : th) t.join();
    TC_CHECKF(bad == 0, "%d concurrent encodes mismatched single-thread result", bad.load());
}

// --- Cache boundary mutation tests ---
// Ye tests CACHE_MAX_ENTRIES >= mutation (>= → > ya >=→==) aur
// cache_.clear() void-call mutation (clear ko noop banana) ko pakdte hain.
static void test_cache_boundary_mutations() {
    TC_SECTION("cache: CACHE_MAX_ENTRIES boundary (kills >= -> > mutant)");
    Tokenizer& tok = shared_tok();

    // Step 1: cache fill karlein aur verify karein ke encoding correct rehti hai.
    // Agar >= ki jagah > hai, to cache_ kabhi clear nahi hogi aur memory
    // indefinitely badhti rahegi (memory-correctness problem).
    // Hum 2 000 000 unique pieces banate hain (ya usse zyada) aur phir
    // verify karte hain ke naye pieces abhi bhi sahi encode hote hain.
    // Note: hum Tokenizer ki ek nayi copy use karte hain taaki shared cache
    // ke saath interfere na ho.
    {
        Tokenizer t;
        {
            tc::Silence s;
            t.build(tc::make_corpus(80 * 1024, 42), 300);
        }

        // 1a: Cache eviction ke baad bhi encoding correct honi chahiye.
        //     Hum ek reference encoding lete hain, phir bahut saari strings
        //     encode karte hain (cache fill/evict), aur phir wahi string
        //     dobara encode karte hain — result same hona chahiye.
        const std::string ref_text = "cache eviction test reference string hello world";
        const auto ref_ids = t.encode_raw(ref_text);
        TC_CHECK(!ref_ids.empty(), "reference encoding non-empty");

        // Bahut saari distinct strings encode karein (cache overflow force karo)
        for (int i = 0; i < 50; ++i) {
            (void)t.encode_raw(tc::make_corpus(500, 1000 + i));
        }

        // Ab wahi ref_text dobara encode karo — same result aana chahiye
        // (clear ke baad cache rebuild hoti hai, correctness preserve honi chahiye)
        const auto ref_ids2 = t.encode_raw(ref_text);
        TC_CHECK(ref_ids == ref_ids2,
                 "post-eviction encode must match pre-eviction (kills cache_.clear() noop mutant)");
    }

    // Step 2: cache_.clear() void-call mutation test.
    // Agar cache_.clear() ko noop (void-call) banaya jaye, to cache hamesha
    // badhti rahegi. Hum verify karte hain ke after eviction, encoding result
    // unchanged hai — jo sirf tab true hai jab clear() sahi kaam kare.
    {
        Tokenizer t;
        {
            tc::Silence s;
            t.build(tc::make_corpus(80 * 1024, 55), 300);
        }

        std::vector<std::string> samples;
        for (int i = 0; i < 20; ++i)
            samples.push_back(tc::make_corpus(300, 2000 + i));

        // Pehle pass: encoding lete hain (cache cold)
        std::vector<std::vector<int>> first_pass;
        for (const auto& s : samples)
            first_pass.push_back(t.encode_raw(s));

        // Cache overflow karo: bahut saari alag-alag strings encode karo
        for (int i = 0; i < 100; ++i)
            (void)t.encode_raw(tc::make_corpus(200, 5000 + i));

        // Dusra pass: same samples phir se encode karo (cache evicted + rebuilt)
        bool evict_ok = true;
        for (int i = 0; i < (int)samples.size(); ++i) {
            if (t.encode_raw(samples[i]) != first_pass[i]) { evict_ok = false; break; }
        }
        TC_CHECK(evict_ok,
                 "encoding after cache evict matches first-pass (kills cache_.clear() void-call mutant)");
    }

    // Step 3: CACHE_MAX_ENTRIES exact boundary check.
    // Agar >= ki jagah > mutation hoti hai, to exactly CACHE_MAX_ENTRIES par
    // clear nahi hogi. Hum directly verify karte hain ke cache semantics
    // (evict-then-reinsert) sahi kaam karte hain.
    {
        Tokenizer t;
        {
            tc::Silence s;
            t.build(tc::make_corpus(50 * 1024, 77), 300);
        }
        // Ek string jo hum boundary ke aas-paas bahut baar encode karenge
        const std::string boundary_str = "boundary mutation kill test";
        const auto boundary_ids = t.encode_raw(boundary_str);
        TC_CHECK(!boundary_ids.empty(), "boundary string encodes to something");

        // Encode karo jab tak cache full na ho jaye, phir verify karo
        for (int batch = 0; batch < 5; ++batch) {
            for (int i = 0; i < 40; ++i)
                (void)t.encode_raw(tc::make_corpus(150, 10000 + batch * 40 + i));
            // Har batch ke baad boundary string same rehni chahiye
            TC_CHECK(t.encode_raw(boundary_str) == boundary_ids,
                     "boundary string consistent across cache fill cycles (kills >= -> > mutant)");
        }
    }
}

int main() {
    std::printf("LOGOS T20: Tokenizer unit tests\n");
    test_pieces_golden();
    test_piece_cap();
    test_golden_bpe();
    test_roundtrip_cases();
    test_specials_and_encode();
    test_truncation_prefix();
    test_determinism_and_compression();
    test_empty_and_clamp();
    test_persistence();
    test_copy_semantics();
    test_thread_safety();
    test_cache_boundary_mutations();
    return tc::finish("T20_Tokenizer_Unit");
}
