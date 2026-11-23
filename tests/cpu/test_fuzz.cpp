// T22: Seeded property/fuzz tests (deterministic; reproduce with LOGOS_FUZZ_SEED).
//   LOGOS_FUZZ_ITERS  (default 3000)  — iterations of the main tokenizer property loop
//   LOGOS_FUZZ_SEED   (default 1337)  — RNG seed (print on failure for reproduction)
#include "test_common.hpp"
#include "TokenBin.hpp"
#include "PathIntegral.hpp"
#include <limits>

static std::string rand_input(std::mt19937& g, int mode) {
    std::string s;
    const size_t len = g() % 600;
    static const char WS[] = {' ', '\n', '\t', '\r', '\v', '\f', 'a', 'b'};
    static const char* U[] = {"\xE0\xA4\x85", "\xE0\xA4\x95", "\xC3\xA9", "\xE5\xAD\x97", " ", "\n", "ab", "  "};
    switch (mode) {
        case 0:  for (size_t i = 0; i < len; ++i) s.push_back(static_cast<char>(g() & 0xFF)); break;
        case 1:  for (size_t i = 0; i < len; ++i) s.push_back(WS[g() % 8]); break;
        case 2:  for (size_t i = 0; i < len; ++i) s.push_back(" abcdefghij"[g() % 11]); break;
        default: for (size_t i = 0; i < len / 2 + 1; ++i) s += U[g() % 8]; break;
    }
    return s;
}

static void fuzz_tokenizer(Tokenizer& tok, uint32_t seed, long iters) {
    TC_SECTION("fuzz: tokenizer properties");
    std::mt19937 g(seed);
    long bad_rt = 0, bad_range = 0, bad_pieces = 0, bad_split = 0, bad_prefix = 0, bad_enc = 0;
    for (long it = 0; it < iters; ++it) {
        std::string s = rand_input(g, static_cast<int>(it % 4));
        std::vector<int> ids = tok.encode_raw(s);

        if (tok.decode(ids) != s) ++bad_rt;
        for (int id : ids) if (id < 4 || id >= tok.vocab_size) { ++bad_range; break; }

        // pieces: concat == input, each 1..128 bytes; piece-boundary splitting is additive
        std::vector<size_t> offsets{0};
        std::string cat;
        bool pieces_ok = true;
        Tokenizer::for_each_piece(s.data(), s.size(), [&](const char* p, size_t l) {
            if (l == 0 || l > Tokenizer::MAX_PIECE_BYTES) pieces_ok = false;
            cat.append(p, l);
            offsets.push_back(cat.size());
            return true;
        });
        if (!pieces_ok || cat != s) ++bad_pieces;

        size_t cut = offsets[g() % offsets.size()];
        std::vector<int> left  = tok.encode_raw(s.substr(0, cut));
        std::vector<int> right = tok.encode_raw(s.substr(cut));
        left.insert(left.end(), right.begin(), right.end());
        if (left != ids) ++bad_split;

        int64_t m = static_cast<int64_t>(g() % (ids.size() + 3));
        std::vector<int> pre = tok.encode_raw(s, m);
        size_t want = std::min<size_t>(static_cast<size_t>(m), ids.size());
        if (pre.size() != want || !std::equal(pre.begin(), pre.end(), ids.begin())) ++bad_prefix;

        int L = static_cast<int>(g() % 41);
        std::vector<int> e = tok.encode(s, L);
        bool enc_ok = static_cast<int>(e.size()) <= L;
        if (L >= 2) enc_ok = enc_ok && e.size() >= 2 && e.front() == TOKEN_BOS && e.back() == TOKEN_EOS;
        if (!enc_ok) ++bad_enc;
    }
    TC_CHECKF(bad_rt == 0,     "%ld/%ld round-trip failures (seed=%u)", bad_rt, iters, seed);
    TC_CHECKF(bad_range == 0,  "%ld id-range violations (seed=%u)", bad_range, seed);
    TC_CHECKF(bad_pieces == 0, "%ld piece invariants violated (seed=%u)", bad_pieces, seed);
    TC_CHECKF(bad_split == 0,  "%ld piece-boundary additivity violations (seed=%u)", bad_split, seed);
    TC_CHECKF(bad_prefix == 0, "%ld max_tokens prefix violations (seed=%u)", bad_prefix, seed);
    TC_CHECKF(bad_enc == 0,    "%ld encode() BOS/EOS/max_len violations (seed=%u)", bad_enc, seed);
}

static void fuzz_corrupt_vocab(Tokenizer& tok, uint32_t seed, long iters) {
    TC_SECTION("fuzz: corrupted vocab.bin must never crash load()/encode()");
    std::string good_path = tc::tmp_file("fuzz_vocab_good.bin");
    { tc::Silence s; tok.save(good_path); }
    const std::string good = tc::read_all(good_path);
    std::mt19937 g(seed ^ 0xA5A5u);
    std::string path = tc::tmp_file("fuzz_vocab_bad.bin");
    std::string sample = tc::make_corpus(2000, 4);
    long accepted = 0, rejected = 0;
    for (long it = 0; it < iters; ++it) {
        std::string d = good;
        int kind = static_cast<int>(g() % 4);
        if (kind == 0) {                                   // random byte overwrites
            int k = 1 + static_cast<int>(g() % 8);
            for (int j = 0; j < k; ++j) d[g() % d.size()] = static_cast<char>(g() & 0xFF);
        } else if (kind == 1) {                            // bit flips in header region
            for (int j = 0; j < 4; ++j) d[g() % 16] ^= static_cast<char>(1u << (g() % 8));
        } else if (kind == 2) {                            // truncate
            d.resize(g() % d.size());
        } else {                                           // append garbage
            size_t extra = g() % 64;
            for (size_t j = 0; j < extra; ++j) d.push_back(static_cast<char>(g() & 0xFF));
        }
        tc::write_all(path, d);
        tc::Silence s;
        Tokenizer t;
        bool ok = t.load(path);
        if (ok) {
            ++accepted;
            std::vector<int> ids = t.encode_raw(sample);
            (void)t.decode(ids);
            (void)t.fingerprint();
        } else {
            ++rejected;
        }
    }
    TC_CHECKF(accepted + rejected == iters, "all %ld corrupted loads returned (accepted=%ld rejected=%ld)",
              iters, accepted, rejected);
}

static void fuzz_token_bin(Tokenizer& tok, uint32_t seed, long iters) {
    TC_SECTION("fuzz: corrupted token .bin header");
    std::string txt = tc::tmp_file("fuzz_bin_src.txt");
    std::string src = tc::make_corpus(20 * 1024, 8);
    tc::write_all(txt, src);
    std::string bin = tc::tmp_file("fuzz_bin_good.bin");
    bool made;
    { tc::Silence s; made = pretokenize_file(txt, tok, bin); }
    TC_CHECK(made, "baseline bin created");
    const std::string good = tc::read_all(bin);
    TC_CHECK(token_bin_matches(bin, tok, (int64_t)src.size()), "baseline matches");

    std::mt19937 g(seed ^ 0x5A5Au);
    std::string bad_path = tc::tmp_file("fuzz_bin_bad.bin");
    long header_hits_matched = 0, truncs_matched = 0, loader_crash_guard = 0;
    for (long it = 0; it < iters; ++it) {
        std::string d = good;
        bool header_changed = false;
        int kind = static_cast<int>(g() % 3);
        if (kind == 0) {                                   // mutate a header byte
            size_t pos = g() % static_cast<size_t>(TOKBIN_HEADER_BYTES);
            char nb = static_cast<char>(g() & 0xFF);
            header_changed = (nb != d[pos]);
            d[pos] = nb;
        } else if (kind == 1) {                            // truncate (never to full size)
            d.resize(g() % d.size());
        } else {                                           // mutate token payload only
            if (d.size() > static_cast<size_t>(TOKBIN_HEADER_BYTES) + 4)
                d[static_cast<size_t>(TOKBIN_HEADER_BYTES) + g() % (d.size() - TOKBIN_HEADER_BYTES)] ^= 0x55;
        }
        tc::write_all(bad_path, d);
        TokenBinInfo info = read_token_bin_header(bad_path);   // must not crash
        (void)info;
        bool m = token_bin_matches(bad_path, tok, (int64_t)src.size());
        if (kind == 0 && header_changed && m) ++header_hits_matched;
        if (kind == 1 && m) ++truncs_matched;
        try {
            TokenBinLoader ld(bad_path, 8, 1);
            std::vector<int> in, tg;
            for (int j = 0; j < 4; ++j) ld.next_batch(in, tg);
            ++loader_crash_guard;
        } catch (const std::exception&) {
            ++loader_crash_guard;
        }
    }
    TC_CHECKF(header_hits_matched == 0, "%ld header-mutated bins still reported as matching", header_hits_matched);
    TC_CHECKF(truncs_matched == 0, "%ld truncated bins still reported as matching", truncs_matched);
    TC_CHECKF(loader_crash_guard == iters, "TokenBinLoader survived all %ld corrupted files", iters);
}

static void fuzz_path_integral(uint32_t seed, long iters) {
    TC_SECTION("fuzz: WeightPathIntegral invariants");
    std::mt19937 g(seed ^ 0x1234u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float hbars[] = {0.0f, 1e-3f, 1.0f, 100.0f, -2.0f};
    long violations = 0;
    for (int run = 0; run < 50; ++run) {
        WeightPathIntegral p(hbars[g() % 5]);
        float prev_log = p.log_amplitude;
        for (long i = 0; i < iters / 50 + 1; ++i) {
            float loss, norm;
            switch (g() % 8) {
                case 0: loss = nan; norm = U(g); break;
                case 1: loss = U(g); norm = inf; break;
                case 2: loss = -U(g) * 10.0f; norm = U(g); break;
                case 3: loss = U(g); norm = -U(g); break;
                case 4: loss = 0.0f; norm = U(g) * 1e3f; break;
                default: loss = U(g) * 1e2f; norm = U(g) * 1e2f; break;
            }
            p.record_step_norm(loss, norm);
            if (std::isnan(p.log_amplitude) || std::isnan(p.ema_action) || std::isnan(p.cumulative_action)) ++violations;
            if (p.log_amplitude > prev_log) ++violations;                 // never increases
            prev_log = p.log_amplitude;
            float a = p.amplitude();
            if (!(a > 0.0f && a <= 1.0f)) ++violations;
            float fr = U(g);
            float sc = p.lr_scale_ema(fr);
            if (!(sc >= fr - 1e-6f && sc <= 1.0f + 1e-6f)) ++violations;
            if (g() % 50 == 0) p.reset_best();
            float ra = p.relative_amplitude();
            if (!(ra > 0.0f && ra <= 1.0f + 1e-6f)) ++violations;
        }
    }
    TC_CHECKF(violations == 0, "%ld WeightPathIntegral invariant violations (seed=%u)", violations, seed);
}

int main() {
    const long iters   = tc::env_long("LOGOS_FUZZ_ITERS", 3000);
    const uint32_t seed = static_cast<uint32_t>(tc::env_long("LOGOS_FUZZ_SEED", 1337));
    std::printf("LOGOS T22: fuzz tests  iters=%ld seed=%u\n", iters, seed);

    Tokenizer tok;
    { tc::Silence s; tok.build(tc::make_corpus(120 * 1024, seed), 700); }

    fuzz_tokenizer(tok, seed, iters);
    fuzz_corrupt_vocab(tok, seed, std::max(100L, iters / 10));
    fuzz_token_bin(tok, seed, std::max(100L, iters / 10));
    fuzz_path_integral(seed, iters);
    return tc::finish("T22_Fuzz");
}
