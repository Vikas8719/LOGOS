#pragma once

#include "Tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <mutex>

static constexpr int TOKEN_UNK = 0;
static constexpr int TOKEN_BOS = 1;
static constexpr int TOKEN_EOS = 2;
static constexpr int TOKEN_PAD = 3;

static constexpr char     VOCAB_MAGIC[4]  = {'L','G','V','B'};
// v2: byte-level pre-tokenizer (whitespace/newline preserved, no implicit leading-space per word).
// v1 vocab.bin (purana word-split scheme) ab load nahi hota — rebuild karo (LOGOS_REBUILD_VOCAB=1).
static constexpr uint16_t VOCAB_VERSION   = 2;
static constexpr int      MAX_VOCAB_SIZE  = 8192;
static constexpr int      MAX_TOKEN_BYTES = 4096;
static constexpr int      MAX_MERGE_COUNT = MAX_VOCAB_SIZE * 4;

static inline std::string word_key(const std::vector<std::string>& toks) {
    std::string k;
    k.reserve(toks.size() * 4);
    for (int i = 0; i < (int)toks.size(); ++i) {
        if (i) k += '|';
        k += toks[i];
    }
    return k;
}

class Tokenizer {
public:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string>             id_to_token;
    std::vector<std::pair<std::string,std::string>> merges;
    int vocab_size = 0;

    Tokenizer() = default;
    // Cache + mutex copy nahi hote; copy ke baad cache khaali se shuru hota hai.
    Tokenizer(const Tokenizer& o) { copy_from(o); }
    Tokenizer& operator=(const Tokenizer& o) { if (this != &o) copy_from(o); return *this; }

    // ── Pre-tokenizer ────────────────────────────────────────────
    // Text ko pieces me todta hai (byte-level, GPT-2 style):
    //   " word"  = ek space + non-whitespace run
    //   "word"   = line/text ke start ka word (leading space nahi)
    //   "\n", "\n\n", "  " ... = whitespace runs (aakhri single space agle word ke saath jaata hai)
    // Whitespace ASCII hai, UTF-8 multibyte ke andar kabhi nahi aata, isliye Hindi safe hai.
    static constexpr size_t MAX_PIECE_BYTES = 128;

    static inline bool is_ws_byte(unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f';
    }

    // emit(const char*, size_t) -> bool (false = stop)
    template <class F>
    static void for_each_piece(const char* s, size_t n, F&& emit) {
        auto emit_capped = [&](const char* p, size_t len) -> bool {
            while (len > MAX_PIECE_BYTES) {
                size_t k = MAX_PIECE_BYTES;
                while (k > 0 && (static_cast<unsigned char>(p[k]) & 0xC0) == 0x80) --k;  // UTF-8 beech me na kaato
                if (k == 0) k = MAX_PIECE_BYTES;
                if (!emit(p, k)) return false;
                p += k; len -= k;
            }
            return emit(p, len);
        };
        size_t i = 0;
        while (i < n) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (!is_ws_byte(c)) {
                size_t j = i + 1;
                while (j < n && !is_ws_byte(static_cast<unsigned char>(s[j]))) ++j;
                if (!emit_capped(s + i, j - i)) return;
                i = j;
            } else if (c == ' ' && i + 1 < n && !is_ws_byte(static_cast<unsigned char>(s[i + 1]))) {
                size_t j = i + 2;
                while (j < n && !is_ws_byte(static_cast<unsigned char>(s[j]))) ++j;
                if (!emit_capped(s + i, j - i)) return;
                i = j;
            } else {
                size_t j = i + 1;
                while (j < n && is_ws_byte(static_cast<unsigned char>(s[j]))) ++j;
                if (j < n && s[j - 1] == ' ' && (j - i) > 1) --j;   // aakhri space agle word ko
                if (!emit_capped(s + i, j - i)) return;
                i = j;
            }
        }
    }

    // ── Build (byte-level BPE, id-space) ─────────────────────────
    void build(const std::string& text, int target_vocab = 4096) {
        auto t_start = std::chrono::steady_clock::now();
        _reset();
        // load() MAX_VOCAB_SIZE se bada n reject karta hai -> build me hi clamp (save/load round-trip safe)
        if (target_vocab > MAX_VOCAB_SIZE) target_vocab = MAX_VOCAB_SIZE;
        if (target_vocab < 260) target_vocab = 260;   // 4 specials + 256 bytes minimum

        _add("<UNK>");   // 0
        _add("<BOS>");   // 1
        _add("<EOS>");   // 2
        _add("<PAD>");   // 3
        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end()) _add(ch);
        }
        _rebuild_byte_ids();

        std::unordered_map<std::string, int64_t> piece_count;
        piece_count.reserve(1 << 16);
        int64_t pieces_total = 0;
        for_each_piece(text.data(), text.size(), [&](const char* p, size_t l) {
            ++piece_count[std::string(p, l)];
            ++pieces_total;
            return true;
        });
        printf("  BPE init: %d unique piece types from %lld pieces\n",
               (int)piece_count.size(), (long long)pieces_total);
        fflush(stdout);

        if (piece_count.size() > 150000) {
            size_t before = piece_count.size();
            int min_freq = 2;
            for (;; ++min_freq) {
                for (auto it = piece_count.begin(); it != piece_count.end(); ) {
                    if (it->second < min_freq) it = piece_count.erase(it);
                    else ++it;
                }
                if (piece_count.size() <= 120000 || min_freq >= 6) break;
            }
            printf("  BPE prune: %zu -> %zu piece types (freq >= %d)\n",
                   before, piece_count.size(), min_freq);
            fflush(stdout);
        }

        if (piece_count.empty()) {
            std::cout << "⚠️  Empty corpus — no BPE merges done.\n";
            _rebuild_rank_map();
            return;
        }

        struct Word { std::vector<int> ids; int64_t freq; };
        std::vector<Word> words;
        words.reserve(piece_count.size());
        for (auto& kv : piece_count) {
            Word w;
            w.freq = kv.second;
            w.ids.reserve(kv.first.size());
            for (unsigned char c : kv.first) w.ids.push_back(byte_id_[c]);
            words.push_back(std::move(w));
        }
        piece_count.clear();

        std::unordered_map<uint64_t, int64_t> pair_freq;
        int merge_idx = 0;
        const int merges_target = target_vocab - vocab_size;

        while (vocab_size < target_vocab && (int)merges.size() < MAX_MERGE_COUNT) {
            pair_freq.clear();
            pair_freq.reserve(1 << 15);
            for (const auto& w : words) {
                if (w.ids.size() < 2) continue;
                for (size_t i = 0; i + 1 < w.ids.size(); ++i)
                    pair_freq[pair_key(w.ids[i], w.ids[i + 1])] += w.freq;
            }
            if (pair_freq.empty()) break;

            uint64_t best_key = 0; int64_t best_cnt = -1;
            for (const auto& kv : pair_freq) {
                if (kv.second > best_cnt || (kv.second == best_cnt && kv.first < best_key)) {
                    best_cnt = kv.second; best_key = kv.first;   // deterministic tie-break
                }
            }
            if (best_cnt < 2) break;

            const int a = (int)(best_key >> 32);
            const int b = (int)(best_key & 0xFFFFFFFFull);
            const std::string left  = id_to_token[a];
            const std::string right = id_to_token[b];
            const std::string merged = left + right;

            int new_id;
            auto vit = vocab.find(merged);
            if (vit == vocab.end()) { new_id = vocab_size; _add(merged); }
            else                    { new_id = vit->second; }   // same string pehle se hai (alag path se)
            merges.push_back({left, right});

            for (auto& w : words) {
                auto& ids = w.ids;
                if (ids.size() < 2) continue;
                size_t wr = 0;
                for (size_t i = 0; i < ids.size(); ) {
                    if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) { ids[wr++] = new_id; i += 2; }
                    else ids[wr++] = ids[i++];
                }
                ids.resize(wr);
            }

            ++merge_idx;
            if (merge_idx % 500 == 0) {
                float secs = std::chrono::duration<float>(
                    std::chrono::steady_clock::now() - t_start).count();
                printf("  BPE merge %d/%d | vocab=%d | %.1fs elapsed\n",
                       merge_idx, merges_target, vocab_size, secs);
                fflush(stdout);
            }
        }

        _rebuild_rank_map();

        float total_secs = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t_start).count();
        printf("✅ Tokenizer built: vocab_size=%d  merges=%d  time=%.1fs\n",
               vocab_size, (int)merges.size(), total_secs);
        fflush(stdout);
    }

    // ── Encode ───────────────────────────────────────────────────
    // Raw: koi BOS/EOS nahi. Training data / chunks ke liye YEHI use karo
    // (chunk ke beech me BOS/EOS lagne se stream corrupt hoti thi).
    std::vector<int> encode_raw(const std::string& text, int64_t max_tokens = INT64_MAX) const {
        std::vector<int> ids;
        if (max_tokens <= 0 || text.empty()) return ids;
        ids.reserve(std::min<size_t>(text.size() / 3 + 16, static_cast<size_t>(max_tokens)));
        for_each_piece(text.data(), text.size(), [&](const char* p, size_t l) {
            _bpe_piece(p, l, ids);
            return (int64_t)ids.size() < max_tokens;
        });
        if ((int64_t)ids.size() > max_tokens) ids.resize(static_cast<size_t>(max_tokens));
        return ids;
    }

    // Legacy: BOS + text + EOS (max_len tak). Sirf single prompt/sentence ke liye.
    std::vector<int> encode(const std::string& text, int max_len = 512) const {
        std::vector<int> ids;
        ids.push_back(TOKEN_BOS);
        const int64_t room = (int64_t)max_len - 2;
        if (room > 0) {
            auto body = encode_raw(text, room);
            ids.insert(ids.end(), body.begin(), body.end());
        }
        ids.push_back(TOKEN_EOS);
        if ((int)ids.size() > max_len) ids.resize(static_cast<size_t>(std::max(max_len, 0)));
        return ids;
    }

    // Exact round-trip: decode(encode_raw(x)) == x
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        result.reserve(ids.size() * 3);
        for (int id : ids) {
            if (id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD) continue;
            if (id >= 0 && id < (int)id_to_token.size())
                result += id_to_token[id];
            else
                result += "<UNK>";
        }
        return result;
    }

    // Vocab + merges ka fingerprint (pre-tokenized .bin cache validity ke liye)
    uint64_t fingerprint() const {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const void* d, size_t n) {
            const unsigned char* p = static_cast<const unsigned char*>(d);
            for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        };
        uint32_t vs = (uint32_t)vocab_size, ms = (uint32_t)merges.size();
        mix(&vs, 4); mix(&ms, 4);
        for (const auto& t : id_to_token) {
            uint32_t l = (uint32_t)t.size();
            mix(&l, 4); mix(t.data(), t.size());
        }
        return h;
    }

    bool save(const std::string& path = "vocab.bin") const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ save: cannot open " << path << "\n"; return false; }

        f.write(VOCAB_MAGIC, 4);
        f.write(reinterpret_cast<const char*>(&VOCAB_VERSION), sizeof(uint16_t));

        int32_t n = (int32_t)id_to_token.size();
        int32_t m = (int32_t)merges.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(&m), sizeof(int32_t));

        for (const auto& tok : id_to_token) {
            int32_t len = (int32_t)tok.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
            f.write(tok.data(), len);
        }

        auto write_str = [&](const std::string& s) {
            int32_t l = (int32_t)s.size();
            f.write(reinterpret_cast<const char*>(&l), sizeof(int32_t));
            f.write(s.data(), l);
        };
        for (const auto& [a, b] : merges) {
            write_str(a);
            write_str(b);
        }

        if (!f) { std::cerr << "❌ save: write error: " << path << "\n"; return false; }
        std::cout << "✅ Vocab saved: " << path
                  << "  tokens=" << n << "  merges=" << m
                  << "  format=LGVBv" << VOCAB_VERSION << "\n";
        return true;
    }

    bool load(const std::string& path = "vocab.bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ load: cannot open " << path << "\n"; return false; }

        char magic[4] = {};
        f.read(magic, 4);
        if (!f || std::memcmp(magic, VOCAB_MAGIC, 4) != 0) {
            std::cerr << "❌ load: bad magic in " << path
                      << "  got='" << magic[0] << magic[1] << magic[2] << magic[3] << "'"
                      << "  expected='LGVB'\n"
                      << "   → Purana vocab.bin hai. Retrain karo.\n";
            return false;
        }

        uint16_t ver = 0;
        f.read(reinterpret_cast<char*>(&ver), sizeof(uint16_t));
        if (!f || ver != VOCAB_VERSION) {
            std::cerr << "❌ load: version mismatch  file=v" << ver
                      << "  expected=v" << VOCAB_VERSION << "\n"
                      << "   → Purana tokenizer format. vocab.bin rebuild karo (LOGOS_REBUILD_VOCAB=1).\n";
            return false;
        }

        int32_t n = 0, m = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&m), sizeof(int32_t));
        if (!f) { std::cerr << "❌ load: truncated header\n"; return false; }

        if (n <= 0 || n > MAX_VOCAB_SIZE) {
            std::cerr << "❌ load: n=" << n << " out of range\n"; return false;
        }
        if (m < 0 || m > MAX_MERGE_COUNT) {
            std::cerr << "❌ load: m=" << m << " out of range\n"; return false;
        }

        id_to_token.clear(); id_to_token.reserve(n);
        vocab.clear();       vocab.reserve(n);

        for (int32_t i = 0; i < n; ++i) {
            int32_t len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
            if (!f || len < 0 || len > MAX_TOKEN_BYTES) {
                std::cerr << "❌ load: bad token len=" << len << " at idx=" << i << "\n";
                return false;
            }
            std::string tok(static_cast<size_t>(len), '\0');
            f.read(tok.data(), len);
            if (!f) { std::cerr << "❌ load: EOF at token idx=" << i << "\n"; return false; }
            vocab[tok] = i;
            id_to_token.push_back(std::move(tok));
        }

        merges.clear(); merges.reserve(m);
        auto read_str = [&](std::string& s) -> bool {
            int32_t l = 0;
            f.read(reinterpret_cast<char*>(&l), sizeof(int32_t));
            if (!f || l < 0 || l > MAX_TOKEN_BYTES) {
                std::cerr << "❌ load: bad merge str len=" << l << "\n";
                return false;
            }
            s.resize(static_cast<size_t>(l));
            f.read(s.data(), l);
            return !!f;
        };

        for (int32_t i = 0; i < m; ++i) {
            std::string a, b;
            if (!read_str(a) || !read_str(b)) {
                std::cerr << "❌ load: EOF in merge table at idx=" << i << "\n";
                return false;
            }
            merges.push_back({std::move(a), std::move(b)});
        }

        vocab_size = n;
        _rebuild_rank_map();
        std::cout << "✅ Vocab loaded: " << path
                  << "  tokens=" << n << "  merges=" << m
                  << "  format=LGVBv" << ver << "\n";
        return true;
    }

private:
    static constexpr size_t CACHE_MAX_ENTRIES = 2000000;

    // (left_id, right_id) -> (rank, merged_id)
    std::unordered_map<uint64_t, std::pair<uint32_t, int>> merge_rank_;
    int byte_id_[256] = {};

    // Persistent piece cache: encode calls ke beech bhi rehta hai (thread-safe).
    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;

    static inline uint64_t pair_key(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

    void copy_from(const Tokenizer& o) {
        vocab       = o.vocab;
        id_to_token = o.id_to_token;
        merges      = o.merges;
        vocab_size  = o.vocab_size;
        merge_rank_ = o.merge_rank_;
        std::memcpy(byte_id_, o.byte_id_, sizeof(byte_id_));
        std::lock_guard<std::mutex> lk(cache_mu_);
        cache_.clear();
    }

    void _reset() {
        vocab.clear(); id_to_token.clear(); merges.clear();
        vocab_size = 0;
        merge_rank_.clear();
        std::lock_guard<std::mutex> lk(cache_mu_);
        cache_.clear();
    }

    void _add(const std::string& tok) {
        vocab[tok] = vocab_size;
        id_to_token.push_back(tok);
        ++vocab_size;
    }

    void _rebuild_byte_ids() {
        for (int c = 0; c < 256; ++c) {
            auto it = vocab.find(std::string(1, static_cast<char>(c)));
            byte_id_[c] = (it != vocab.end()) ? it->second : TOKEN_UNK;
        }
    }

    void _rebuild_rank_map() {
        _rebuild_byte_ids();
        merge_rank_.clear();
        merge_rank_.reserve(merges.size() * 2 + 16);
        for (size_t r = 0; r < merges.size(); ++r) {
            auto la = vocab.find(merges[r].first);
            auto lb = vocab.find(merges[r].second);
            auto lm = vocab.find(merges[r].first + merges[r].second);
            if (la == vocab.end() || lb == vocab.end() || lm == vocab.end()) continue;
            merge_rank_.emplace(pair_key(la->second, lb->second),
                                std::make_pair(static_cast<uint32_t>(r), lm->second));  // emplace: sabse chhota rank rehta hai
        }
        std::lock_guard<std::mutex> lk(cache_mu_);
        cache_.clear();
    }

    // Ek piece ko BPE se encode karke out me append karta hai (rank-order merges + cache).
    void _bpe_piece(const char* p, size_t len, std::vector<int>& out) const {
        std::string key(p, len);
        {
            std::lock_guard<std::mutex> lk(cache_mu_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                out.insert(out.end(), it->second.begin(), it->second.end());
                return;
            }
        }

        std::vector<int> seq;
        seq.reserve(len);
        for (size_t k = 0; k < len; ++k) seq.push_back(byte_id_[static_cast<unsigned char>(p[k])]);

        while (seq.size() >= 2) {
            uint32_t best_rank = UINT32_MAX;
            uint64_t best_key = 0;
            int best_new = 0;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                uint64_t k2 = pair_key(seq[i], seq[i + 1]);
                auto it = merge_rank_.find(k2);
                if (it != merge_rank_.end() && it->second.first < best_rank) {
                    best_rank = it->second.first; best_key = k2; best_new = it->second.second;
                }
            }
            if (best_rank == UINT32_MAX) break;
            const int a = static_cast<int>(best_key >> 32);
            const int b = static_cast<int>(best_key & 0xFFFFFFFFull);
            size_t wr = 0;
            for (size_t i = 0; i < seq.size(); ) {
                if (i + 1 < seq.size() && seq[i] == a && seq[i + 1] == b) { seq[wr++] = best_new; i += 2; }
                else seq[wr++] = seq[i++];
            }
            seq.resize(wr);
        }

        out.insert(out.end(), seq.begin(), seq.end());
        {
            std::lock_guard<std::mutex> lk(cache_mu_);
            if (cache_.size() >= CACHE_MAX_ENTRIES) cache_.clear();
            cache_.emplace(std::move(key), std::move(seq));
        }
    }
};
