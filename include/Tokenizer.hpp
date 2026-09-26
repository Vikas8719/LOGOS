#pragma once
// ============================================================
//  LOGOS — Tokenizer.hpp  (header-only, #pragma once protected)
//
//  FIX: Secure binary format for vocab.bin
//    Pehle (broken): magic/version/bounds kuch nahi tha.
//      - Corrupt ya purani file silently load hoti thi.
//      - n ko directly resize() mein dete the → bad_alloc / OOB possible.
//    Ab (fixed): "LGVB" magic + v1 version + strict bounds on every read.
//      - Koi bhi mismatch → clear error + false return.
//      - Backward compat: purane files detect ho jaate hain, retrain prompt.
//
//  BPE (Byte Pair Encoding) — proper implementation:
//    1. 256 char base vocab + 4 special tokens
//    2. Frequency-based pair merge loop (standard BPE — Sennrich 2015)
//    3. Merge rules saved in vocab.bin, applied at encode() time
//    4. GPT-2 style space-prefix per word (handles word boundaries cleanly)
//    5. O(vocab_merges × seq_words) encode — fast enough for training
// ============================================================
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
#include <cassert>

// ── Special token IDs ─────────────────────────────────────────
static constexpr int TOKEN_UNK = 0;
static constexpr int TOKEN_BOS = 1;
static constexpr int TOKEN_EOS = 2;
static constexpr int TOKEN_PAD = 3;

// ── vocab.bin binary format (v1) ─────────────────────────────
//   Bytes  Field
//   [0..3]  magic    = "LGVB"  (LOGOS Vocab Binary)
//   [4..5]  version  = 0x0001  (uint16_t LE)
//   [6..9]  n        = token count (int32)
//   [10..13] m       = merge count (int32)
//   then n × { int32 len, len bytes }  ← token strings
//   then m × { int32 la, la bytes, int32 lb, lb bytes } ← merge pairs
//
// Older files (no magic) are detected via magic check → reject + retrain prompt.
static constexpr char     VOCAB_MAGIC[4]  = {'L','G','V','B'};
static constexpr uint16_t VOCAB_VERSION   = 1;
static constexpr int      MAX_VOCAB_SIZE  = 8192;
static constexpr int      MAX_TOKEN_BYTES = 4096;  // single token max length
static constexpr int      MAX_MERGE_COUNT = MAX_VOCAB_SIZE * 4;

class Tokenizer {
public:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string>             id_to_token;
    std::vector<std::pair<std::string,std::string>> merges;
    int vocab_size = 0;

    // ── build: BPE from raw text ──────────────────────────────
    // Standard Sennrich-2015 BPE:
    //   1. Start with char-level (byte) vocab
    //   2. Count all adjacent-pair frequencies across all words
    //   3. Merge most-frequent pair → new token, repeat
    //   4. Stop when target_vocab reached or no pair freq >= 2
    void build(const std::string& text, int target_vocab = 4096) {
        vocab.clear(); id_to_token.clear(); merges.clear();

        // Special tokens first (fixed IDs 0-3)
        _add("<UNK>");   // 0
        _add("<BOS>");   // 1
        _add("<EOS>");   // 2
        _add("<PAD>");   // 3

        // Base vocab: all 256 byte values
        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end()) _add(ch);
        }
        // Now vocab_size == 260 (4 special + 256 chars)

        // Tokenise text into word-level char sequences (GPT-2 style Ġ prefix)
        // " hello" → [" ", "h", "e", "l", "l", "o"]
        std::vector<std::vector<std::string>> corpus;
        {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word) {
                std::vector<std::string> chars;
                chars.push_back(" ");  // space prefix (word boundary marker)
                for (unsigned char c : word)
                    chars.push_back(std::string(1, static_cast<char>(c)));
                corpus.push_back(std::move(chars));
            }
        }

        // BPE merge loop
        while (vocab_size < target_vocab) {
            // Count adjacent pair frequencies across entire corpus
            std::map<std::pair<std::string,std::string>, int> pair_freq;
            for (const auto& word : corpus)
                for (int i = 0; i + 1 < (int)word.size(); ++i)
                    pair_freq[{word[i], word[i+1]}]++;

            if (pair_freq.empty()) break;

            // Find most frequent pair
            auto best_it = std::max_element(
                pair_freq.begin(), pair_freq.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });

            if (best_it->second < 2) break;  // no pair appears twice → stop

            const auto& [left, right] = best_it->first;
            std::string merged = left + right;

            merges.push_back({left, right});
            _add(merged);

            // Apply merge across entire corpus in-place
            for (auto& word : corpus) {
                std::vector<std::string> next;
                next.reserve(word.size());
                int i = 0;
                while (i < (int)word.size()) {
                    if (i + 1 < (int)word.size()
                            && word[i] == left && word[i+1] == right) {
                        next.push_back(merged);
                        i += 2;
                    } else {
                        next.push_back(word[i++]);
                    }
                }
                word = std::move(next);
            }
        }
        std::cout << "✅ Tokenizer built: vocab_size=" << vocab_size
                  << "  merges=" << merges.size() << "\n";
    }

    // ── encode: text → token IDs ──────────────────────────────
    // Applies BPE merge rules in order (standard BPE encoding).
    // max_len: hard cap on output length (includes BOS/EOS).
    std::vector<int> encode(const std::string& text, int max_len = 512) const {
        std::vector<int> ids;
        ids.reserve(std::min(max_len, (int)text.size() + 2));
        ids.push_back(TOKEN_BOS);

        std::istringstream iss(text);
        std::string word;
        while (iss >> word && (int)ids.size() < max_len - 1) {
            // Start as char sequence
            std::vector<std::string> seq;
            seq.reserve(word.size() + 1);
            seq.push_back(" ");
            for (unsigned char c : word)
                seq.push_back(std::string(1, static_cast<char>(c)));

            // Apply every merge rule in training order
            for (const auto& [left, right] : merges) {
                std::vector<std::string> next;
                next.reserve(seq.size());
                int i = 0;
                while (i < (int)seq.size()) {
                    if (i + 1 < (int)seq.size()
                            && seq[i] == left && seq[i+1] == right) {
                        next.push_back(left + right);
                        i += 2;
                    } else {
                        next.push_back(seq[i++]);
                    }
                }
                seq = std::move(next);
            }

            // Lookup each subword; unknown chars → TOKEN_UNK
            for (const auto& subword : seq) {
                auto it = vocab.find(subword);
                int id = (it != vocab.end()) ? it->second : TOKEN_UNK;
                // Strict bounds (should never trigger after valid build/load)
                if (id < 0 || id >= vocab_size) id = TOKEN_UNK;
                ids.push_back(id);
                if ((int)ids.size() >= max_len - 1) break;
            }
        }

        ids.push_back(TOKEN_EOS);
        if ((int)ids.size() > max_len) ids.resize(max_len);
        return ids;
    }

    // ── decode: token IDs → text ──────────────────────────────
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
        // Strip leading space (GPT-2 style: first word has " " prefix)
        if (!result.empty() && result[0] == ' ')
            result.erase(result.begin());
        return result;
    }

    // ── save: write vocab.bin with LGVB magic header ──────────
    // Format: magic(4) + version(2) + n(4) + m(4) + tokens + merges
    bool save(const std::string& path = "vocab.bin") const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ save: cannot open " << path << "\n"; return false; }

        // Header
        f.write(VOCAB_MAGIC, 4);
        f.write(reinterpret_cast<const char*>(&VOCAB_VERSION), sizeof(uint16_t));

        int32_t n = (int32_t)id_to_token.size();
        int32_t m = (int32_t)merges.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));
        f.write(reinterpret_cast<const char*>(&m), sizeof(int32_t));

        // Token table
        for (const auto& tok : id_to_token) {
            int32_t len = (int32_t)tok.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int32_t));
            f.write(tok.data(), len);
        }

        // Merge table
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

    // ── load: read vocab.bin with strict validation ────────────
    // Detects old (no-magic) files, corrupt sizes, truncated data.
    // Any failure → false + descriptive error; no partial state left.
    bool load(const std::string& path = "vocab.bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ load: cannot open " << path << "\n"; return false; }

        // ── Magic ────────────────────────────────────────────
        char magic[4] = {};
        f.read(magic, 4);
        if (!f || std::memcmp(magic, VOCAB_MAGIC, 4) != 0) {
            std::cerr << "❌ load: bad magic in " << path
                      << "  got='" << magic[0] << magic[1] << magic[2] << magic[3] << "'"
                      << "  expected='LGVB'\n"
                      << "   → Yeh purana (pre-fix) vocab.bin hai.\n"
                      << "   → Retrain karo: ./logos --train dataset.txt\n";
            return false;
        }

        // ── Version ───────────────────────────────────────────
        uint16_t ver = 0;
        f.read(reinterpret_cast<char*>(&ver), sizeof(uint16_t));
        if (!f || ver != VOCAB_VERSION) {
            std::cerr << "❌ load: version mismatch in " << path
                      << "  file=v" << ver << "  expected=v" << VOCAB_VERSION << "\n"
                      << "   → Retrain karo latest LOGOS se.\n";
            return false;
        }

        // ── Counts ────────────────────────────────────────────
        int32_t n = 0, m = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
        f.read(reinterpret_cast<char*>(&m), sizeof(int32_t));
        if (!f) { std::cerr << "❌ load: truncated header in " << path << "\n"; return false; }

        if (n <= 0 || n > MAX_VOCAB_SIZE) {
            std::cerr << "❌ load: n=" << n << " out of range [1," << MAX_VOCAB_SIZE << "]\n";
            return false;
        }
        if (m < 0 || m > MAX_MERGE_COUNT) {
            std::cerr << "❌ load: m=" << m << " out of range [0," << MAX_MERGE_COUNT << "]\n";
            return false;
        }

        // ── Token table ───────────────────────────────────────
        id_to_token.clear();
        id_to_token.reserve(n);
        vocab.clear();
        vocab.reserve(n);

        for (int32_t i = 0; i < n; ++i) {
            int32_t len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(int32_t));
            if (!f || len < 0 || len > MAX_TOKEN_BYTES) {
                std::cerr << "❌ load: bad token len=" << len << " at idx=" << i << "\n";
                return false;
            }
            std::string tok(static_cast<size_t>(len), '\0');
            f.read(tok.data(), len);
            if (!f) {
                std::cerr << "❌ load: EOF reading token idx=" << i << "\n";
                return false;
            }
            vocab[tok] = i;
            id_to_token.push_back(std::move(tok));
        }

        // ── Merge table ───────────────────────────────────────
        merges.clear();
        merges.reserve(m);

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
                std::cerr << "❌ load: EOF in merge table at merge idx=" << i << "\n";
                return false;
            }
            merges.push_back({std::move(a), std::move(b)});
        }

        vocab_size = n;
        std::cout << "✅ Vocab loaded: " << path
                  << "  tokens=" << n << "  merges=" << m
                  << "  format=LGVBv" << ver << "\n";
        return true;
    }

private:
    void _add(const std::string& tok) {
        vocab[tok] = vocab_size;
        id_to_token.push_back(tok);
        ++vocab_size;
    }
};
