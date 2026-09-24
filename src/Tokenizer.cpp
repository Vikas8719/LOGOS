// ============================================================
//  LOGOS — Tokenizer.cpp
//  BPE (Byte Pair Encoding) Tokenizer
//  Text → Token IDs (same vocab.bin training + inference mein)
//
//  Pipeline:
//    1. Character-level vocabulary build karo
//    2. Most frequent adjacent pairs merge karo (BPE)
//    3. vocab.bin mein save karo (magic header + version + size check)
//    4. Encode: text → vector<int>
//    5. Decode: vector<int> → text
//
//  SECURITY FIX (vocab.bin header/version/size validation):
//    Pehle: koi magic/version nahi tha — corrupt/wrong file silently
//           load hoti thi aur OOB reads + bad_alloc possible the.
//    Ab:
//      - 4-byte magic: "LGVB" (LOGOS Vocab Binary)
//      - 2-byte version: VOCAB_VERSION (currently 1)
//      - 4-byte expected_n: save ke waqt ka vocab size
//      - Load par: magic, version, n bounds — teeno check hote hain
//        kisi bhi mismatch par load fail hota hai (no silent corruption)
// ============================================================

#include "../include/Tensor.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>

// ── vocab.bin binary format constants ────────────────────────
static constexpr char     VOCAB_MAGIC[4] = {'L','G','V','B'};  // "LGVB"
static constexpr uint16_t VOCAB_VERSION  = 1;

// ── Special tokens ────────────────────────────────────────────
static constexpr int TOKEN_UNK = 0;   // Unknown
static constexpr int TOKEN_BOS = 1;   // Beginning of sequence
static constexpr int TOKEN_EOS = 2;   // End of sequence
static constexpr int TOKEN_PAD = 3;   // Padding

class Tokenizer {
public:
    // vocab: token_string → id
    std::unordered_map<std::string, int> vocab;
    // id → token_string (for decode)
    std::vector<std::string> id_to_token;
    // BPE merge rules: (a, b) → merged_id
    std::vector<std::pair<std::string,std::string>> merges;

    int vocab_size = 0;
    static constexpr int MAX_VOCAB        = 8192;   // 10M model ke liye enough
    static constexpr int MAX_TOKEN_LEN    = 4096;   // single token max bytes
    static constexpr int MAX_MERGE_COUNT  = MAX_VOCAB * 4;

    // ── Build vocabulary from text ────────────────────────────
    void build(const std::string& text, int target_vocab = 4096) {
        vocab.clear();
        id_to_token.clear();
        merges.clear();

        // Special tokens pehle
        add_token("<UNK>");  // 0
        add_token("<BOS>");  // 1
        add_token("<EOS>");  // 2
        add_token("<PAD>");  // 3

        // Character-level base vocab (UTF-8 bytes 0-255)
        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end())
                add_token(ch);
        }

        // Tokenize text into char sequences (word-level split first)
        std::vector<std::vector<std::string>> word_tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::vector<std::string> chars;
            chars.push_back(" ");   // GPT-2 style: space prefix
            for (char c : word)
                chars.push_back(std::string(1, c));
            word_tokens.push_back(chars);
        }

        // BPE merge loop
        while (vocab_size < target_vocab) {
            std::map<std::pair<std::string,std::string>, int> pair_freq;
            for (const auto& seq : word_tokens) {
                for (int i = 0; i + 1 < (int)seq.size(); ++i)
                    pair_freq[{seq[i], seq[i+1]}]++;
            }
            if (pair_freq.empty()) break;

            auto best = std::max_element(pair_freq.begin(), pair_freq.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });
            if (best->second < 2) break;

            auto [left, right] = best->first;
            std::string merged = left + right;
            merges.push_back({left, right});
            add_token(merged);

            for (auto& seq : word_tokens) {
                std::vector<std::string> new_seq;
                int i = 0;
                while (i < (int)seq.size()) {
                    if (i + 1 < (int)seq.size() && seq[i] == left && seq[i+1] == right) {
                        new_seq.push_back(merged);
                        i += 2;
                    } else {
                        new_seq.push_back(seq[i++]);
                    }
                }
                seq = std::move(new_seq);
            }
        }
        std::cout << "✅ Tokenizer built: vocab_size=" << vocab_size << "\n";
    }

    // ── Encode: string → token IDs ────────────────────────────
    std::vector<int> encode(const std::string& text, int max_len = 512) const {
        std::vector<int> ids;
        ids.push_back(TOKEN_BOS);

        std::istringstream iss(text);
        std::string word;
        while (iss >> word && (int)ids.size() < max_len - 1) {
            std::vector<std::string> seq;
            seq.push_back(" ");
            for (char c : word) seq.push_back(std::string(1, c));

            for (const auto& [left, right] : merges) {
                std::vector<std::string> new_seq;
                int i = 0;
                while (i < (int)seq.size()) {
                    if (i+1 < (int)seq.size() && seq[i]==left && seq[i+1]==right) {
                        new_seq.push_back(left + right);
                        i += 2;
                    } else {
                        new_seq.push_back(seq[i++]);
                    }
                }
                seq = std::move(new_seq);
            }

            for (const auto& tok : seq) {
                // ── SECURITY: token ID bounds strict check ──
                auto it = vocab.find(tok);
                int id = (it != vocab.end()) ? it->second : TOKEN_UNK;
                // Clamp: id must be in [0, vocab_size)
                if (id < 0 || id >= vocab_size) id = TOKEN_UNK;
                ids.push_back(id);
                if ((int)ids.size() >= max_len - 1) break;
            }
        }

        ids.push_back(TOKEN_EOS);
        if ((int)ids.size() > max_len) ids.resize(max_len);
        return ids;
    }

    // ── Decode: token IDs → string ────────────────────────────
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD) continue;
            // ── SECURITY: strict OOB check before indexing ──
            if (id >= 0 && id < (int)id_to_token.size())
                result += id_to_token[id];
            else
                result += "<UNK>";
        }
        if (!result.empty() && result[0] == ' ') result = result.substr(1);
        return result;
    }

    // ── Save vocab to binary file ─────────────────────────────
    // Format (v1):
    //   [4]  magic     = "LGVB"
    //   [2]  version   = VOCAB_VERSION (uint16_t, little-endian)
    //   [4]  n         = vocab token count (int)
    //   [4]  m         = merge rule count (int)
    //   then token entries: [4 len][len bytes] * n
    //   then merge entries: [4 la][la bytes][4 lb][lb bytes] * m
    bool save(const std::string& path = "vocab.bin") const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }

        // ── Header ──
        f.write(VOCAB_MAGIC, 4);
        f.write(reinterpret_cast<const char*>(&VOCAB_VERSION), sizeof(uint16_t));

        int n = (int)id_to_token.size();
        int m = (int)merges.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(int));
        f.write(reinterpret_cast<const char*>(&m), sizeof(int));

        // ── Token table ──
        for (const auto& tok : id_to_token) {
            int len = (int)tok.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int));
            f.write(tok.data(), len);
        }

        // ── Merge table ──
        auto write_str = [&](const std::string& s) {
            int l = (int)s.size();
            f.write(reinterpret_cast<const char*>(&l), sizeof(int));
            f.write(s.data(), l);
        };
        for (const auto& [a, b] : merges) {
            write_str(a);
            write_str(b);
        }

        if (!f) { std::cerr << "❌ Write error: " << path << "\n"; return false; }
        std::cout << "✅ Vocab saved to " << path
                  << " (" << n << " tokens, " << m << " merges, v" << VOCAB_VERSION << ")\n";
        return true;
    }

    // ── Load vocab from binary file ───────────────────────────
    // SECURITY: magic, version, n, len — sab validate karte hain
    // Pehle: koi validation nahi — corrupt file pe bad_alloc / OOB possible
    // Ab:    strict checks, kisi bhi mismatch par early return false
    bool load(const std::string& path = "vocab.bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ Cannot open " << path << "\n"; return false; }

        // ── Magic check ──
        char magic[4] = {};
        f.read(magic, 4);
        if (!f || std::memcmp(magic, VOCAB_MAGIC, 4) != 0) {
            std::cerr << "❌ load(): invalid magic in " << path
                      << " — expected 'LGVB', got '"
                      << magic[0] << magic[1] << magic[2] << magic[3] << "'\n"
                      << "   Kya yeh purana format (no-header) ka vocab.bin hai?\n"
                      << "   Agar haan, toh retrain karo: --train dataset.txt\n";
            return false;
        }

        // ── Version check ──
        uint16_t version = 0;
        f.read(reinterpret_cast<char*>(&version), sizeof(uint16_t));
        if (!f || version != VOCAB_VERSION) {
            std::cerr << "❌ load(): vocab version mismatch in " << path
                      << " — expected v" << VOCAB_VERSION
                      << ", got v" << version << "\n"
                      << "   Retrain karo latest LOGOS se.\n";
            return false;
        }

        // ── n (token count) validate ──
        int n = 0, m = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(int));
        f.read(reinterpret_cast<char*>(&m), sizeof(int));
        if (!f) {
            std::cerr << "❌ load(): truncated header in " << path << "\n";
            return false;
        }
        if (n <= 0 || n > MAX_VOCAB) {
            std::cerr << "❌ load(): invalid vocab count n=" << n
                      << " (expected 1.." << MAX_VOCAB << ") in " << path << "\n";
            return false;
        }
        if (m < 0 || m > MAX_MERGE_COUNT) {
            std::cerr << "❌ load(): invalid merge count m=" << m
                      << " (expected 0.." << MAX_MERGE_COUNT << ") in " << path << "\n";
            return false;
        }

        // ── Token table ──
        id_to_token.clear();
        id_to_token.reserve(n);
        vocab.clear();

        for (int i = 0; i < n; ++i) {
            int len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(int));
            if (!f || len < 0 || len > MAX_TOKEN_LEN) {
                std::cerr << "❌ load(): invalid token length len=" << len
                          << " at index=" << i << " in " << path << "\n";
                return false;
            }
            std::string tok(len, '\0');
            f.read(tok.data(), len);
            if (!f) {
                std::cerr << "❌ load(): unexpected EOF reading token i=" << i
                          << " in " << path << "\n";
                return false;
            }
            id_to_token.push_back(tok);
            vocab[tok] = i;
        }

        // ── Merge table ──
        merges.resize(m);
        auto read_str = [&](std::string& s) -> bool {
            int l = 0;
            f.read(reinterpret_cast<char*>(&l), sizeof(int));
            if (!f || l < 0 || l > MAX_TOKEN_LEN) {
                std::cerr << "❌ load(): invalid merge string length l=" << l
                          << " in " << path << "\n";
                return false;
            }
            s.resize(l);
            f.read(s.data(), l);
            if (!f) {
                std::cerr << "❌ load(): unexpected EOF reading merge string\n";
                return false;
            }
            return true;
        };
        for (auto& [a, b] : merges) {
            if (!read_str(a) || !read_str(b)) return false;
        }

        vocab_size = n;
        std::cout << "✅ Vocab loaded from " << path
                  << " (" << n << " tokens, " << m << " merges, v" << version << ")\n";
        return true;
    }

private:
    void add_token(const std::string& tok) {
        vocab[tok] = vocab_size;
        id_to_token.push_back(tok);
        ++vocab_size;
    }
};

// ── Quick Test (compile with -DLOGOS_TEST_TOKENIZER) ─────────
#ifdef LOGOS_TEST_TOKENIZER
int main() {
    Tokenizer tok;
    std::string corpus = "hello world hello logos vedic math is cool hello world";
    tok.build(corpus, 300);
    tok.save("vocab.bin");

    Tokenizer tok2;
    tok2.load("vocab.bin");

    std::string test = "hello world";
    auto ids = tok2.encode(test);
    std::cout << "Encode '" << test << "' → [";
    for (int i = 0; i < (int)ids.size(); ++i)
        std::cout << ids[i] << (i+1<(int)ids.size()?",":"");
    std::cout << "]\n";

    std::string decoded = tok2.decode(ids);
    std::cout << "Decode → '" << decoded << "'\n";
    if (decoded.find("hello") != std::string::npos &&
        decoded.find("world") != std::string::npos)
        std::cout << "✅ Tokenizer PASS\n";
    else
        std::cout << "❌ Tokenizer FAIL\n";
    return 0;
}
#endif
