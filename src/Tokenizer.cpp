// ============================================================
//  LOGOS — Tokenizer.cpp
//  BPE (Byte Pair Encoding) Tokenizer
//  Text → Token IDs (same vocab.bin training + inference mein)
//
//  Pipeline:
//    1. Character-level vocabulary build karo
//    2. Most frequent adjacent pairs merge karo (BPE)
//    3. vocab.bin mein save karo
//    4. Encode: text → vector<int>
//    5. Decode: vector<int> → text
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
    static constexpr int MAX_VOCAB = 8192;   // 10M model ke liye enough

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

        // Tokenize text into char sequences
        // Word-level split first (space-separated)
        std::vector<std::vector<std::string>> word_tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::vector<std::string> chars;
            // Add space prefix (GPT-2 style: ▁word)
            chars.push_back(" ");
            for (char c : word)
                chars.push_back(std::string(1, c));
            word_tokens.push_back(chars);
        }

        // BPE merge loop
        while (vocab_size < target_vocab) {
            // Count adjacent pairs
            std::map<std::pair<std::string,std::string>, int> pair_freq;
            for (const auto& seq : word_tokens) {
                for (int i = 0; i + 1 < (int)seq.size(); ++i)
                    pair_freq[{seq[i], seq[i+1]}]++;
            }
            if (pair_freq.empty()) break;

            // Most frequent pair
            auto best = std::max_element(pair_freq.begin(), pair_freq.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });
            if (best->second < 2) break;  // No pair appears >= 2 times

            auto [left, right] = best->first;
            std::string merged = left + right;
            merges.push_back({left, right});
            add_token(merged);

            // Apply merge to all sequences
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

        // Split into words
        std::istringstream iss(text);
        std::string word;
        while (iss >> word && (int)ids.size() < max_len - 1) {
            // Char-level split
            std::vector<std::string> seq;
            seq.push_back(" ");
            for (char c : word) seq.push_back(std::string(1, c));

            // Apply BPE merges in order
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

            // Token → ID
            for (const auto& tok : seq) {
                auto it = vocab.find(tok);
                ids.push_back(it != vocab.end() ? it->second : TOKEN_UNK);
                if ((int)ids.size() >= max_len - 1) break;
            }
        }

        ids.push_back(TOKEN_EOS);

        // Truncate hard limit — Context Overflow protection
        if ((int)ids.size() > max_len) ids.resize(max_len);
        return ids;
    }

    // ── Decode: token IDs → string ────────────────────────────
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD) continue;
            if (id >= 0 && id < (int)id_to_token.size())
                result += id_to_token[id];
            else
                result += "<UNK>";
        }
        // Remove leading space artifact
        if (!result.empty() && result[0] == ' ') result = result.substr(1);
        return result;
    }

    // ── Save vocab to binary file ─────────────────────────────
    // CRITICAL: Same vocab.bin training + inference dono mein
    bool save(const std::string& path = "vocab.bin") const {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }

        int n = (int)id_to_token.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(int));
        for (const auto& tok : id_to_token) {
            int len = (int)tok.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int));
            f.write(tok.data(), len);
        }
        int m = (int)merges.size();
        f.write(reinterpret_cast<const char*>(&m), sizeof(int));
        for (const auto& [a,b] : merges) {
            auto write_str = [&](const std::string& s){
                int l = s.size();
                f.write(reinterpret_cast<const char*>(&l), sizeof(int));
                f.write(s.data(), l);
            };
            write_str(a); write_str(b);
        }
        std::cout << "✅ Vocab saved to " << path << " (" << n << " tokens)\n";
        return true;
    }

    // ── Load vocab from binary file ───────────────────────────
    bool load(const std::string& path = "vocab.bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }

        int n; f.read(reinterpret_cast<char*>(&n), sizeof(int));
        id_to_token.resize(n);
        vocab.clear();
        for (int i = 0; i < n; ++i) {
            int len; f.read(reinterpret_cast<char*>(&len), sizeof(int));
            id_to_token[i].resize(len);
            f.read(id_to_token[i].data(), len);
            vocab[id_to_token[i]] = i;
        }
        int m; f.read(reinterpret_cast<char*>(&m), sizeof(int));
        merges.resize(m);
        for (auto& [a,b] : merges) {
            auto read_str = [&](std::string& s){
                int l; f.read(reinterpret_cast<char*>(&l), sizeof(int));
                s.resize(l); f.read(s.data(), l);
            };
            read_str(a); read_str(b);
        }
        vocab_size = n;
        std::cout << "✅ Vocab loaded from " << path << " (" << n << " tokens)\n";
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
