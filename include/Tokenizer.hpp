#pragma once
// ============================================================
//  LOGOS — Tokenizer.hpp  (header-only, #pragma once protected)
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

static constexpr int TOKEN_UNK = 0;
static constexpr int TOKEN_BOS = 1;
static constexpr int TOKEN_EOS = 2;
static constexpr int TOKEN_PAD = 3;

class Tokenizer {
public:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> id_to_token;
    std::vector<std::pair<std::string,std::string>> merges;
    int vocab_size = 0;
    static constexpr int MAX_VOCAB = 8192;

    void build(const std::string& text, int target_vocab = 4096) {
        vocab.clear(); id_to_token.clear(); merges.clear();
        add_token("<UNK>"); add_token("<BOS>");
        add_token("<EOS>"); add_token("<PAD>");

        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end()) add_token(ch);
        }

        std::vector<std::vector<std::string>> word_tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::vector<std::string> chars;
            chars.push_back(" ");
            for (char c : word) chars.push_back(std::string(1, c));
            word_tokens.push_back(chars);
        }

        while (vocab_size < target_vocab) {
            std::map<std::pair<std::string,std::string>, int> pair_freq;
            for (const auto& seq : word_tokens)
                for (int i = 0; i+1 < (int)seq.size(); ++i)
                    pair_freq[{seq[i], seq[i+1]}]++;
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
                    if (i+1 < (int)seq.size() && seq[i]==left && seq[i+1]==right) {
                        new_seq.push_back(merged); i += 2;
                    } else { new_seq.push_back(seq[i++]); }
                }
                seq = std::move(new_seq);
            }
        }
        std::cout << "Tokenizer built: vocab_size=" << vocab_size << "\n";
    }

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
                        new_seq.push_back(left+right); i += 2;
                    } else { new_seq.push_back(seq[i++]); }
                }
                seq = std::move(new_seq);
            }
            for (const auto& tok : seq) {
                auto it = vocab.find(tok);
                ids.push_back(it != vocab.end() ? it->second : TOKEN_UNK);
                if ((int)ids.size() >= max_len - 1) break;
            }
        }
        ids.push_back(TOKEN_EOS);
        if ((int)ids.size() > max_len) ids.resize(max_len);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD) continue;
            if (id >= 0 && id < (int)id_to_token.size())
                result += id_to_token[id];
            else result += "<UNK>";
        }
        if (!result.empty() && result[0] == ' ') result = result.substr(1);
        return result;
    }

    bool save(const std::string& path = "vocab.bin") const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
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
            auto ws = [&](const std::string& s){
                int l = s.size();
                f.write(reinterpret_cast<const char*>(&l), sizeof(int));
                f.write(s.data(), l);
            };
            ws(a); ws(b);
        }
        std::cout << "Vocab saved: " << path << " (" << n << " tokens)\n";
        return true;
    }

    bool load(const std::string& path = "vocab.bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        int n; f.read(reinterpret_cast<char*>(&n), sizeof(int));
        id_to_token.resize(n); vocab.clear();
        for (int i = 0; i < n; ++i) {
            int len; f.read(reinterpret_cast<char*>(&len), sizeof(int));
            id_to_token[i].resize(len);
            f.read(id_to_token[i].data(), len);
            vocab[id_to_token[i]] = i;
        }
        int m; f.read(reinterpret_cast<char*>(&m), sizeof(int));
        merges.resize(m);
        for (auto& [a,b] : merges) {
            auto rs = [&](std::string& s){
                int l; f.read(reinterpret_cast<char*>(&l), sizeof(int));
                s.resize(l); f.read(s.data(), l);
            };
            rs(a); rs(b);
        }
        vocab_size = n;
        std::cout << "Vocab loaded: " << path << " (" << n << " tokens)\n";
        return true;
    }

private:
    void add_token(const std::string& tok) {
        vocab[tok] = vocab_size;
        id_to_token.push_back(tok);
        ++vocab_size;
    }
};
