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
#include <cassert>
#include <chrono>

static constexpr int TOKEN_UNK = 0;
static constexpr int TOKEN_BOS = 1;
static constexpr int TOKEN_EOS = 2;
static constexpr int TOKEN_PAD = 3;

static constexpr char     VOCAB_MAGIC[4]  = {'L','G','V','B'};
static constexpr uint16_t VOCAB_VERSION   = 1;
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

    void build(const std::string& text, int target_vocab = 4096) {
        auto t_start = std::chrono::steady_clock::now();
        vocab.clear(); id_to_token.clear(); merges.clear();

        _add("<UNK>");   // 0
        _add("<BOS>");   // 1
        _add("<EOS>");   // 2
        _add("<PAD>");   // 3

        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end()) _add(ch);
        }

        std::unordered_map<std::string,
            std::pair<std::vector<std::string>, int>> word_dict;
        word_dict.reserve(1 << 16);  // pre-alloc ~65k unique words

        {
            std::istringstream iss(text);
            std::string word;
            int words_processed = 0;
            while (iss >> word) {
                std::vector<std::string> chars;
                chars.reserve(word.size() + 1);
                chars.push_back(" ");
                for (unsigned char c : word)
                    chars.push_back(std::string(1, (char)c));

                std::string key = word_key(chars);
                auto it = word_dict.find(key);
                if (it == word_dict.end()) {
                    word_dict[key] = {chars, 1};
                } else {
                    it->second.second++;
                }
                ++words_processed;
            }
            printf("  BPE init: %d unique word types from %d tokens\n",
                   (int)word_dict.size(), words_processed);
            fflush(stdout);
        }

        if (word_dict.size() > 150000) {
            size_t before = word_dict.size();
            int min_freq = 2;
            for (;; ++min_freq) {
                for (auto it = word_dict.begin(); it != word_dict.end(); ) {
                    if (it->second.second < min_freq) it = word_dict.erase(it);
                    else ++it;
                }
                if (word_dict.size() <= 120000 || min_freq >= 6) break;
            }
            printf("  BPE prune: %zu → %zu word types (freq >= %d)\n",
                   before, word_dict.size(), min_freq);
            fflush(stdout);
        }

        if (word_dict.empty()) {
            std::cout << "⚠️  Empty corpus — no BPE merges done.\n";
            return;
        }

        int merges_needed = target_vocab - vocab_size;  // typically 4096-260=3836

        for (int merge_idx = 0; merge_idx < merges_needed; ++merge_idx) {
            std::unordered_map<std::string, int> pair_freq_flat;
            pair_freq_flat.reserve(1 << 14);

            for (auto& [key, val] : word_dict) {
                const auto& toks  = val.first;
                int          freq  = val.second;
                if ((int)toks.size() < 2) continue;
                for (int i = 0; i + 1 < (int)toks.size(); ++i) {
                    std::string pk = toks[i] + '\x01' + toks[i+1];
                    pair_freq_flat[pk] += freq;
                }
            }

            if (pair_freq_flat.empty()) break;

            auto best_it = std::max_element(
                pair_freq_flat.begin(), pair_freq_flat.end(),
                [](const auto& a, const auto& b){ return a.second < b.second; });

            if (best_it->second < 2) break;  // no pair appears 2+ times

            const std::string& pk = best_it->first;
            size_t sep = pk.find('\x01');
            std::string left  = pk.substr(0, sep);
            std::string right = pk.substr(sep + 1);
            std::string merged = left + right;

            merges.push_back({left, right});
            _add(merged);

            std::unordered_map<std::string,
                std::pair<std::vector<std::string>, int>> new_dict;
            new_dict.reserve(word_dict.size());

            for (auto& [key, val] : word_dict) {
                auto& toks = val.first;
                int   freq = val.second;

                bool has_pair = false;
                for (int i = 0; i + 1 < (int)toks.size(); ++i) {
                    if (toks[i] == left && toks[i+1] == right) {
                        has_pair = true; break;
                    }
                }

                if (!has_pair) {
                    new_dict[key] = std::move(val);
                } else {
                    std::vector<std::string> next;
                    next.reserve(toks.size());
                    int i = 0;
                    while (i < (int)toks.size()) {
                        if (i + 1 < (int)toks.size()
                                && toks[i] == left && toks[i+1] == right) {
                            next.push_back(merged);
                            i += 2;
                        } else {
                            next.push_back(toks[i++]);
                        }
                    }
                    std::string new_key = word_key(next);
                    auto it2 = new_dict.find(new_key);
                    if (it2 == new_dict.end()) {
                        new_dict[new_key] = {std::move(next), freq};
                    } else {
                        it2->second.second += freq;
                    }
                }
            }
            word_dict = std::move(new_dict);

            if (merge_idx % 500 == 0 && merge_idx > 0) {
                auto now = std::chrono::steady_clock::now();
                float secs = std::chrono::duration<float>(now - t_start).count();
                printf("  BPE merge %d/%d | vocab=%d | %.1fs elapsed\n",
                       merge_idx, merges_needed, vocab_size, secs);
                fflush(stdout);
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        float total_secs = std::chrono::duration<float>(t_end - t_start).count();
        printf("✅ Tokenizer built: vocab_size=%d  merges=%d  time=%.1fs\n",
               vocab_size, (int)merges.size(), total_secs);
        fflush(stdout);
    }

    std::vector<int> encode(const std::string& text, int max_len = 512) const {
        std::vector<int> ids;
        ids.reserve(std::min(max_len, (int)text.size() + 2));
        ids.push_back(TOKEN_BOS);

        std::istringstream iss(text);
        std::string word;
        std::unordered_map<std::string, std::vector<int>> word_cache;
        while (iss >> word && (int)ids.size() < max_len - 1) {
            auto cached = word_cache.find(word);
            if (cached != word_cache.end()) {
                for (int cid : cached->second) {
                    ids.push_back(cid);
                    if ((int)ids.size() >= max_len - 1) break;
                }
                continue;
            }
            std::vector<int> word_ids;
            std::vector<std::string> seq;
            seq.reserve(word.size() + 1);
            seq.push_back(" ");
            for (unsigned char c : word)
                seq.push_back(std::string(1, static_cast<char>(c)));

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

            for (const auto& subword : seq) {
                auto it = vocab.find(subword);
                int id = (it != vocab.end()) ? it->second : TOKEN_UNK;
                if (id < 0 || id >= vocab_size) id = TOKEN_UNK;
                word_ids.push_back(id);
                ids.push_back(id);
                if ((int)ids.size() >= max_len - 1) break;
            }
            if ((int)ids.size() < max_len - 1)
                word_cache.emplace(word, std::move(word_ids));
        }

        ids.push_back(TOKEN_EOS);
        if ((int)ids.size() > max_len) ids.resize(max_len);
        return ids;
    }

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
        if (!result.empty() && result[0] == ' ')
            result.erase(result.begin());
        return result;
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
                      << "  expected=v" << VOCAB_VERSION << "\n";
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
