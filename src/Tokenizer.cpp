

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


static constexpr char     VOCAB_MAGIC[4] = {'L','G','V','B'};  
static constexpr uint16_t VOCAB_VERSION  = 1;

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
    static constexpr int MAX_VOCAB        = 8192;   
    static constexpr int MAX_TOKEN_LEN    = 4096;   
    static constexpr int MAX_MERGE_COUNT  = MAX_VOCAB * 4;

    void build(const std::string& text, int target_vocab = 4096) {
        vocab.clear();
        id_to_token.clear();
        merges.clear();

        add_token("<UNK>");  // 0
        add_token("<BOS>");  // 1
        add_token("<EOS>");  // 2
        add_token("<PAD>");  // 3

        for (int c = 0; c < 256; ++c) {
            std::string ch(1, static_cast<char>(c));
            if (vocab.find(ch) == vocab.end())
                add_token(ch);
        }

        std::vector<std::vector<std::string>> word_tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::vector<std::string> chars;
            chars.push_back(" ");   
            for (char c : word)
                chars.push_back(std::string(1, c));
            word_tokens.push_back(chars);
        }

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
               
                auto it = vocab.find(tok);
                int id = (it != vocab.end()) ? it->second : TOKEN_UNK;
                
                if (id < 0 || id >= vocab_size) id = TOKEN_UNK;
                ids.push_back(id);
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
            else
                result += "<UNK>";
        }
        if (!result.empty() && result[0] == ' ') result = result.substr(1);
        return result;
    }

    
    bool save(const std::string& path = "vocab.bin") const {
        if (path.empty() || path.find('\0') != std::string::npos) {
            std::cerr << "❌ save(): invalid path\n"; return false;
        }
        {
            std::string norm = path;
            for (char& c : norm) if (c == '\\') c = '/';
            std::string seg; std::istringstream ss(norm);
            while (std::getline(ss, seg, '/'))
                if (seg == "..") { std::cerr << "❌ save(): path traversal rejected: " << path << "\n"; return false; }
        }
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }

        f.write(VOCAB_MAGIC, 4);
        f.write(reinterpret_cast<const char*>(&VOCAB_VERSION), sizeof(uint16_t));

        int n = (int)id_to_token.size();
        int m = (int)merges.size();
        f.write(reinterpret_cast<const char*>(&n), sizeof(int));
        f.write(reinterpret_cast<const char*>(&m), sizeof(int));

        for (const auto& tok : id_to_token) {
            int len = (int)tok.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(int));
            f.write(tok.data(), len);
        }

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

    
    bool load(const std::string& path = "vocab.bin") {
        if (path.empty() || path.find('\0') != std::string::npos) {
            std::cerr << "❌ load(): invalid path\n"; return false;
        }
        {
            std::string norm = path;
            for (char& c : norm) if (c == '\\') c = '/';
            std::string seg; std::istringstream ss(norm);
            while (std::getline(ss, seg, '/'))
                if (seg == "..") { std::cerr << "❌ load(): path traversal rejected: " << path << "\n"; return false; }
        }
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "❌ Cannot open " << path << "\n"; return false; }

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

        uint16_t version = 0;
        f.read(reinterpret_cast<char*>(&version), sizeof(uint16_t));
        if (!f || version != VOCAB_VERSION) {
            std::cerr << "❌ load(): vocab version mismatch in " << path
                      << " — expected v" << VOCAB_VERSION
                      << ", got v" << version << "\n"
                      << "   Retrain karo latest LOGOS se.\n";
            return false;
        }

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
