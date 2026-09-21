// ============================================================
//  LOGOS — DataLoader.cpp
//  Text file se 512-token batches banana
//  TinyStories / Wikipedia / any .txt file support
// ============================================================
#pragma once
#include "../include/Tensor.hpp"
#include "Tokenizer.cpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

class DataLoader {
public:
    std::vector<int> all_tokens;   // Poora dataset flat token IDs mein
    int seq_len;
    int batch_size;
    int current_pos = 0;
    std::mt19937 rng{42};

    DataLoader(const std::string& text_file,
               Tokenizer& tok,
               int seq_len_ = 512,
               int batch_size_ = 32)
        : seq_len(seq_len_), batch_size(batch_size_)
    {
        std::cout << "📂 Loading dataset: " << text_file << "\n";
        std::ifstream f(text_file);
        if (!f) throw std::runtime_error("Cannot open: " + text_file);

        // Poora file ek string mein
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        std::cout << "   File size: " << text.size() / 1024 << " KB\n";

        // Tokenize
        all_tokens = tok.encode(text, (int)text.size());
        std::cout << "   Total tokens: " << all_tokens.size() << "\n";
        std::cout << "   Batches available: "
                  << (all_tokens.size() / seq_len) << "\n";
    }

    // Ek batch return karo: input_ids + target_ids
    // target = input shifted right by 1 (next-token prediction)
    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        if (current_pos + seq_len + 1 >= (int)all_tokens.size()) {
            current_pos = 0;   // Epoch complete — reset
            return false;       // Caller ko pata chalega epoch done
        }

        input_ids.assign(all_tokens.begin() + current_pos,
                         all_tokens.begin() + current_pos + seq_len);
        target_ids.assign(all_tokens.begin() + current_pos + 1,
                          all_tokens.begin() + current_pos + seq_len + 1);
        current_pos += seq_len;
        return true;
    }

    int total_tokens() const { return (int)all_tokens.size(); }
    int total_batches() const { return (int)all_tokens.size() / seq_len; }
};
