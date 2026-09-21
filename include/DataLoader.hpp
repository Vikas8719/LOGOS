#pragma once
// ============================================================
//  LOGOS — DataLoader.hpp
// ============================================================
#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class DataLoader {
public:
    std::vector<int> all_tokens;
    int seq_len, batch_size, current_pos = 0;

    DataLoader(const std::string& text_file, Tokenizer& tok,
               int seq_len_ = 128, int batch_size_ = 1)
        : seq_len(seq_len_), batch_size(batch_size_)
    {
        std::ifstream f(text_file);
        if (!f) throw std::runtime_error("Cannot open: " + text_file);
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        std::cout << "Dataset: " << text.size()/1024 << " KB\n";
        all_tokens = tok.encode(text, (int)text.size());
        std::cout << "Tokens: " << all_tokens.size()
                  << " | Batches: " << all_tokens.size()/seq_len << "\n";
    }

    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        if (current_pos + seq_len + 1 >= (int)all_tokens.size()) {
            current_pos = 0; return false;
        }
        input_ids .assign(all_tokens.begin()+current_pos,
                          all_tokens.begin()+current_pos+seq_len);
        target_ids.assign(all_tokens.begin()+current_pos+1,
                          all_tokens.begin()+current_pos+seq_len+1);
        current_pos += seq_len;
        return true;
    }

    int total_batches() const { return (int)all_tokens.size() / seq_len; }
};
