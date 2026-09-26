// ============================================================
//  LOGOS — DataLoader.cpp
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
    int current_pos = 0;
    std::mt19937 rng{42};

    // ── Constructor ───────────────────────────────────────────
    // BUG 11 FIX: batch_size_ param removed — was stored but never used.
    DataLoader(const std::string& text_file,
               Tokenizer& tok,
               int seq_len_ = 512)
        : seq_len(seq_len_)
    {
        std::cout << "📂 Loading dataset: " << text_file << "\n";
        std::ifstream f(text_file);
        if (!f) throw std::runtime_error("Cannot open: " + text_file);

        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        std::cout << "   File size: " << text.size() / 1024 << " KB\n";

        // Tokenize
        all_tokens = tok.encode(text, (int)text.size());
        std::cout << "   Total tokens (raw): " << all_tokens.size() << "\n";

        // ── SECURITY: Token bounds strict check ──────────────
        // Dataset mein OOB token IDs → embedding OOB → silent corruption
        // Ab: clamp karo, bad tokens count karo, warn karo
        clamp_tokens(tok.vocab_size);

        std::cout << "   Batches available: "
                  << (all_tokens.size() / seq_len) << "\n";
    }

    // ── next_batch: ek input/target pair return karo ─────────
    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        if (current_pos + seq_len + 1 >= (int)all_tokens.size()) {
            current_pos = 0;   // Epoch complete — reset
            return false;
        }

        input_ids.assign(all_tokens.begin() + current_pos,
                         all_tokens.begin() + current_pos + seq_len);
        target_ids.assign(all_tokens.begin() + current_pos + 1,
                          all_tokens.begin() + current_pos + seq_len + 1);

        // ── SECURITY: Double-check bounds before returning ──
        // (pehle clamp ho chuki hai lekin defensive check keeps model safe)
        for (int& id : input_ids)  if (id < 0 || id >= (int)all_tokens.size()) id = 0;
        for (int& id : target_ids) if (id < 0 || id >= (int)all_tokens.size()) id = 0;

        current_pos += seq_len;
        return true;
    }

    int total_tokens()  const { return (int)all_tokens.size(); }
    int total_batches() const { return (int)all_tokens.size() / seq_len; }

private:
    // ── clamp_tokens: OOB token IDs ko TOKEN_UNK se replace karo ──
    void clamp_tokens(int vocab_size) {
        if (vocab_size <= 0) {
            std::cerr << "⚠️  DataLoader: vocab_size=" << vocab_size
                      << " — token bounds check skip (vocab not built yet?)\n";
            return;
        }

        int bad_count = 0;
        for (int& id : all_tokens) {
            if (id < 0 || id >= vocab_size) {
                ++bad_count;
                id = 0;  // TOKEN_UNK
            }
        }

        if (bad_count > 0) {
            std::cerr << "⚠️  DataLoader: " << bad_count
                      << " out-of-range token IDs found (vocab_size=" << vocab_size << ")\n"
                      << "   In-range expected: [0, " << vocab_size - 1 << "]\n"
                      << "   Replaced with TOKEN_UNK (0).\n"
                      << "   Possible cause: vocab.bin aur dataset mismatch — retrain karo.\n";
        } else {
            std::cout << "✅ Token bounds OK — all " << all_tokens.size()
                      << " tokens in [0, " << vocab_size - 1 << "]\n";
        }
    }
};
