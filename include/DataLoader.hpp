#pragma once
// ============================================================
//  LOGOS — DataLoader.hpp
//
//  BUG 9 FIX: Large dataset OOM prevention — streaming mode
//    Pehle: poori dataset std::string mein load hoti thi, phir
//           poori tokenize hoti thi → all_tokens vector.
//           50MB TinyStories → ~50MB string + ~200MB tokens = ~250MB RAM
//           Large corpora pe OOM crash.
//    Ab:    File size check → 64MB se bade datasets streaming mode mein:
//           Sirf 1MB chunk ek waqt mein pad ke tokenize karo.
//           Peak RAM: ~10MB (chunk buffer only).
//           Small datasets (<64MB) pehle ki tarah in-memory.
//
//  BUG 10 FIX: next_batch() ka epoch-end reset logic galat tha
//    Pehle:
//      if (current_pos + seq_len + 1 >= all_tokens.size()) {
//          current_pos = 0;   // Reset kiya...
//          return false;      // ...lekin false return kiya
//      }
//      // Caller (main.cpp):
//      loader.current_pos = 0;  // Manual reset — DUPLICATE, useless
//    Problems:
//      1. Internal reset + caller reset = double reset (confusing)
//      2. Internal reset = silently discards end of dataset on next call
//         (false return ke baad phir call karo → pos=0 se shuru,
//          woh ek batch jo boundary pe tha → skip ho jaata hai)
//      3. Caller ko reset karna ZAROOR tha — agar wo bhool jaata toh
//         next epoch sirf ek batch ke baad end ho jaati silently.
//    Fix:
//      - next_batch() mein current_pos = 0 HATA DIYA on epoch end
//      - False return sirf boundary signal hai — state disturb mat karo
//      - Caller hi reset karega explicitly (clear contract)
//      - main.cpp mein "loader.current_pos = 0" SAHI aur ZARURI hai
//      - DataLoader ab STATELESS epoch management: caller drives epochs
//
//  BUG 11 FIX (carry forward): batch_size parameter removed — was unused
// ============================================================
#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Datasets above this threshold use chunk streaming instead of full load
static constexpr size_t DATALOADER_STREAM_THRESHOLD = 64ULL * 1024 * 1024; // 64MB

class DataLoader {
public:
    // In-memory mode (small datasets <64MB)
    std::vector<int> all_tokens;

    int seq_len;
    int current_pos = 0;  // Caller resets this each epoch (see BUG 10 FIX)

    // Streaming mode (large datasets >=64MB)
    bool        streaming     = false;
    std::string file_path;
    Tokenizer*  tokenizer_ref = nullptr;

    // Chunk buffer for streaming
    std::vector<int> chunk_tokens;
    static constexpr int CHUNK_CHARS = 1 * 1024 * 1024;  // 1MB per chunk
    std::ifstream stream_file;
    bool          stream_eof  = false;

    // BUG 11 FIX (carry forward): batch_size removed — was stored but never used.
    // BUG 9 FIX: streaming mode activated for large files.
    DataLoader(const std::string& text_file, Tokenizer& tok, int seq_len_ = 128)
        : seq_len(seq_len_), file_path(text_file), tokenizer_ref(&tok)
    {
        // Probe file size first
        std::ifstream probe(text_file, std::ios::ate | std::ios::binary);
        if (!probe) throw std::runtime_error("Cannot open dataset: " + text_file);
        size_t file_size = static_cast<size_t>(probe.tellg());
        probe.close();

        if (file_size > DATALOADER_STREAM_THRESHOLD) {
            // BUG 9 FIX: Large file — streaming mode
            streaming = true;
            std::cout << "DataLoader: " << file_size/1024/1024
                      << " MB dataset → streaming mode (1MB chunks)\n";
            stream_file.open(text_file);
            if (!stream_file) throw std::runtime_error("Cannot open stream: " + text_file);
            _load_next_chunk();
        } else {
            // Small file — load fully (safe, convenient, fast random access)
            streaming = false;
            std::ifstream f(text_file);
            std::string text((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            std::cout << "DataLoader: " << file_size/1024 << " KB → in-memory mode\n";
            all_tokens = tok.encode(text, static_cast<int>(text.size()) * 4);

            // Token bounds clamp (security: OOB token IDs → UNK)
            _clamp_tokens(tok.vocab_size);

            std::cout << "Tokens    : " << all_tokens.size()
                      << " | Batches: " << all_tokens.size() / seq_len << "\n";
        }
    }

    // ── next_batch ────────────────────────────────────────────
    // BUG 10 FIX: No more internal current_pos = 0 on epoch end.
    //
    // CONTRACT (clear, explicit):
    //   - Returns true  → input_ids / target_ids filled, ready to use
    //   - Returns false → epoch ended (no data filled)
    //                     Caller MUST reset: loader.current_pos = 0
    //                     then call next_batch() again for next epoch.
    //
    // Pehle (broken):
    //   on false: internal reset happened silently
    //   → caller also reset → double reset, confusing
    //   → boundary batch sometimes skipped silently
    //
    // Ab:
    //   false = "epoch done, YOU decide what to do next"
    //   Caller's explicit "loader.current_pos = 0" in main.cpp is CORRECT
    //   and the ONLY reset that should happen. ✅
    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        if (streaming)
            return _next_batch_streaming(input_ids, target_ids);

        // In-memory mode
        if (current_pos + seq_len + 1 >= (int)all_tokens.size()) {
            // BUG 10 FIX: DO NOT reset current_pos here.
            // Just return false — caller handles reset for next epoch.
            // Pehle: current_pos = 0 yahan hota tha → double reset confusion.
            return false;
        }

        input_ids .assign(all_tokens.begin() + current_pos,
                          all_tokens.begin() + current_pos + seq_len);
        target_ids.assign(all_tokens.begin() + current_pos + 1,
                          all_tokens.begin() + current_pos + seq_len + 1);
        current_pos += seq_len;
        return true;
    }

    int total_batches() const {
        if (streaming)
            return static_cast<int>(chunk_tokens.size()) / seq_len; // estimate
        return static_cast<int>(all_tokens.size()) / seq_len;
    }

    int total_tokens() const {
        if (streaming) return static_cast<int>(chunk_tokens.size());
        return static_cast<int>(all_tokens.size());
    }

private:
    // Load next 1MB chunk into chunk_tokens buffer
    void _load_next_chunk() {
        if (!stream_file || stream_eof) return;

        char buf[CHUNK_CHARS];
        stream_file.read(buf, CHUNK_CHARS);
        std::streamsize got = stream_file.gcount();
        if (got <= 0) { stream_eof = true; return; }
        if (stream_file.eof()) stream_eof = true;

        // Keep tail for sequence continuity across chunks
        std::vector<int> tail;
        if (chunk_tokens.size() > static_cast<size_t>(seq_len + 1)) {
            tail.assign(chunk_tokens.end() - (seq_len + 1), chunk_tokens.end());
        }

        std::string chunk_text(buf, static_cast<size_t>(got));
        auto new_toks = tokenizer_ref->encode(chunk_text, static_cast<int>(got) * 4);

        chunk_tokens = std::move(tail);
        chunk_tokens.insert(chunk_tokens.end(), new_toks.begin(), new_toks.end());
        current_pos = 0;
    }

    // BUG 10 FIX (streaming): same contract — no silent reset on epoch end
    bool _next_batch_streaming(std::vector<int>& input_ids,
                               std::vector<int>& target_ids)
    {
        while (current_pos + seq_len + 1 >= (int)chunk_tokens.size()) {
            if (stream_eof) {
                // End of file — signal epoch end WITHOUT resetting.
                // Caller must reset the stream for next epoch.
                // (Streaming reset: stream_file.seekg(0) + stream_eof=false)
                // For simplicity: main.cpp creates fresh DataLoader each epoch
                // OR caller calls reset_stream() below.
                return false;
            }
            _load_next_chunk();
        }

        input_ids .assign(chunk_tokens.begin() + current_pos,
                          chunk_tokens.begin() + current_pos + seq_len);
        target_ids.assign(chunk_tokens.begin() + current_pos + 1,
                          chunk_tokens.begin() + current_pos + seq_len + 1);
        current_pos += seq_len;
        return true;
    }

    // Token bounds clamp: OOB token IDs → TOKEN_UNK (0)
    void _clamp_tokens(int vocab_size) {
        if (vocab_size <= 0) {
            std::cerr << "⚠️  DataLoader: vocab_size=" << vocab_size
                      << " — skipping bounds check\n";
            return;
        }
        int bad = 0;
        for (int& id : all_tokens)
            if (id < 0 || id >= vocab_size) { ++bad; id = 0; }
        if (bad > 0)
            std::cerr << "⚠️  DataLoader: " << bad
                      << " OOB tokens replaced with UNK (vocab_size=" << vocab_size << ")\n";
        else
            std::cout << "✅ Token bounds OK\n";
    }

public:
    // Reset streaming mode for next epoch (caller-driven)
    void reset_stream() {
        if (!streaming) { current_pos = 0; return; }
        stream_file.clear();
        stream_file.seekg(0, std::ios::beg);
        stream_eof = false;
        chunk_tokens.clear();
        current_pos = 0;
        _load_next_chunk();
    }
};
