#pragma once
#include "Tensor.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>

static constexpr size_t DATALOADER_STREAM_THRESHOLD = 64ULL * 1024 * 1024; // 64MB

class DataLoader {
public:
    std::vector<int> all_tokens;

    int seq_len;
    int current_pos = 0;  // Caller resets this each epoch (see BUG 10 FIX)

    bool        streaming     = false;
    std::string file_path;
    Tokenizer*  tokenizer_ref = nullptr;

    std::vector<int> chunk_tokens;
    static constexpr int CHUNK_CHARS = 1 * 1024 * 1024;  // 1MB per chunk
    std::ifstream stream_file;
    bool          stream_eof  = false;

    DataLoader(const std::string& text_file, Tokenizer& tok, int seq_len_ = 128)
        : seq_len(seq_len_), file_path(text_file), tokenizer_ref(&tok)
    {
        std::ifstream probe(text_file, std::ios::ate | std::ios::binary);
        if (!probe) throw std::runtime_error("Cannot open dataset: " + text_file);
        size_t file_size = static_cast<size_t>(probe.tellg());
        probe.close();

        if (file_size > DATALOADER_STREAM_THRESHOLD) {
            streaming = true;
            std::cout << "DataLoader: " << file_size/1024/1024
                      << " MB dataset → streaming mode (1MB chunks)\n";
            stream_file.open(text_file);
            if (!stream_file) throw std::runtime_error("Cannot open stream: " + text_file);
            _load_next_chunk();
        } else {
            streaming = false;
            std::ifstream f(text_file);
            std::string text((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            std::cout << "DataLoader: " << file_size/1024 << " KB → in-memory mode\n";
            all_tokens = tok.encode(text, static_cast<int>(text.size()) * 4);

            _clamp_tokens(tok.vocab_size);

            std::cout << "Tokens    : " << all_tokens.size()
                      << " | Batches: " << all_tokens.size() / seq_len << "\n";
        }
    }

    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        if (streaming)
            return _next_batch_streaming(input_ids, target_ids);

        if (current_pos + seq_len + 1 >= (int)all_tokens.size()) {
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
    void _load_next_chunk() {
        if (!stream_file || stream_eof) return;

        std::vector<char> buf(CHUNK_CHARS);
        stream_file.read(buf.data(), CHUNK_CHARS);
        std::streamsize got = stream_file.gcount();
        if (got <= 0) { stream_eof = true; return; }
        if (stream_file.eof()) stream_eof = true;

        std::vector<int> tail;
        if (chunk_tokens.size() > static_cast<size_t>(seq_len + 1)) {
            tail.assign(chunk_tokens.end() - (seq_len + 1), chunk_tokens.end());
        }

        std::string chunk_text(buf.data(), static_cast<size_t>(got));
        auto new_toks = tokenizer_ref->encode(chunk_text, static_cast<int>(got) * 4);

        chunk_tokens = std::move(tail);
        chunk_tokens.insert(chunk_tokens.end(), new_toks.begin(), new_toks.end());
        current_pos = 0;
    }

    bool _next_batch_streaming(std::vector<int>& input_ids,
                               std::vector<int>& target_ids)
    {
        while (current_pos + seq_len + 1 >= (int)chunk_tokens.size()) {
            if (stream_eof) {
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
