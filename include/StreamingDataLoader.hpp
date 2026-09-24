#pragma once
// ============================================================
//  LOGOS — include/StreamingDataLoader.hpp  (Bug 6 Fix)
//
//  BUG 6 FIX: Huge data compatibility
//
//  Problems fixed:
//  1. Tokenizer sirf 50MB padh raha tha lekin ds_size full file se
//     → Vocab ab actual_text_size par base hoga, not raw file size
//  2. Dataset fully RAM mein load ho raha tha
//     → True streaming: file kabhi fully load nahi hoti
//  3. Batch size fixed=1, no gradient accumulation
//     → grad_accum_steps parameter add kiya
//  4. int indexing — 200MB+ datasets par step counter overflow
//     → int64_t / size_t everywhere
//  5. total_batches() streaming mode mein sirf 1 chunk estimate karta tha
//     → ab actual file scan se accurate count milta hai
//  6. No prefetching — CPU tokenization stalls GPU
//     → Background thread mein next chunk prefetch
// ============================================================
#include "Tokenizer.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>

// ── Shard: ek file ya file ka ek part ─────────────────────────
struct DataShard {
    std::string file_path;
    int64_t byte_offset = 0;    // kahan se padho
    int64_t byte_length = -1;   // -1 = end of file
};

// ── BUG 6 FIX: Dataset scan — actual processed text size ──────
// train_gpu.cu mein pehle ds_size = raw file size
// lekin tokenizer sirf 50MB padha → mismatch
// Ab: scan_dataset_size() actual bytes jo tokenizer dekhega
inline int64_t scan_dataset_size(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) return 0;
    return (int64_t)f.tellg();
}

// ── BUG 6 FIX: Vocab size decide karo processed bytes se ──────
// Pehle: vocab_sz = ds_size>200MB ? 8192 : ...
//        lekin tokenizer ne sirf 50MB dekha → wrong vocab
// Ab:    vocab decision actual text size par hoga
inline int decide_vocab_size(int64_t actual_text_bytes) {
    if (actual_text_bytes > 200LL * 1024 * 1024) return 8192;
    if (actual_text_bytes >  50LL * 1024 * 1024) return 4096;
    if (actual_text_bytes >  10LL * 1024 * 1024) return 2048;
    return 1024;
}

// ── BUG 6 FIX: Model config decide karo actual bytes se ───────
struct ScaledConfig {
    int d_model, num_heads, num_layers, max_seq_len;
};

inline ScaledConfig decide_model_config(int64_t actual_text_bytes) {
    if (actual_text_bytes > 200LL * 1024 * 1024)
        return {256, 8, 6, 256};
    if (actual_text_bytes >  50LL * 1024 * 1024)
        return {128, 8, 4, 128};
    if (actual_text_bytes >  10LL * 1024 * 1024)
        return { 96, 6, 3, 128};
    return { 64, 4, 2,  64};
}

// ── StreamingDataLoader ───────────────────────────────────────
// True streaming: kabhi full file RAM mein nahi aata
// Prefetching: background thread next chunk tokenize karta hai
// int64_t: large file indexing safe
class StreamingDataLoader {
public:
    // BUG 6 FIX: int64_t step/batch counters (int = ~2B limit)
    int64_t total_tokens_seen  = 0;
    int64_t total_steps_done   = 0;
    int     seq_len;
    int     grad_accum_steps;   // BUG 6: gradient accumulation support
    int     current_epoch      = 0;

    // Shard support — multiple files
    std::vector<DataShard> shards;
    int     current_shard      = 0;

    // ── Constructor: Single file ──────────────────────────────
    StreamingDataLoader(const std::string& text_file,
                        Tokenizer& tok,
                        int seq_len_        = 128,
                        int grad_accum_     = 1,
                        int64_t chunk_bytes = 4LL * 1024 * 1024)  // 4MB default chunk
        : seq_len(seq_len_)
        , grad_accum_steps(grad_accum_)
        , tokenizer_ref_(&tok)
        , chunk_bytes_(chunk_bytes)
    {
        // Validate
        if (seq_len_ <= 0)
            throw std::invalid_argument("seq_len must be > 0");
        if (grad_accum_ <= 0)
            throw std::invalid_argument("grad_accum_steps must be > 0");

        int64_t file_size = scan_dataset_size(text_file);
        if (file_size == 0)
            throw std::runtime_error("Dataset empty or not found: " + text_file);

        DataShard shard;
        shard.file_path   = text_file;
        shard.byte_offset = 0;
        shard.byte_length = file_size;
        shards.push_back(shard);

        // BUG 6 FIX: actual_text_size = file_size (not hardcapped 50MB)
        // train_gpu.cu mein vocab/config decision ye value se karega
        actual_text_size_ = file_size;

        _open_current_shard();
        _load_next_chunk_sync();  // First chunk synchronous
        _launch_prefetch();       // Next chunk prefetch start
    }

    // ── Constructor: Multiple shards / files ──────────────────
    StreamingDataLoader(const std::vector<DataShard>& shards_,
                        Tokenizer& tok,
                        int seq_len_        = 128,
                        int grad_accum_     = 1,
                        int64_t chunk_bytes = 4LL * 1024 * 1024)
        : seq_len(seq_len_)
        , grad_accum_steps(grad_accum_)
        , tokenizer_ref_(&tok)
        , chunk_bytes_(chunk_bytes)
        , shards(shards_)
    {
        if (shards.empty())
            throw std::invalid_argument("No shards provided");

        // BUG 6 FIX: total size = sum of all shards
        actual_text_size_ = 0;
        for (const auto& s : shards) {
            int64_t sz = s.byte_length >= 0 ? s.byte_length
                                             : scan_dataset_size(s.file_path);
            actual_text_size_ += sz;
        }

        _open_current_shard();
        _load_next_chunk_sync();
        _launch_prefetch();
    }

    ~StreamingDataLoader() {
        prefetch_stop_ = true;
        prefetch_cv_.notify_all();
        if (prefetch_thread_.joinable()) prefetch_thread_.join();
    }

    // BUG 6 FIX: Actual text size exposed for vocab/config decisions
    int64_t actual_text_size() const { return actual_text_size_; }

    // ── next_batch: returns one seq_len window ────────────────
    // Returns false at epoch boundary (loader resets automatically)
    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        // Need seq_len + 1 tokens
        while (curr_pos_ + seq_len + 1 >= (int)curr_chunk_.size()) {
            // Swap in prefetched chunk
            bool got = _swap_prefetched_chunk();
            if (!got) {
                // End of all shards → epoch done
                current_epoch++;
                current_shard = 0;
                curr_pos_     = 0;
                _open_current_shard();
                _load_next_chunk_sync();
                _launch_prefetch();
                return false;
            }
        }

        input_ids .assign(curr_chunk_.begin() + curr_pos_,
                          curr_chunk_.begin() + curr_pos_ + seq_len);
        target_ids.assign(curr_chunk_.begin() + curr_pos_ + 1,
                          curr_chunk_.begin() + curr_pos_ + seq_len + 1);
        curr_pos_ += seq_len;
        total_tokens_seen += seq_len;
        return true;
    }

    // ── next_accum_batch: returns grad_accum_steps micro-batches ─
    // BUG 6 FIX: Gradient accumulation — mehrein micro-batch ek
    // optimizer step ke liye
    // Returns: vector of (input, target) pairs to accumulate over
    bool next_accum_batch(
        std::vector<std::pair<std::vector<int>, std::vector<int>>>& micro_batches)
    {
        micro_batches.clear();
        for (int i = 0; i < grad_accum_steps; ++i) {
            std::vector<int> inp, tgt;
            bool ok = next_batch(inp, tgt);
            if (!ok) {
                // Epoch ended mid-accumulation — flush what we have
                return !micro_batches.empty();
            }
            micro_batches.push_back({std::move(inp), std::move(tgt)});
        }
        return true;
    }

    // ── BUG 6 FIX: Accurate total_batches ─────────────────────
    // Pehle: sirf current chunk size / seq_len → underestimate
    // Ab:    total file size / avg_tokens_per_byte estimate
    int64_t total_batches_per_epoch() const {
        // BPE tokens average ~0.6–0.8 per byte of raw text
        // Conservative estimate: 0.65
        double tokens_per_byte = 0.65;
        int64_t est_tokens = (int64_t)(actual_text_size_ * tokens_per_byte);
        return est_tokens / seq_len;
    }

    // ── Checkpoint: save/restore loader state ─────────────────
    // BUG 6 FIX: OOM-safe checkpointing — save loader position
    // so training can resume without re-reading from start
    struct LoaderState {
        int     current_shard;
        int64_t byte_pos;       // position in current shard
        int64_t total_tokens_seen;
        int64_t total_steps_done;
        int     current_epoch;
    };

    LoaderState get_state() const {
        return {
            current_shard,
            shard_byte_pos_,
            total_tokens_seen,
            total_steps_done,
            current_epoch
        };
    }

    void restore_state(const LoaderState& state) {
        current_shard     = state.current_shard;
        total_tokens_seen = state.total_tokens_seen;
        total_steps_done  = state.total_steps_done;
        current_epoch     = state.current_epoch;

        // Reposition file stream
        if (current_shard < (int)shards.size()) {
            _open_current_shard();
            int64_t skip = state.byte_pos + shards[current_shard].byte_offset;
            stream_.seekg(skip, std::ios::beg);
            shard_byte_pos_ = state.byte_pos;
        }
        curr_chunk_.clear();
        curr_pos_ = 0;
        _load_next_chunk_sync();
        _launch_prefetch();
    }

private:
    Tokenizer*    tokenizer_ref_;
    int64_t       chunk_bytes_;
    int64_t       actual_text_size_ = 0;

    // Current active chunk
    std::vector<int>  curr_chunk_;
    int               curr_pos_ = 0;

    // Prefetch state
    std::vector<int>  prefetch_chunk_;
    bool              prefetch_ready_ = false;
    bool              prefetch_stop_  = false;
    bool              shard_eof_      = false;
    std::thread             prefetch_thread_;
    std::mutex              prefetch_mutex_;
    std::condition_variable prefetch_cv_;

    // File stream
    std::ifstream  stream_;
    int64_t        shard_byte_pos_ = 0;  // bytes consumed in current shard

    void _open_current_shard() {
        if (stream_.is_open()) stream_.close();
        if (current_shard >= (int)shards.size()) return;

        const auto& s = shards[current_shard];
        stream_.open(s.file_path, std::ios::binary);
        if (!stream_)
            throw std::runtime_error("Cannot open shard: " + s.file_path);

        stream_.seekg(s.byte_offset, std::ios::beg);
        shard_byte_pos_ = 0;
        shard_eof_      = false;
        curr_chunk_.clear();
        curr_pos_ = 0;
    }

    // Read next chunk_bytes_ bytes, tokenize, return tokens
    std::vector<int> _read_and_tokenize_chunk() {
        if (!stream_ || shard_eof_) return {};

        const auto& shard = shards[current_shard];
        int64_t remaining = (shard.byte_length >= 0)
            ? shard.byte_length - shard_byte_pos_
            : INT64_MAX;
        if (remaining <= 0) { shard_eof_ = true; return {}; }

        int64_t to_read = std::min(chunk_bytes_, remaining);
        std::vector<char> buf(static_cast<size_t>(to_read));
        stream_.read(buf.data(), to_read);
        std::streamsize got = stream_.gcount();

        if (got <= 0) { shard_eof_ = true; return {}; }
        if (stream_.eof()) shard_eof_ = true;

        shard_byte_pos_ += got;

        // Tokenize this chunk of text
        std::string text(buf.data(), static_cast<size_t>(got));
        // max_len estimate: ~4 tokens per byte upper bound
        return tokenizer_ref_->encode(text, static_cast<int>(got) * 4);
    }

    void _load_next_chunk_sync() {
        auto new_tokens = _read_and_tokenize_chunk();

        if (new_tokens.empty() && shard_eof_) {
            // Try next shard
            current_shard++;
            if (current_shard < (int)shards.size()) {
                _open_current_shard();
                new_tokens = _read_and_tokenize_chunk();
            }
        }

        // Keep tail for context continuity
        std::vector<int> tail;
        if (curr_chunk_.size() > static_cast<size_t>(seq_len + 1)) {
            tail.assign(curr_chunk_.end() - seq_len - 1, curr_chunk_.end());
        }
        curr_chunk_ = std::move(tail);
        curr_chunk_.insert(curr_chunk_.end(), new_tokens.begin(), new_tokens.end());
        curr_pos_ = 0;
    }

    // ── BUG 6 FIX: Background prefetch thread ─────────────────
    // CPU tokenization GPU ko stall karta tha (synchronous)
    // Ab: next chunk background mein tokenize hota hai
    void _launch_prefetch() {
        // Stop previous thread if running
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_ready_ = false;
        }
        if (prefetch_thread_.joinable()) prefetch_thread_.join();
        if (prefetch_stop_) return;

        prefetch_thread_ = std::thread([this]() {
            auto tokens = _read_and_tokenize_chunk();

            if (tokens.empty() && shard_eof_) {
                // Try next shard in background
                int next_shard = current_shard + 1;
                if (next_shard < (int)shards.size()) {
                    // Read from next shard (don't modify current_shard yet)
                    std::ifstream tmp(shards[next_shard].file_path, std::ios::binary);
                    if (tmp) {
                        tmp.seekg(shards[next_shard].byte_offset, std::ios::beg);
                        std::vector<char> buf(static_cast<size_t>(chunk_bytes_));
                        tmp.read(buf.data(), chunk_bytes_);
                        std::streamsize got = tmp.gcount();
                        if (got > 0) {
                            std::string text(buf.data(), static_cast<size_t>(got));
                            tokens = tokenizer_ref_->encode(
                                text, static_cast<int>(got) * 4);
                        }
                    }
                }
            }

            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_chunk_ = std::move(tokens);
            prefetch_ready_ = true;
            prefetch_cv_.notify_one();
        });
    }

    // Swap prefetched chunk into curr_chunk_, launch next prefetch
    // Returns false if no more data
    bool _swap_prefetched_chunk() {
        // Wait for prefetch
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this]{ return prefetch_ready_ || prefetch_stop_; });

        if (prefetch_chunk_.empty()) return false;

        // Merge: keep tail of curr_chunk for continuity
        std::vector<int> tail;
        if (curr_chunk_.size() > static_cast<size_t>(seq_len + 1)) {
            tail.assign(curr_chunk_.end() - seq_len - 1, curr_chunk_.end());
        }
        curr_chunk_ = std::move(tail);
        curr_chunk_.insert(curr_chunk_.end(),
                           prefetch_chunk_.begin(), prefetch_chunk_.end());
        prefetch_chunk_.clear();
        prefetch_ready_ = false;
        curr_pos_ = 0;
        lock.unlock();

        // Check if shard changed
        if (shard_eof_) {
            current_shard++;
            if (current_shard < (int)shards.size()) {
                _open_current_shard();
            }
        }

        // Launch next prefetch
        _launch_prefetch();
        return true;
    }
};

// ── Convenience: Create shards from a directory ────────────────
// Usage: auto shards = shards_from_dir("/data/tinystories/", ".txt");
#include <filesystem>
inline std::vector<DataShard> shards_from_dir(
    const std::string& dir, const std::string& ext = ".txt")
{
    std::vector<DataShard> result;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ext) {
            DataShard s;
            s.file_path   = entry.path().string();
            s.byte_offset = 0;
            s.byte_length = -1;
            result.push_back(s);
        }
    }
    std::sort(result.begin(), result.end(),
        [](const DataShard& a, const DataShard& b){
            return a.file_path < b.file_path;
        });
    return result;
}
