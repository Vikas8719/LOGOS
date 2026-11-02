#pragma once
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

struct DataShard {
    std::string file_path;
    int64_t byte_offset = 0;    // kahan se padho
    int64_t byte_length = -1;   // -1 = end of file
};

inline int64_t scan_dataset_size(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) return 0;
    return (int64_t)f.tellg();
}

inline int decide_vocab_size(int64_t actual_text_bytes) {
    (void)actual_text_bytes;
    return 8192;
}

struct ScaledConfig {
    int d_model, num_heads, num_layers, max_seq_len;
};

inline ScaledConfig decide_model_config(int64_t actual_text_bytes) {
    (void)actual_text_bytes;

    return {1024, 16, 16, 8192};
}

class StreamingDataLoader {
public:
    int64_t total_tokens_seen  = 0;
    int64_t total_steps_done   = 0;
    int     seq_len;
    int     grad_accum_steps;   // BUG 6: gradient accumulation support
    int     current_epoch      = 0;

    std::vector<DataShard> shards;
    int     current_shard      = 0;

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

        actual_text_size_ = file_size;

        _open_current_shard();
        _load_next_chunk_sync();  // First chunk synchronous
        _launch_prefetch();       // Next chunk prefetch start
    }

    StreamingDataLoader(const std::string& text_file,
                        Tokenizer& tok,
                        int seq_len_,
                        int grad_accum_,
                        int64_t chunk_bytes,
                        int64_t start_byte,
                        int64_t end_byte)
        : seq_len(seq_len_)
        , grad_accum_steps(grad_accum_)
        , tokenizer_ref_(&tok)
        , chunk_bytes_(chunk_bytes)
    {
        if (seq_len_ <= 0)
            throw std::invalid_argument("seq_len must be > 0");
        if (grad_accum_ <= 0)
            throw std::invalid_argument("grad_accum_steps must be > 0");

        int64_t file_size = scan_dataset_size(text_file);
        if (file_size == 0)
            throw std::runtime_error("Dataset empty or not found: " + text_file);

        if (start_byte < 0) start_byte = 0;
        if (end_byte <= 0 || end_byte > file_size) end_byte = file_size;
        if (start_byte >= end_byte)
            throw std::invalid_argument("StreamingDataLoader: empty byte range");

        if (start_byte > 0) {
            std::ifstream probe(text_file, std::ios::binary);
            probe.seekg(start_byte, std::ios::beg);
            char c = 0;
            int64_t skipped = 0;
            while (start_byte + skipped < end_byte && probe.get(c)) {
                ++skipped;
                if (c == '\n') break;
            }
            if (start_byte + skipped < end_byte) start_byte += skipped;
        }

        DataShard shard;
        shard.file_path   = text_file;
        shard.byte_offset = start_byte;
        shard.byte_length = end_byte - start_byte;
        shards.push_back(shard);

        actual_text_size_ = shard.byte_length;

        _open_current_shard();
        _load_next_chunk_sync();
        _launch_prefetch();
    }

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

    int64_t actual_text_size() const { return actual_text_size_; }

    bool next_batch(std::vector<int>& input_ids,
                    std::vector<int>& target_ids)
    {
        while (curr_pos_ + seq_len + 1 >= (int)curr_chunk_.size()) {
            bool got = _swap_prefetched_chunk();
            if (!got) {
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

    bool next_accum_batch(
        std::vector<std::pair<std::vector<int>, std::vector<int>>>& micro_batches)
    {
        micro_batches.clear();
        for (int i = 0; i < grad_accum_steps; ++i) {
            std::vector<int> inp, tgt;
            bool ok = next_batch(inp, tgt);
            if (!ok) {
                return !micro_batches.empty();
            }
            micro_batches.push_back({std::move(inp), std::move(tgt)});
        }
        return true;
    }

    int64_t total_batches_per_epoch() const {
        double tokens_per_byte = 0.65;
        int64_t est_tokens = (int64_t)(actual_text_size_ * tokens_per_byte);
        return est_tokens / seq_len;
    }

    void reset() {
        {
            std::lock_guard<std::mutex> lk(prefetch_mutex_);
            prefetch_stop_ = false;   // we are NOT destroying, just rewinding
            prefetch_ready_ = false;
        }
        if (prefetch_thread_.joinable()) prefetch_thread_.join();

        current_shard   = 0;
        total_tokens_seen = 0;
        total_steps_done  = 0;
        _open_current_shard();        // reseek to byte_offset of shard 0
        _load_next_chunk_sync();      // first chunk synchronous
        _launch_prefetch();           // background prefetch restart
    }

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

    std::vector<int>  curr_chunk_;
    int               curr_pos_ = 0;

    std::vector<int>  prefetch_chunk_;
    bool              prefetch_ready_ = false;
    bool              prefetch_stop_  = false;
    std::atomic<bool> shard_eof_{false};
    std::atomic<int>  current_shard_atomic_{0};
    std::thread             prefetch_thread_;
    std::mutex              prefetch_mutex_;
    std::condition_variable prefetch_cv_;

    std::ifstream  stream_;
    int64_t        shard_byte_pos_ = 0;
    std::mutex     stream_mutex_;  

    void _open_current_shard() {
        std::lock_guard<std::mutex> slk(stream_mutex_);
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

    std::vector<int> _read_and_tokenize_chunk() {
        if (shard_eof_) return {};
        std::lock_guard<std::mutex> slk(stream_mutex_);
        if (!stream_) return {};

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

        std::string text(buf.data(), static_cast<size_t>(got));
        int64_t max_tokens = static_cast<int64_t>(got) * 4;
        int capped = (max_tokens > INT_MAX) ? INT_MAX : static_cast<int>(max_tokens);
        return tokenizer_ref_->encode(text, capped);
    }

    void _load_next_chunk_sync() {
        auto new_tokens = _read_and_tokenize_chunk();

        if (new_tokens.empty() && shard_eof_) {
            current_shard++;
            if (current_shard < (int)shards.size()) {
                _open_current_shard();
                new_tokens = _read_and_tokenize_chunk();
            }
        }

        std::vector<int> tail;
        if (curr_chunk_.size() > static_cast<size_t>(seq_len + 1)) {
            tail.assign(curr_chunk_.end() - seq_len - 1, curr_chunk_.end());
        }
        curr_chunk_ = std::move(tail);
        curr_chunk_.insert(curr_chunk_.end(), new_tokens.begin(), new_tokens.end());
        curr_pos_ = 0;
    }

    void _launch_prefetch() {
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_ready_ = false;
        }
        if (prefetch_thread_.joinable()) prefetch_thread_.join();
        if (prefetch_stop_) return;

        prefetch_thread_ = std::thread([this]() {
            try {
                auto tokens = _read_and_tokenize_chunk();

                if (tokens.empty() && shard_eof_) {
                    int next_shard = current_shard + 1;
                    if (next_shard < (int)shards.size()) {
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

            } catch (const std::exception& ex) {
                fprintf(stderr,
                    "\n[PREFETCH WARN] Exception in prefetch thread: %s\n"
                    "  Training will stall at epoch boundary (data not lost).\n"
                    "  Check dataset file integrity if this repeats.\n",
                    ex.what());
                std::lock_guard<std::mutex> lock(prefetch_mutex_);
                prefetch_chunk_.clear();   // empty → swap_prefetched_chunk returns false
                prefetch_ready_ = true;    // unblock waiter so training doesn't hang
                prefetch_cv_.notify_one();
            } catch (...) {
                fprintf(stderr,
                    "\n[PREFETCH WARN] Unknown exception in prefetch thread.\n"
                    "  Training will stall at epoch boundary.\n");
                std::lock_guard<std::mutex> lock(prefetch_mutex_);
                prefetch_chunk_.clear();
                prefetch_ready_ = true;
                prefetch_cv_.notify_one();
            }
        });
    }

    bool _swap_prefetched_chunk() {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this]{ return prefetch_ready_ || prefetch_stop_; });

        if (prefetch_chunk_.empty()) return false;

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

        if (shard_eof_) {
            current_shard++;
            if (current_shard < (int)shards.size()) {
                _open_current_shard();
            }
        }

        _launch_prefetch();
        return true;
    }
};

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

#include "DataLoader.hpp"

#include "DataLoader.hpp"

class UnifiedDataLoader {
public:
    int current_pos = 0;   // exposed for epoch-reset compat with main.cpp
    int seq_len;

    explicit UnifiedDataLoader(const std::string& file,
                               Tokenizer& tok,
                               int seq_len_    = 128,
                               int grad_accum  = 1)
        : seq_len(seq_len_)
    {
        int64_t file_bytes = scan_dataset_size(file);
        constexpr int64_t STREAM_THRESHOLD = 64LL * 1024 * 1024;  // 64 MB

        if (file_bytes >= STREAM_THRESHOLD) {
            std::cout << "UnifiedDataLoader: " << file_bytes/1024/1024
                      << " MB → StreamingDataLoader (prefetch, chunk-mode)\n";
            stream_.reset(new StreamingDataLoader(file, tok, seq_len_, grad_accum));
            use_stream_ = true;
        } else {
            std::cout << "UnifiedDataLoader: " << file_bytes/1024
                      << " KB → DataLoader (in-memory)\n";
            direct_.reset(new DataLoader(file, tok, seq_len_));
            use_stream_ = false;
        }

        actual_text_size_ = file_bytes;
    }

    bool next_batch(std::vector<int>& input_ids, std::vector<int>& target_ids) {
        bool ok;
        if (use_stream_) {
            ok = stream_->next_batch(input_ids, target_ids);
            current_pos = (int)(stream_->total_tokens_seen % INT_MAX);
        } else {
            ok = direct_->next_batch(input_ids, target_ids);
            current_pos = direct_->current_pos;
        }
        return ok;
    }

    void reset() {
        if (use_stream_) {
            stream_->reset();
            current_pos = 0;
        } else {
            direct_->current_pos = 0;
            current_pos          = 0;
        }
    }

    int total_batches() const {
        if (use_stream_)
            return (int)std::min(stream_->total_batches_per_epoch(),
                                 (int64_t)INT_MAX);
        return direct_->total_batches();
    }

    int64_t actual_text_size() const { return actual_text_size_; }

private:
    bool use_stream_ = false;
    int64_t actual_text_size_ = 0;
    std::unique_ptr<StreamingDataLoader> stream_;
    std::unique_ptr<DataLoader>          direct_;
};
