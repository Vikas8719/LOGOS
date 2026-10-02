#pragma once
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

// ── [v22-SCALE] Vocab: Fixed 8192 ─────────────────────────────
// Pehle: vocab_sz = ds_size>200MB ? 8192 : ...
//        lekin tokenizer ne sirf 50MB dekha → wrong vocab
// Ab:    vocab FIXED 8192 — 8192-context ke saath consistent
//        Hindi+English 1.5GB ke liye BPE coverage sufficient hai
// [v22-SCALE] Override possible: LOGOS_VOCAB_SIZE env var (future use)
inline int decide_vocab_size(int64_t actual_text_bytes) {
    // [v22-SCALE] Fixed 8192 vocab — matches 219M model's embedding table
    // Chhote test runs ke liye bhi 8192 maintain karo → checkpoint compat
    (void)actual_text_bytes;
    return 8192;
}

// ── [v22-SCALE] 219M Parameter Model — 8192 Context ──────────
// Target: ~219M params, context=8192 tokens
//
// Architecture Math:
//   d=1024, H=16, DH=64, FFN=4096 (4×d), L=16 layers
//
//   Per-layer params:
//     Q,K,V weights:  H × d × DH × 3 = 16×1024×64×3 = 3,145,728
//     O weights:      H × DH × d     = 16×64×1024   = 1,048,576
//     W_proj:         d × d          = 1,048,576
//     W1,b1:          d × 4d + 4d    = 4,194,304 + 4,096
//     W2,b2:          4d × d + d     = 4,194,304 + 1,024
//     LN params (×2): 4d             = 4,096
//     Total/layer:    ≈ 13,640,704   ≈ 13.6M
//
//   16 layers:  16 × 13.6M  = 217.7M
//   Embedding:  8192 × 1024 =   8.4M
//   LM Head:    1024 × 8192 =   8.4M (often weight-tied, counted once)
//   GRAND TOTAL ≈ 219M params ✅
//
// Context=8192 feasibility on Kaggle T4 (16GB VRAM):
//   Weights (fp32):     219M × 4B  ≈  876 MB
//   Nikhilam INT8 KV:   2 × seq × DH × H × L × 1B
//                     = 2 × 8192 × 64 × 16 × 16 × 1B ≈ 268 MB
//   Activations:        seq × d × 4B × ~10 per layer ≈ 85MB/layer
//                       with grad_accum=8 → 8×85MB = 680 MB
//   Optimizer (SHM):    velocities = 219M × 4B ≈ 876 MB
//   TOTAL:              ≈ 876+268+680+876 ≈ 2.7 GB  (fits T4 ✅)
//   grad_accum=8:       effective batch = 8 × 8192 = 65,536 tokens/step
//
// Shunyam sparse attention window=64 stride=16:
//   Per-head complexity: O(seq × window) = O(8192 × 64) vs O(8192²)
//   Memory: 8192 × 64 scores vs 8192² → 128× reduction ✅
//
struct ScaledConfig {
    int d_model, num_heads, num_layers, max_seq_len;
};

inline ScaledConfig decide_model_config(int64_t actual_text_bytes) {
    // [v22-SCALE] FIXED: 219M param model, 8192 context
    // Env override possible in generate_gpu() via LOGOS_D_MODEL etc.
    // This function always returns 219M config — no dataset-size branching.
    (void)actual_text_bytes;

    // d=1024, H=16, L=16, seq=8192  →  ~219M params
    // H=16 divides d=1024 evenly: DH = 64 ✓
    return {1024, 16, 16, 8192};
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

    // ── Constructor: Byte-range of a single file ──────────────
    // [v17] Train/Val split ke liye: sirf [start_byte, end_byte) padhta hai.
    // train_gpu.cu pehle isi 7-arg constructor ko call kar raha tha lekin ye exist hi
    // nahi karta tha → build fail hota. start_byte > 0 ho to agli '\n' tak aage badhte
    // hain taaki UTF-8 character / article beech se na kate (Hindi = 3 bytes/char).
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
            // agar range me newline hi nahi mila to original start rakho
            if (start_byte + skipped < end_byte) start_byte += skipped;
        }

        DataShard shard;
        shard.file_path   = text_file;
        shard.byte_offset = start_byte;
        shard.byte_length = end_byte - start_byte;
        shards.push_back(shard);

        // total_batches_per_epoch() ab sirf is range ke size par based hai
        actual_text_size_ = shard.byte_length;

        _open_current_shard();
        _load_next_chunk_sync();
        _launch_prefetch();
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

    // ── reset(): rewind to the beginning of the byte range ───────
    // train_gpu.cu val_loader.reset() calls ke liye zaroori.
    // Shard(s) ko byte_offset par wapas le jaata hai aur
    // curr_chunk_ / curr_pos_ clear karta hai.
    void reset() {
        // Stop any in-flight prefetch thread first
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

// ── UnifiedDataLoader ─────────────────────────────────────────
// FIX: StreamingDataLoader had a different interface from DataLoader,
// so it was included but never actually used in the training loop.
//
// UnifiedDataLoader wraps BOTH behind ONE interface identical to what
// main.cpp's training loop expects:
//   - next_batch(input_ids, target_ids) → bool
//   - total_batches() → int
//   - current_pos (resettable by caller for new epoch)
//
// Selection policy (matches previous DataLoader heuristic):
//   file_size < 64 MB  → DataLoader   (in-memory, fast random access)
//   file_size >= 64 MB → StreamingDataLoader (chunk streaming, prefetch)
//
// main.cpp training loop uses UnifiedDataLoader; no code change needed
// in the loop itself — only the construction site changes.
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

        // Expose actual text size for vocab/config scaling decisions
        actual_text_size_ = file_bytes;
    }

    bool next_batch(std::vector<int>& input_ids, std::vector<int>& target_ids) {
        bool ok;
        if (use_stream_) {
            ok = stream_->next_batch(input_ids, target_ids);
            // Mirror current_pos so callers that read it get a rough value
            current_pos = (int)(stream_->total_tokens_seen % INT_MAX);
        } else {
            ok = direct_->next_batch(input_ids, target_ids);
            current_pos = direct_->current_pos;
        }
        return ok;
    }

    // Reset for next epoch — mirrors DataLoader contract
    void reset() {
        if (use_stream_) {
            // StreamingDataLoader rewinds on next epoch automatically;
            // after next_batch() returns false, next call restarts.
            // No explicit API needed — just let it return false once.
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
