#pragma once
// Pre-tokenized dataset: text ek baar tokenize hota hai aur uint16 .bin me save.
// Layout: [magic 'LGTK' (4)] [version u32] [vocab_size u32] [tokenizer fingerprint u64]
//         [src text size i64] [n_tokens i64] [n_tokens x uint16 (little-endian)]
#include "StreamingDataLoader.hpp"   // IBatchLoader + Tokenizer
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr char     TOKBIN_MAGIC[4]      = {'L','G','T','K'};
static constexpr uint32_t TOKBIN_VERSION       = 1;
static constexpr int64_t  TOKBIN_HEADER_BYTES  = 4 + 4 + 4 + 8 + 8 + 8;   // 36

struct TokenBinInfo {
    bool     valid       = false;
    uint32_t vocab_size  = 0;
    uint64_t fingerprint = 0;
    int64_t  src_size    = 0;
    int64_t  n_tokens    = 0;
};

inline TokenBinInfo read_token_bin_header(const std::string& path) {
    TokenBinInfo info;
    std::ifstream f(path, std::ios::binary);
    if (!f) return info;
    char magic[4] = {};
    uint32_t ver = 0, vs = 0;
    uint64_t fp = 0;
    int64_t src = 0, n = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&vs),  4);
    f.read(reinterpret_cast<char*>(&fp),  8);
    f.read(reinterpret_cast<char*>(&src), 8);
    f.read(reinterpret_cast<char*>(&n),   8);
    if (!f || std::memcmp(magic, TOKBIN_MAGIC, 4) != 0 || ver != TOKBIN_VERSION) return info;
    info.valid = true; info.vocab_size = vs; info.fingerprint = fp;
    info.src_size = src; info.n_tokens = n;
    return info;
}

// .bin tab valid hai jab: header ok, same tokenizer (fingerprint), same source size, file size poori.
inline bool token_bin_matches(const std::string& path, const Tokenizer& tok, int64_t src_size) {
    TokenBinInfo info = read_token_bin_header(path);
    if (!info.valid || info.n_tokens <= 0) return false;
    // corrupt header: 2*n_tokens + header int64 me overflow (UB) na kare
    if (info.n_tokens > (INT64_MAX - TOKBIN_HEADER_BYTES) / 2) return false;
    if (info.vocab_size != (uint32_t)tok.vocab_size) return false;
    if (info.fingerprint != tok.fingerprint()) return false;
    if (info.src_size != src_size) return false;
    std::error_code ec;
    auto fsz = std::filesystem::file_size(path, ec);
    if (ec) return false;
    return (int64_t)fsz == TOKBIN_HEADER_BYTES + 2 * info.n_tokens;
}

namespace tokbin_detail {

inline void write_header(std::ofstream& out, const Tokenizer& tok, int64_t src_size, int64_t n_tokens) {
    uint32_t ver = TOKBIN_VERSION, vs = (uint32_t)tok.vocab_size;
    uint64_t fp = tok.fingerprint();
    out.write(TOKBIN_MAGIC, 4);
    out.write(reinterpret_cast<const char*>(&ver), 4);
    out.write(reinterpret_cast<const char*>(&vs),  4);
    out.write(reinterpret_cast<const char*>(&fp),  8);
    out.write(reinterpret_cast<const char*>(&src_size), 8);
    out.write(reinterpret_cast<const char*>(&n_tokens), 8);
}

inline bool is_ws(unsigned char c) { return Tokenizer::is_ws_byte(c); }   // Tokenizer ke saath EXACT same definition (\v,\f included)

// 'want' bytes padho; agar aur data baaki hai to aakhri whitespace-run se pehle kaato
// (kata hua hissa agli call me wapas aayega). Returns bytes in 'out'.
inline size_t read_chunk_ws_aligned(std::ifstream& in, int64_t want, std::string& out) {
    out.assign(static_cast<size_t>(want), '\0');
    in.read(out.data(), want);
    std::streamsize got = in.gcount();
    out.resize(static_cast<size_t>(got > 0 ? got : 0));
    if (got <= 0) return 0;
    if (!in.eof() && got > 1) {
        const std::streamsize lo = (got > 8192) ? got - 8192 : 0;
        std::streamsize k = got - 1;
        while (k >= lo && !is_ws((unsigned char)out[(size_t)k])) --k;
        if (k >= lo) {
            while (k > lo && is_ws((unsigned char)out[(size_t)(k - 1)])) --k;
            if (k > 0 && k < got) {
                in.clear();
                in.seekg(-(std::streamoff)(got - k), std::ios::cur);
                out.resize(static_cast<size_t>(k));
            }
        }
    }
    return out.size();
}

} // namespace tokbin_detail

// Text file -> uint16 .bin (tmp file + rename: adhoora .bin kabhi valid nahi dikhta).
inline bool pretokenize_file(const std::string& text_path, const Tokenizer& tok,
                             const std::string& bin_path,
                             int64_t chunk_bytes = 32LL * 1024 * 1024)
{
    if (tok.vocab_size <= 0 || tok.vocab_size > 65535) {
        std::cerr << "❌ pretokenize: vocab_size=" << tok.vocab_size << " uint16 me fit nahi hota\n";
        return false;
    }
    std::ifstream in(text_path, std::ios::binary);
    if (!in) { std::cerr << "❌ pretokenize: cannot open " << text_path << "\n"; return false; }
    in.seekg(0, std::ios::end);
    const int64_t src_size = (int64_t)in.tellg();
    in.seekg(0, std::ios::beg);

    const std::string tmp = bin_path + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) { std::cerr << "❌ pretokenize: cannot write " << tmp << "\n"; return false; }
    tokbin_detail::write_header(out, tok, src_size, 0);

    int64_t n_tokens = 0, done_bytes = 0, next_report = 0;
    std::string chunk;
    std::vector<uint16_t> buf16;
    while (true) {
        size_t got = tokbin_detail::read_chunk_ws_aligned(in, chunk_bytes, chunk);
        if (got == 0) break;
        std::vector<int> ids = tok.encode_raw(chunk);
        buf16.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) buf16[i] = static_cast<uint16_t>(ids[i]);
        out.write(reinterpret_cast<const char*>(buf16.data()),
                  static_cast<std::streamsize>(buf16.size() * sizeof(uint16_t)));
        if (!out) { std::cerr << "❌ pretokenize: write error (disk full?)\n"; return false; }
        n_tokens   += (int64_t)ids.size();
        done_bytes += (int64_t)got;
        if (done_bytes >= next_report) {
            printf("  [PRETOK] %lld / %lld MB | %lld M tokens\n",
                   (long long)(done_bytes >> 20), (long long)(src_size >> 20),
                   (long long)(n_tokens / 1000000));
            fflush(stdout);
            next_report += 256LL * 1024 * 1024;
        }
    }

    if (n_tokens <= 0) {
        out.close();
        std::error_code ec0; std::filesystem::remove(tmp, ec0);
        std::cerr << "❌ pretokenize: 0 tokens\n";
        return false;
    }
    out.seekp(0, std::ios::beg);
    tokbin_detail::write_header(out, tok, src_size, n_tokens);
    out.close();
    if (!out) { std::cerr << "❌ pretokenize: header write error\n"; return false; }

    std::error_code ec;
    std::filesystem::remove(bin_path, ec);
    ec.clear();
    std::filesystem::rename(tmp, bin_path, ec);
    if (ec) { std::cerr << "❌ pretokenize: rename fail: " << ec.message() << "\n"; return false; }

    printf("  [PRETOK] done: %lld tokens (%.1f MB) -> %s\n",
           (long long)n_tokens, (double)(n_tokens * 2) / 1024.0 / 1024.0, bin_path.c_str());
    fflush(stdout);
    return true;
}

// Pre-tokenized .bin se sequential windows (input = [p, p+seq), target = [p+1, p+seq+1)).
// Range [start_tok, end_tok) tokens me — train/val split token-based.
class TokenBinLoader : public IBatchLoader {
public:
    int     seq_len;
    int     grad_accum_steps;
    int64_t total_tokens_seen = 0;
    int64_t total_steps_done  = 0;
    int     current_epoch     = 0;

    TokenBinLoader(const std::string& bin_path, int seq_len_, int grad_accum_,
                   int64_t start_tok = 0, int64_t end_tok = 0)
        : seq_len(seq_len_), grad_accum_steps(grad_accum_)
    {
        if (seq_len_ <= 0)    throw std::invalid_argument("seq_len must be > 0");
        if (grad_accum_ <= 0) throw std::invalid_argument("grad_accum_steps must be > 0");
        TokenBinInfo info = read_token_bin_header(bin_path);
        if (!info.valid) throw std::runtime_error("Bad/missing token .bin: " + bin_path);
        if (info.n_tokens <= 0 || info.n_tokens > (INT64_MAX - TOKBIN_HEADER_BYTES) / 2)
            throw std::runtime_error("Corrupt token .bin header (n_tokens): " + bin_path);
        {
            std::error_code ec;
            auto fsz = std::filesystem::file_size(bin_path, ec);
            if (ec || (int64_t)fsz < TOKBIN_HEADER_BYTES + 2 * info.n_tokens)
                throw std::runtime_error("Truncated token .bin (file shorter than header says): " + bin_path);
        }
        if (start_tok < 0) start_tok = 0;
        if (end_tok <= 0 || end_tok > info.n_tokens) end_tok = info.n_tokens;
        if (end_tok - start_tok < (int64_t)seq_len_ + 2)
            throw std::invalid_argument("TokenBinLoader: token range too small for seq_len");
        start_ = start_tok;
        range_ = end_tok - start_tok;
        f_.open(bin_path, std::ios::binary);
        if (!f_) throw std::runtime_error("Cannot open token .bin: " + bin_path);
    }

    bool next_batch(std::vector<int>& input_ids, std::vector<int>& target_ids) override {
        if (pos_ + seq_len + 1 > range_) {          // epoch end (StreamingDataLoader jaisa: false + rewind)
            ++current_epoch;
            pos_ = 0;
            return false;
        }
        tmp_.resize(static_cast<size_t>(seq_len) + 1);
        f_.clear();
        f_.seekg(TOKBIN_HEADER_BYTES + 2 * (start_ + pos_), std::ios::beg);
        const std::streamsize bytes = (std::streamsize)(tmp_.size() * sizeof(uint16_t));
        f_.read(reinterpret_cast<char*>(tmp_.data()), bytes);
        if (f_.gcount() != bytes) return false;
        input_ids.resize(static_cast<size_t>(seq_len));
        target_ids.resize(static_cast<size_t>(seq_len));
        for (int i = 0; i < seq_len; ++i) {
            input_ids[(size_t)i]  = tmp_[(size_t)i];
            target_ids[(size_t)i] = tmp_[(size_t)i + 1];
        }
        pos_ += seq_len;
        total_tokens_seen += seq_len;
        return true;
    }

    bool next_accum_batch(
        std::vector<std::pair<std::vector<int>, std::vector<int>>>& micro_batches) override
    {
        micro_batches.clear();
        for (int i = 0; i < grad_accum_steps; ++i) {
            std::vector<int> inp, tgt;
            if (!next_batch(inp, tgt)) return !micro_batches.empty();
            micro_batches.push_back({std::move(inp), std::move(tgt)});
        }
        return true;
    }

    int64_t total_batches_per_epoch() const override { return range_ / seq_len; }

    void reset() override {
        pos_ = 0;
        total_tokens_seen = 0;
        total_steps_done  = 0;
    }

    LoaderState get_state() const override {
        return { 0, pos_, total_tokens_seen, total_steps_done, current_epoch };
    }

    void restore_state(const LoaderState& s) override {
        pos_ = std::max<int64_t>(0, std::min<int64_t>(s.byte_pos, range_));
        total_tokens_seen = s.total_tokens_seen;
        total_steps_done  = s.total_steps_done;
        current_epoch     = s.current_epoch;
    }

private:
    std::ifstream f_;
    int64_t start_ = 0;
    int64_t range_ = 0;
    int64_t pos_   = 0;
    std::vector<uint16_t> tmp_;
};
