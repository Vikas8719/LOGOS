#pragma once
// Shared helpers for the libFuzzer targets (clang -fsanitize=fuzzer,address,undefined).
// A violated invariant calls abort() -> libFuzzer saves the crashing input as an artifact.
#include "test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <string>

namespace fz {

// Tokenizer::load / pretokenize_file / load_checkpoint print a lot: send cout/cerr to a null sink once.
struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
    std::streamsize xsputn(const char*, std::streamsize n) override { return n; }
};
inline void silence_streams() {
    static NullBuf nb;
    std::cout.rdbuf(&nb);
    std::cerr.rdbuf(&nb);
}

#define FZ_ASSERT(cond)                                                                   \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "FUZZ INVARIANT VIOLATED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::abort();                                                                 \
        }                                                                                 \
    } while (0)

// Structure-aware mutation of a known-good file: the fuzz input is read as
//   [n_edits (1 byte)] [n_edits x (offset u16 LE, value u8)] [tail bytes]
// edits are applied to a copy of `good`; the tail is appended (or, if the first edit byte is
// odd, the file is truncated to `tail.size()` bytes first). Gives libFuzzer cheap access to
// "valid header with one corrupted field" inputs instead of having to invent a valid file.
inline std::string mutate_good(const std::string& good, const uint8_t* data, size_t size) {
    std::string d = good;
    if (size == 0 || d.empty()) return d;
    size_t p = 0;
    const unsigned edits = data[p++];
    for (unsigned i = 0; i < edits && p + 3 <= size; ++i) {
        const size_t off = (static_cast<size_t>(data[p]) | (static_cast<size_t>(data[p + 1]) << 8)) % d.size();
        d[off] = static_cast<char>(data[p + 2]);
        p += 3;
    }
    if (p < size) {
        const size_t tail = size - p;
        if (edits & 1u) d.resize(tail % (d.size() + 1));
        else            d.append(reinterpret_cast<const char*>(data + p), tail);
    }
    return d;
}

} // namespace fz
