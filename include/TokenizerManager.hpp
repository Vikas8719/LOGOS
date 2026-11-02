#pragma once

#include "Tokenizer.hpp"

#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <vector>
#include <filesystem>

static inline bool _is_safe_path(const std::string& p) {
    if (p.empty()) return false;
    std::filesystem::path fp(p);
    std::string norm = fp.lexically_normal().string();
    if (norm.find("..") != std::string::npos) return false;
    if (norm.find('\0') != std::string::npos) return false;
    if (fp.extension() != ".bin") return false;
    return true;
}

static inline bool _tok_file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static inline int64_t _tok_file_size(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) return 0;
    return (int64_t)f.tellg();
}

class TokenizerManager {
public:

    static Tokenizer load_or_build(
        const std::string& dataset_path  = "",
        int                target_vocab  = 8192,
        bool               force_rebuild = false,
        const std::string& save_path     = "")
    {
        if (!force_rebuild) {
            if (_cache_valid()) {
                std::cout << "Tokenizer [process cache hit] vocab=" << _cached_tok.vocab_size << "\n";
                return _cached_tok;
            }
        } else {
            _cached_path.clear();
            std::cout << "force_rebuild=true — cache aur saved vocab dono ignore\n";
        }

        std::string found_path = "";
        if (!force_rebuild) {
            found_path = _find_existing_vocab();
        }

        if (!found_path.empty()) {
            Tokenizer tok;
            if (tok.load(found_path)) {
                std::cout << "Tokenizer loaded: " << found_path << "\n";
                std::cout << "   vocab=" << tok.vocab_size
                          << "  merges=" << tok.merges.size() << "\n";
                _cached_tok  = tok;
                _cached_path = found_path;
                return tok;
            }
            std::cerr << found_path << " corrupt hai — rebuild karta hun\n";
        }

        if (dataset_path.empty()) {
            throw std::runtime_error(
                "\nvocab.bin nahi mila aur dataset_path bhi nahi diya.\n"
                "   Kya karo:\n"
                "     Option A) Pehle ek baar build karo:\n"
                "               logos --train dataset.txt\n"
                "               -> vocab.bin output mein milega, download karo\n"
                "     Option B) Kaggle Dataset mein vocab.bin upload karo\n"
                "               (logos-ckpt dataset mein)\n");
        }

        std::cout << "\nvocab.bin nahi mila — pehli baar build ho raha hai...\n";
        std::cout << "   Dataset : " << dataset_path << "\n";

        std::ifstream probe(dataset_path, std::ios::ate | std::ios::binary);
        if (!probe) {
            throw std::runtime_error("Dataset file nahi mili: " + dataset_path);
        }
        int64_t ds_bytes = (int64_t)probe.tellg();
        probe.close();
        std::cout << "   Size    : " << ds_bytes / 1024 / 1024 << " MB\n";

        if (target_vocab <= 0) {
            target_vocab = 8192;
            if (ds_bytes < 10LL * 1024 * 1024)        target_vocab = 2048;
            else if (ds_bytes < 100LL * 1024 * 1024)  target_vocab = 4096;
            std::cout << "   Vocab   : " << target_vocab << " (auto)\n";
        } else {
            std::cout << "   Vocab   : " << target_vocab << "\n";
        }
        std::cout << "\n";

        std::ifstream f(dataset_path);
        if (!f) {
            throw std::runtime_error("Dataset file nahi mili: " + dataset_path);
        }
        std::string text;
        {
            constexpr size_t BUF = 64 * 1024 * 1024;
            text.reserve(std::min((size_t)ds_bytes, BUF));
            std::string chunk(BUF, '\0');
            while (f.read(&chunk[0], BUF) || f.gcount() > 0)
                text.append(chunk.data(), (size_t)f.gcount());
        }

        Tokenizer tok;
        tok.build(text, target_vocab);
        text.clear();
        text.shrink_to_fit();

        std::string out_path = save_path.empty() ? _default_save_path() : save_path;
        if (tok.save(out_path)) {
            std::cout << "\nvocab.bin saved: " << out_path << "\n";
            std::cout << "   vocab=" << tok.vocab_size
                      << "  merges=" << tok.merges.size() << "\n\n";
            std::cout << "------------------------------------------\n";
            std::cout << "  NEXT STEP: vocab.bin download karo\n";
            std::cout << "  Kaggle Output tab -> vocab.bin -> Download\n";
            std::cout << "  Phir logos-ckpt Dataset mein upload karo\n";
            std::cout << "  Agli baar rebuild nahi hoga\n";
            std::cout << "------------------------------------------\n\n";

            _cached_tok = tok;
            _cached_path = out_path;
        } else {
            std::cerr << "vocab.bin save fail: " << out_path << "\n";
        }

        return tok;
    }

    static Tokenizer load() {
        if (_cache_valid()) {
            std::cout << "Tokenizer [process cache hit] vocab=" << _cached_tok.vocab_size << "\n";
            return _cached_tok;
        }
        std::string found = _find_existing_vocab();
        if (found.empty()) {
            throw std::runtime_error(
                "\nvocab.bin kahi nahi mili.\n"
                "   Pehle training run karo ya logos-ckpt dataset mein upload karo.\n");
        }
        Tokenizer tok;
        if (!tok.load(found)) {
            throw std::runtime_error("vocab.bin load fail: " + found);
        }
        std::cout << "Tokenizer loaded: " << found
                  << "  vocab=" << tok.vocab_size << "\n";
        _cached_tok  = tok;
        _cached_path = found;
        return tok;
    }

    static void print_status() {
        std::cout << "\n-- TokenizerManager Status --\n";
        std::string found = _find_existing_vocab();
        if (!found.empty()) {
            int64_t sz = _tok_file_size(found);
            std::cout << "  vocab.bin found : " << found
                      << "  (" << sz / 1024 << " KB)\n";
        } else {
            std::cout << "  vocab.bin nahi mili\n";
            std::cout << "     Search kiya   :\n";
            for (const auto& p : _search_paths()) {
                std::cout << "       " << p << "\n";
            }
            std::cout << "     Kya karo      : training run karo -> Output download -> upload\n";
        }
        std::cout << "-----------------------------\n\n";
    }

private:

    inline static Tokenizer    _cached_tok;
    inline static std::string  _cached_path;

    static bool _cache_valid() {
        return _cached_tok.vocab_size > 0 && !_cached_path.empty();
    }

    static std::string _find_existing_vocab() {
        for (const auto& path : _search_paths()) {
            if (!_is_safe_path(path)) continue;
            if (_tok_file_exists(path)) {
                return path;
            }
        }
        return "";
    }

    static std::vector<std::string> _search_paths() {
        std::vector<std::string> paths;

        const char* env_vocab = std::getenv("LOGOS_VOCAB");
        if (env_vocab && strlen(env_vocab) > 0) {
            std::string ep(env_vocab);
            if (_is_safe_path(ep))
                paths.push_back(ep);
            else
                std::cerr << "[TokenizerManager] LOGOS_VOCAB path rejected (unsafe): "
                          << ep << "\n";
        }

        paths.push_back("/kaggle/input/logos-ckpt/vocab.bin");
        paths.push_back("/kaggle/working/vocab.bin");
        paths.push_back("vocab.bin");

        return paths;
    }

    static std::string _default_save_path() {
        if (_tok_file_exists("/kaggle/working") ||
            std::getenv("KAGGLE_KERNEL_RUN_TYPE") != nullptr) {
            return "/kaggle/working/vocab.bin";
        }
        return "vocab.bin";
    }
};
