#include "Checkpoint.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 8;
    cfg.d_model = 4;
    cfg.num_heads = 2;
    cfg.num_layers = 1;
    cfg.max_seq_len = 3;

    LOGOSModel source(cfg);
    const std::string base = "logos_checkpoint_roundtrip_test";
    const std::string file = base + "_step7.bin";
    std::remove(file.c_str());

    if (!save_checkpoint(source, base, 7)) {
        std::cerr << "checkpoint save failed\n";
        return 1;
    }

    size_t expected_bytes = sizeof(ModelConfig) +
        static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    for (const Tensor* parameter : source.parameters())
        expected_bytes += static_cast<size_t>(parameter->total_size) * sizeof(float);
    if (std::filesystem::file_size(file) != expected_bytes) {
        std::remove(file.c_str());
        std::cerr << "checkpoint no longer reserves the legacy position block\n";
        return 1;
    }

    // Simulate legacy learned-position weights; loaders must discard them and
    // continue reading lm_head and layer weights from their original offsets.
    {
        std::fstream legacy(file, std::ios::binary | std::ios::in | std::ios::out);
        const auto position_offset = static_cast<std::streamoff>(sizeof(ModelConfig)) +
            static_cast<std::streamoff>(source.embedding.total_size * sizeof(float));
        legacy.seekp(position_offset);
        std::vector<float> old_positions(
            static_cast<size_t>(cfg.max_seq_len) * cfg.d_model, 0.25f);
        legacy.write(reinterpret_cast<const char*>(old_positions.data()),
                     static_cast<std::streamsize>(old_positions.size() * sizeof(float)));
        if (!legacy) {
            std::remove(file.c_str());
            std::cerr << "could not prepare legacy checkpoint fixture\n";
            return 1;
        }
    }

    LOGOSModel restored(cfg);
    if (!load_checkpoint(restored, file)) {
        std::remove(file.c_str());
        std::cerr << "checkpoint load failed\n";
        return 1;
    }

    const auto expected = source.parameters();
    const auto actual = restored.parameters();
    bool equal = expected.size() == actual.size();
    for (size_t i = 0; equal && i < expected.size(); ++i) {
        equal = expected[i]->shape == actual[i]->shape &&
                expected[i]->data == actual[i]->data;
    }
    std::remove(file.c_str());
    if (!equal) {
        std::cerr << "checkpoint round-trip changed model parameters\n";
        return 1;
    }

    std::cout << "checkpoint round-trip passed\n";
    return 0;
}
