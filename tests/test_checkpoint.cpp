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

    // expected_bytes must match save_checkpoint layout exactly:
    //   [ModelConfig]
    //   [embedding floats]
    //   [legacy position block: max_seq_len * d_model floats]
    //   [lm_head floats]
    //   per layer:
    //     [per-head W_Q, W_K, W_V, W_O]
    //     [W_proj]
    //     [ffn.W1, ffn.b1, ffn.W2, ffn.b2]
    //     [ln1.gamma, ln1.beta]
    //     [ln2.gamma, ln2.beta]
    //     [ln1.running_mean, ln1.running_var]   ← ReynoldsBatchNorm stats
    //     [ln2.running_mean, ln2.running_var]   ← ReynoldsBatchNorm stats
    size_t expected_bytes = sizeof(ModelConfig) +
        static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    for (const Tensor* parameter : source.parameters())
        expected_bytes += static_cast<size_t>(parameter->total_size) * sizeof(float);
    // Add ReynoldsBatchNorm running_mean + running_var for each layer (ln1 + ln2)
    // Each running stat vector has d_model floats
    expected_bytes += static_cast<size_t>(cfg.num_layers) * 4 *
                      static_cast<size_t>(cfg.d_model) * sizeof(float);

    if (std::filesystem::file_size(file) != expected_bytes) {
        std::remove(file.c_str());
        std::cerr << "checkpoint size mismatch: got "
                  << std::filesystem::file_size(file)
                  << " expected " << expected_bytes << "\n";
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
