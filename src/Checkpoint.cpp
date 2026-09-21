// ============================================================
//  LOGOS — Checkpoint.cpp
//  Model Save/Load — Har 1000 steps pe save karo
//  Format: Binary (.bin) — fast, small
// ============================================================
#include "../include/Tensor.hpp"
#include "Model.cpp"
#include <fstream>
#include <iostream>
#include <string>

void save_checkpoint(const LOGOSModel& model, const std::string& path, int step) {
    std::string full_path = path + "_step" + std::to_string(step) + ".bin";
    std::ofstream f(full_path, std::ios::binary);
    if (!f) { std::cerr << "❌ Cannot save checkpoint: " << full_path << "\n"; return; }

    // Write config
    f.write(reinterpret_cast<const char*>(&model.cfg), sizeof(ModelConfig));

    // Write all tensors
    auto write_tensor = [&](const Tensor& t) {
        f.write(reinterpret_cast<const char*>(t.data.data()),
                t.total_size * sizeof(float));
    };

    write_tensor(model.embedding);
    write_tensor(model.pos_embedding);
    write_tensor(model.lm_head);

    for (const auto& block : model.layers) {
        // MHA weights
        for (const auto& head : block.mha.heads) {
            write_tensor(head.W_Q);
            write_tensor(head.W_K);
            write_tensor(head.W_V);
            write_tensor(head.W_O);
        }
        write_tensor(block.mha.W_proj);
        // FFN weights
        write_tensor(block.ffn.W1);
        write_tensor(block.ffn.b1);
        write_tensor(block.ffn.W2);
        write_tensor(block.ffn.b2);
        // LayerNorm
        write_tensor(block.ln1.gamma);
        write_tensor(block.ln1.beta);
        write_tensor(block.ln2.gamma);
        write_tensor(block.ln2.beta);
    }

    std::cout << "✅ Checkpoint saved: " << full_path
              << " (step=" << step << ")\n";
}

bool load_checkpoint(LOGOSModel& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "❌ Cannot load: " << path << "\n"; return false; }

    ModelConfig cfg;
    f.read(reinterpret_cast<char*>(&cfg), sizeof(ModelConfig));

    auto read_tensor = [&](Tensor& t) {
        f.read(reinterpret_cast<char*>(t.data.data()),
               t.total_size * sizeof(float));
    };

    read_tensor(model.embedding);
    read_tensor(model.pos_embedding);
    read_tensor(model.lm_head);

    for (auto& block : model.layers) {
        for (auto& head : block.mha.heads) {
            read_tensor(head.W_Q);
            read_tensor(head.W_K);
            read_tensor(head.W_V);
            read_tensor(head.W_O);
        }
        read_tensor(block.mha.W_proj);
        read_tensor(block.ffn.W1);
        read_tensor(block.ffn.b1);
        read_tensor(block.ffn.W2);
        read_tensor(block.ffn.b2);
        read_tensor(block.ln1.gamma);
        read_tensor(block.ln1.beta);
        read_tensor(block.ln2.gamma);
        read_tensor(block.ln2.beta);
    }

    // Verify load karne ke baad
    if (model.embedding.has_nan()) {
        std::cerr << "❌ Checkpoint corrupt — NaN in embeddings\n";
        return false;
    }
    std::cout << "✅ Checkpoint loaded: " << path << "\n";
    return true;
}
