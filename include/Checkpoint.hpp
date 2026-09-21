#pragma once
// ============================================================
//  LOGOS — Checkpoint.hpp
// ============================================================
#include "Model.hpp"
#include <fstream>
#include <iostream>
#include <string>

inline void save_checkpoint(const LOGOSModel& model,
                            const std::string& path, int step) {
    std::string full = path + "_step" + std::to_string(step) + ".bin";
    std::ofstream f(full, std::ios::binary);
    if (!f) { std::cerr << "Cannot save: " << full << "\n"; return; }
    f.write(reinterpret_cast<const char*>(&model.cfg), sizeof(ModelConfig));
    auto wt = [&](const Tensor& t){
        f.write(reinterpret_cast<const char*>(t.data.data()),
                t.total_size * sizeof(float));
    };
    wt(model.embedding); wt(model.pos_embedding); wt(model.lm_head);
    for (const auto& block : model.layers) {
        for (const auto& h : block.mha.heads){
            wt(h.W_Q); wt(h.W_K); wt(h.W_V); wt(h.W_O);
        }
        wt(block.mha.W_proj);
        wt(block.ffn.W1); wt(block.ffn.b1);
        wt(block.ffn.W2); wt(block.ffn.b2);
        wt(block.ln1.gamma); wt(block.ln1.beta);
        wt(block.ln2.gamma); wt(block.ln2.beta);
    }
    std::cout << "Checkpoint saved: " << full << "\n";
}

inline bool load_checkpoint(LOGOSModel& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot load: " << path << "\n"; return false; }
    ModelConfig cfg;
    f.read(reinterpret_cast<char*>(&cfg), sizeof(ModelConfig));
    auto rt = [&](Tensor& t){
        f.read(reinterpret_cast<char*>(t.data.data()),
               t.total_size * sizeof(float));
    };
    rt(model.embedding); rt(model.pos_embedding); rt(model.lm_head);
    for (auto& block : model.layers) {
        for (auto& h : block.mha.heads){
            rt(h.W_Q); rt(h.W_K); rt(h.W_V); rt(h.W_O);
        }
        rt(block.mha.W_proj);
        rt(block.ffn.W1); rt(block.ffn.b1);
        rt(block.ffn.W2); rt(block.ffn.b2);
        rt(block.ln1.gamma); rt(block.ln1.beta);
        rt(block.ln2.gamma); rt(block.ln2.beta);
    }
    if (model.embedding.has_nan()){
        std::cerr << "Checkpoint corrupt\n"; return false;
    }
    std::cout << "Checkpoint loaded: " << path << "\n";
    return true;
}
