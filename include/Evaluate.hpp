#pragma once
// ============================================================
//  LOGOS — Evaluate.hpp
// ============================================================
#include "Tensor.hpp"
#include "Model.hpp"
#include "DataLoader.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

struct EvalResult {
    float loss, perplexity, top1_accuracy;
    int total_tokens;
};

inline EvalResult evaluate(LOGOSModel& model, DataLoader& loader,
                           int max_batches = 50) {
    float total_loss = 0.0f;
    int correct = 0, total = 0, batches = 0;
    loader.current_pos = 0;
    std::vector<int> input_ids, target_ids;

    while (loader.next_batch(input_ids, target_ids) && batches < max_batches) {
        // [WIRED] is_training=false → deterministic eval (no FeynmanDropout
        // noise, ReynoldsBatchNorm uses running stats instead of updating them)
        Tensor logits = model.forward(input_ids, /*is_training=*/false);
        int seq = logits.rows(), vocab = logits.cols();
        for (int i = 0; i < seq && i < (int)target_ids.size(); ++i) {
            int target = target_ids[i];
            if (target < 0 || target >= vocab) continue;
            float max_l = logits.at(i,0);
            for (int v=1;v<vocab;++v) max_l = std::max(max_l, logits.at(i,v));
            float sum=0;
            for (int v=0;v<vocab;++v) sum += std::exp(logits.at(i,v)-max_l);
            total_loss += -(logits.at(i,target) - max_l - std::log(sum));
            int pred=0; float best=logits.at(i,0);
            for (int v=1;v<vocab;++v) if(logits.at(i,v)>best){best=logits.at(i,v);pred=v;}
            if (pred==target) ++correct;
            ++total;
        }
        ++batches;
    }

    float avg_loss   = total>0 ? total_loss/total : 999.f;
    float perplexity = std::exp(avg_loss);
    float accuracy   = total>0 ? 100.f*correct/total : 0.f;
    return {avg_loss, perplexity, accuracy, total};
}

inline void print_eval(const EvalResult& r, int step) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Evaluation";
    if (step >= 0) std::cout << " @ step " << step;
    std::cout << " ===\n";
    std::cout << "  Loss       : " << r.loss        << "\n";
    std::cout << "  Perplexity : " << r.perplexity  << "\n";
    std::cout << "  Top-1 Acc  : " << r.top1_accuracy << "%\n";
    std::cout << "  Tokens     : " << r.total_tokens << "\n\n";
}
