#pragma once
// ============================================================
//  LOGOS — Evaluate.hpp  v5 — FINAL FIX
//
//  Problem: main.cpp line 615 calls evaluate(model, loader, 50)
//  where loader is UnifiedDataLoader. Previous versions required
//  evaluate_unified() — a different name — which broke when the
//  old main.cpp was cached on macOS/Windows runners.
//
//  Solution: provide BOTH overloads under the SAME name evaluate().
//    Overload A: evaluate(model, DataLoader&,        batches)
//    Overload B: evaluate(model, UnifiedDataLoader&, batches)
//  No templates, no concepts, no std::function tricks.
//  Works on GCC 12, Clang 14, AppleClang 15, MinGW — all C++17.
//
//  UnifiedDataLoader is defined in StreamingDataLoader.hpp.
//  main.cpp includes StreamingDataLoader.hpp BEFORE Evaluate.hpp,
//  so the type is complete here.  We use a forward declaration +
//  out-of-line body pattern: the overload is declared inline and
//  defined right here, but the body only calls methods that exist
//  on both loaders (next_batch + reset / current_pos=0).
// ============================================================
#include "Tensor.hpp"
#include "Model.hpp"
#include "DataLoader.hpp"
#include "StreamingDataLoader.hpp"   // brings in UnifiedDataLoader

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

struct EvalResult {
    float loss;
    float perplexity;
    float top1_accuracy;
    int   total_tokens;
};

// ── Shared implementation ─────────────────────────────────────
// Called by both overloads. Iterates next_batch until done or
// max_batches reached. Loader must already be reset by caller.
namespace logos_eval {
inline EvalResult run(LOGOSModel& model,
                      DataLoader& loader,
                      int max_batches)
{
    float total_loss = 0.f;
    int correct = 0, total = 0, batches = 0;
    std::vector<int> inp, tgt;

    while (batches < max_batches && loader.next_batch(inp, tgt)) {
        Tensor logits = model.forward(inp, false);
        int seq = logits.rows(), vocab = logits.cols();
        for (int i = 0; i < seq && i < (int)tgt.size(); ++i) {
            int t = tgt[i];
            if (t < 0 || t >= vocab) continue;
            float mx = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v) mx = std::max(mx, logits.at(i, v));
            float sm = 0.f;
            for (int v = 0; v < vocab; ++v) sm += std::exp(logits.at(i, v) - mx);
            total_loss += -(logits.at(i, t) - mx - std::log(sm + 1e-9f));
            int pred = 0; float best = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v)
                if (logits.at(i, v) > best) { best = logits.at(i, v); pred = v; }
            if (pred == t) ++correct;
            ++total;
        }
        ++batches;
    }
    EvalResult r;
    r.loss          = total > 0 ? total_loss / (float)total : 999.f;
    r.perplexity    = std::exp(std::min(r.loss, 20.f));
    r.top1_accuracy = total > 0 ? 100.f * (float)correct / (float)total : 0.f;
    r.total_tokens  = total;
    return r;
}
} // namespace logos_eval

// ── Overload A: DataLoader ────────────────────────────────────
inline EvalResult evaluate(LOGOSModel& model,
                           DataLoader& loader,
                           int max_batches = 50)
{
    loader.current_pos = 0;
    return logos_eval::run(model, loader, max_batches);
}

// ── Overload B: UnifiedDataLoader ────────────────────────────
// UnifiedDataLoader wraps DataLoader + StreamingDataLoader.
// It exposes .reset() and .next_batch() — but next_batch() is
// defined on the outer UnifiedDataLoader, not on DataLoader.
// So we implement this overload separately with its own loop.
inline EvalResult evaluate(LOGOSModel& model,
                           UnifiedDataLoader& loader,
                           int max_batches = 50)
{
    loader.reset();
    float total_loss = 0.f;
    int correct = 0, total = 0, batches = 0;
    std::vector<int> inp, tgt;

    while (batches < max_batches && loader.next_batch(inp, tgt)) {
        Tensor logits = model.forward(inp, false);
        int seq = logits.rows(), vocab = logits.cols();
        for (int i = 0; i < seq && i < (int)tgt.size(); ++i) {
            int t = tgt[i];
            if (t < 0 || t >= vocab) continue;
            float mx = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v) mx = std::max(mx, logits.at(i, v));
            float sm = 0.f;
            for (int v = 0; v < vocab; ++v) sm += std::exp(logits.at(i, v) - mx);
            total_loss += -(logits.at(i, t) - mx - std::log(sm + 1e-9f));
            int pred = 0; float best = logits.at(i, 0);
            for (int v = 1; v < vocab; ++v)
                if (logits.at(i, v) > best) { best = logits.at(i, v); pred = v; }
            if (pred == t) ++correct;
            ++total;
        }
        ++batches;
    }
    EvalResult r;
    r.loss          = total > 0 ? total_loss / (float)total : 999.f;
    r.perplexity    = std::exp(std::min(r.loss, 20.f));
    r.top1_accuracy = total > 0 ? 100.f * (float)correct / (float)total : 0.f;
    r.total_tokens  = total;
    return r;
}

// ── evaluate_unified: alias kept for backward compat ─────────
// (old main.cpp versions that called evaluate_unified still work)
inline EvalResult evaluate_unified(LOGOSModel& model,
                                   UnifiedDataLoader& loader,
                                   int max_batches = 50)
{
    return evaluate(model, loader, max_batches);
}

inline void print_eval(const EvalResult& r, int step) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Evaluation";
    if (step >= 0) std::cout << " @ step " << step;
    std::cout << " ===\n"
              << "  Loss       : " << r.loss           << "\n"
              << "  Perplexity : " << r.perplexity     << "\n"
              << "  Top-1 Acc  : " << r.top1_accuracy  << "%\n"
              << "  Tokens     : " << r.total_tokens   << "\n\n";
}
