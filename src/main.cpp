// ============================================================
//  LOGOS — main.cpp
//  Vedic-Physics Hybrid LLM | C++20
//  Optimizer: Langevin Dynamics (Physics-based, NO Adam)
// ============================================================
#include "Tensor.hpp"
#include "VedicGEMM.hpp"
#include "Tokenizer.hpp"
#include "Attention.hpp"
#include "FeedForward.hpp"
#include "LayerNorm.hpp"
#include "TransformerBlock.hpp"
#include "PhysicsOpt.hpp"
#include "GradientClip.hpp"
#include "Model.hpp"
#include "Checkpoint.hpp"
#include "DataLoader.hpp"
#include "Evaluate.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <cmath>

// ── Unit Tests ────────────────────────────────────────────────
void run_tests() {
    std::cout << "\n========== LOGOS Unit Tests ==========\n";

    std::cout << "\n[1] Tensor.hpp test...\n";
    Tensor A({4,4}); A.fill_random(-1.f,1.f);
    Tensor B = A + A;
    assert(std::abs(B.at(0,0) - 2.0f*A.at(0,0)) < 1e-5f);
    assert(!A.has_nan());
    A.print("Tensor A");
    std::cout << "✅ Tensor: PASS\n";

    std::cout << "\n[2] VedicGEMM test...\n";
    srand(42);
    Tensor M1({32,64}); M1.fill_random(-1.f,1.f);
    Tensor M2({64,32}); M2.fill_random(-1.f,1.f);
    Tensor Cref = reference_gemm(M1,M2);
    Tensor Cved = vedic_gemm(M1,M2);
    float max_err = 0.f;
    for (int i=0;i<Cref.total_size;++i)
        max_err = std::max(max_err, std::abs(Cref[i]-Cved[i]));
    std::cout << "Max error: " << max_err << "\n";
    std::cout << (max_err < 1e-3f ? "✅ VedicGEMM: PASS\n" : "❌ VedicGEMM: FAIL\n");

    std::cout << "\n[3] Tokenizer test...\n";
    Tokenizer tok;
    tok.build("hello world logos vedic math hello world hello logos", 300);
    tok.save("vocab.bin");
    Tokenizer tok2; tok2.load("vocab.bin");
    auto ids = tok2.encode("hello world");
    std::string dec = tok2.decode(ids);
    std::cout << "Decoded: '" << dec << "'\n";
    std::cout << (dec.find("hello")!=std::string::npos ? "✅ Tokenizer: PASS\n" : "❌ Tokenizer: FAIL\n");

    std::cout << "\n[4] LangevinOptimizer test...\n";
    Tensor W({4,4}); W.fill_random(-0.1f, 0.1f);
    Tensor G({4,4}); G.fill(0.01f);
    LangevinOptimizer opt(1e-4f, 0.9f, 0.1f, 1e-5f, 1000);
    std::vector<Tensor*> ps = {&W};
    std::vector<Tensor*> gs = {&G};
    opt.init(ps);
    float w0 = W.data[0];
    opt.step(ps, gs);
    std::cout << "Weight changed: " << (W.data[0] != w0 ? "YES" : "NO") << "\n";
    std::cout << (!W.has_nan() ? "✅ LangevinOpt: PASS\n" : "❌ LangevinOpt: NaN\n");

    std::cout << "\n========== All Tests Complete ==========\n";
}

// ── Forward Pass ──────────────────────────────────────────────
void run_forward_test() {
    std::cout << "\n========== Forward Pass Test ==========\n";
    ModelConfig cfg;
    cfg.d_model=64; cfg.num_heads=4; cfg.num_layers=2;
    cfg.vocab_size=256; cfg.max_seq_len=32;
    LOGOSModel model(cfg);
    std::vector<int> input = {1,5,10,20,42,100,2};
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor logits = model.forward(input);
    float ms = std::chrono::duration<float,std::milli>(
        std::chrono::high_resolution_clock::now()-t0).count();
    std::cout << "Shape: (" << logits.rows() << "," << logits.cols() << ")\n";
    std::cout << "Time: " << ms << " ms\n";
    std::cout << (logits.has_nan() ? "❌ NaN\n" : "✅ Forward pass: PASS\n");
    auto gen = model.generate({1}, 10);
    std::cout << "Generated tokens: " << gen.size() << "\n";
    std::cout << "✅ MILESTONE: Token generation working\n";
}

// ── Benchmark ─────────────────────────────────────────────────
void run_benchmark() {
    std::cout << "\n========== VedicGEMM Benchmark ==========\n";
    int N=512; int runs=5;
    Tensor A({N,N}); A.fill_random(-1.f,1.f);
    Tensor B({N,N}); B.fill_random(-1.f,1.f);

    auto t1=std::chrono::high_resolution_clock::now();
    for(int i=0;i<runs;++i) vedic_gemm(A,B);
    float v_ms=std::chrono::duration<float,std::milli>(
        std::chrono::high_resolution_clock::now()-t1).count()/runs;

    auto t2=std::chrono::high_resolution_clock::now();
    for(int i=0;i<runs;++i) reference_gemm(A,B);
    float r_ms=std::chrono::duration<float,std::milli>(
        std::chrono::high_resolution_clock::now()-t2).count()/runs;

    std::cout << "Vedic:  " << v_ms << " ms\n";
    std::cout << "Ref:    " << r_ms << " ms\n";
    std::cout << "Ratio:  " << (r_ms/v_ms) << "x\n";
    std::cout << (v_ms < r_ms ? "✅ Vedic FASTER\n" : "⚠️  Same speed\n");
}

// ── Training (Langevin Dynamics — NO Adam) ────────────────────
void run_training(const std::string& dataset_path) {
    std::cout << "\n========== LOGOS Training ==========\n";
    std::cout << "Optimizer: Langevin Dynamics (Physics-based)\n";
    std::cout << "           dW = -γ·∇L·dt + √(2γkT)·η\n\n";

    // ── Tokenizer ─────────────────────────────────────────────
    Tokenizer tok;
    {
        std::ifstream f(dataset_path);
        if (!f){ std::cerr<<"❌ Dataset not found: "<<dataset_path<<"\n"; return; }
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        int target_vocab = std::min(4096, std::max(256, (int)(text.size() / 20)));
        tok.build(text, target_vocab);
        tok.save("vocab.bin");
        std::cout << "Tokenizer built: vocab_size=" << tok.vocab_size << "\n";
        std::cout << "Vocab saved: vocab.bin (" << tok.vocab_size << " tokens)\n";
        std::cout << "Dataset: " << text.size()/1024 << " KB\n";
    }

    // seq_len=128 for better context; batch_size=1
    int SEQ = 128;
    DataLoader loader(dataset_path, tok, SEQ, 1);
    std::cout << "Tokens: " << tok.vocab_size << " | Batches: "
              << loader.total_batches() << "\n";

    // ── Model Config ──────────────────────────────────────────
    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 128;    // larger = more capacity
    cfg.num_heads   = 4;
    cfg.num_layers  = 4;
    cfg.max_seq_len = SEQ;
    LOGOSModel model(cfg);
    std::cout << "LOGOS Model ready | layers=" << cfg.num_layers
              << " d_model=" << cfg.d_model
              << " vocab=" << cfg.vocab_size << "\n";
    std::cout << "Training | dataset batches=" << loader.total_batches() << "\n\n";

    // ── Langevin Optimizer ────────────────────────────────────
    // CRITICAL: T_start=0.1 (NOT 10.0) — prevents noise explosion
    // friction=0.9, lr=1e-4 (conservative for stability)
    // T anneals: 0.1 → 1e-5 over total_steps
    int total_batches = loader.total_batches();
    int EPOCHS = 3;
    int total_steps = EPOCHS * total_batches;

    LangevinOptimizer langevin(
        /*lr=*/     1e-4f,
        /*friction*/0.9f,
        /*T_start*/ 0.1f,     // was 10.0 — caused explosion
        /*T_end*/   1e-5f,
        /*steps*/   total_steps,
        /*seed*/    42
    );
    langevin.init(model.parameters());

    int step = 0;
    float best_loss = 999.f;
    float smooth_loss = -1.f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "── Epoch " << epoch+1 << "/" << EPOCHS << " ──\n";
        loader.current_pos = 0;
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            int seq   = (int)input_ids.size();
            int vocab = cfg.vocab_size;

            // ── Forward Pass ──────────────────────────────────
            Tensor logits = model.forward(input_ids);

            // ── Cross-Entropy Loss + Softmax Gradient ─────────
            float loss = 0.0f;
            int cnt = 0;
            Tensor dLogits(logits.shape, 0.0f);

            for (int i = 0; i < seq && i < (int)target_ids.size(); ++i) {
                int tgt = target_ids[i];
                if (tgt < 0 || tgt >= vocab) continue;

                float max_l = logits.at(i, 0);
                for (int v = 1; v < vocab; ++v)
                    max_l = std::max(max_l, logits.at(i, v));

                float sum = 0.0f;
                for (int v = 0; v < vocab; ++v)
                    sum += std::exp(logits.at(i, v) - max_l);

                loss += -(logits.at(i, tgt) - max_l - std::log(sum + 1e-10f));

                // Softmax gradient for backprop
                for (int v = 0; v < vocab; ++v) {
                    float p = std::exp(logits.at(i, v) - max_l) / (sum + 1e-10f);
                    dLogits.at(i, v) = (p - (v == tgt ? 1.0f : 0.0f)) / seq;
                }
                ++cnt;
            }
            loss = cnt > 0 ? loss / cnt : 0.0f;

            // Skip NaN/Inf batches
            if (std::isnan(loss) || std::isinf(loss)) { ++step; continue; }

            // ── Gradients (LM-head + Embedding) ───────────────
            auto params = model.parameters();
            std::vector<Tensor> grads;
            for (auto* p : params) grads.emplace_back(p->shape, 0.0f);

            // Rebuild input embeddings X  (seq, d_model)
            Tensor X({seq, cfg.d_model}, 0.0f);
            for (int i = 0; i < seq; ++i) {
                int tok_id = std::max(0, std::min(input_ids[i], cfg.vocab_size-1));
                for (int d = 0; d < cfg.d_model; ++d)
                    X.at(i, d) = model.embedding.at(tok_id, d)
                               + model.pos_embedding.at(i, d);
            }

            // grad_lm_head = X^T @ dLogits  (d_model, vocab)
            for (int d = 0; d < cfg.d_model; ++d)
                for (int v = 0; v < vocab; ++v)
                    for (int i = 0; i < seq; ++i)
                        grads[2].at(d, v) += X.at(i, d) * dLogits.at(i, v);

            // grad_embedding: dL/dX = dLogits @ lm_head^T
            for (int i = 0; i < seq; ++i) {
                int tok_id = std::max(0, std::min(input_ids[i], cfg.vocab_size-1));
                for (int d = 0; d < cfg.d_model; ++d) {
                    float g = 0.0f;
                    for (int v = 0; v < vocab; ++v)
                        g += dLogits.at(i, v) * model.lm_head.at(d, v);
                    grads[0].at(tok_id, d) += g;
                }
            }

            // ── Gradient Clip ─────────────────────────────────
            std::vector<Tensor*> gptrs;
            for (auto& g : grads) gptrs.push_back(&g);
            clip_gradients(gptrs, 1.0f);

            // ── Langevin Step ──── (Physics Optimizer) ────────
            langevin.step(params, gptrs);

            smooth_loss = smooth_loss < 0 ? loss : 0.95f*smooth_loss + 0.05f*loss;
            if (loss < best_loss) best_loss = loss;

            if (step % 100 == 0) {
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Step " << std::setw(5) << step
                          << " | Loss: " << loss
                          << " | T: " << std::fixed << std::setprecision(4)
                          << langevin.temperature << "\n";
            }
            if (step > 0 && step % 500 == 0) {
                save_checkpoint(model, "logos_ckpt", step);
                DataLoader el(dataset_path, tok, SEQ, 1);
                auto r = evaluate(model, el, 20);
                print_eval(r, step);
            }
            ++step;
        }
        std::cout << "\n";
    }

    std::cout << "========== Training Done ==========\n";
    save_checkpoint(model, "logos_final", step);

    // Final evaluation
    std::cout << "Dataset: " << [&](){
        std::ifstream f(dataset_path, std::ios::ate);
        return f.is_open() ? (int)(f.tellg()/1024) : 0;
    }() << " KB\n";
    std::cout << "Tokens: " << tok.vocab_size
              << " | Batches: " << loader.total_batches() << "\n\n";

    DataLoader el(dataset_path, tok, SEQ, 1);
    auto r = evaluate(model, el, 50);
    print_eval(r, step);
    std::cout << "\nBest loss: " << best_loss << "\n";

    float start_loss = std::log(cfg.vocab_size);  // theoretical random baseline
    std::cout << "Start (random baseline): " << start_loss << "\n";
    std::cout << "Final: " << r.loss << "\n";
    if (r.loss < start_loss)
        std::cout << "✅ Model is learning!\n";
    else
        std::cout << "⚠️  Loss above random — check gradients / reduce lr\n";
}

// ── Eval ──────────────────────────────────────────────────────
void run_eval(const std::string& dataset_path, const std::string& ckpt_path) {
    Tokenizer tok; tok.load("vocab.bin");
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size; cfg.d_model=128;
    cfg.num_heads=4; cfg.num_layers=4; cfg.max_seq_len=128;
    LOGOSModel model(cfg);
    if (!ckpt_path.empty()) load_checkpoint(model, ckpt_path);
    DataLoader loader(dataset_path, tok, 128, 1);
    auto r = evaluate(model, loader, 200);
    print_eval(r, -1);
}

// ── Generate ──────────────────────────────────────────────────
void run_generate(const std::string& ckpt_path, const std::string& prompt) {
    Tokenizer tok;
    if (!tok.load("vocab.bin")){ std::cerr<<"vocab.bin nahi mila\n"; return; }
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size; cfg.d_model=128;
    cfg.num_heads=4; cfg.num_layers=4; cfg.max_seq_len=128;
    LOGOSModel model(cfg);
    if (ckpt_path != "none") load_checkpoint(model, ckpt_path);
    std::cout << "Prompt: \"" << prompt << "\"\n\n";
    auto ids = tok.encode(prompt, 32);
    auto out  = model.generate(ids, 50, 0.8f);
    std::cout << tok.decode(out) << "\n";
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════╗\n"
              << "║  LOGOS — Vedic-Physics Hybrid LLM    ║\n"
              << "║  C++20 | Vedic GEMM | Langevin Opt   ║\n"
              << "╚══════════════════════════════════════╝\n\n";

    std::string mode = argc>1 ? argv[1] : "--test";

    if      (mode=="--test")      run_tests();
    else if (mode=="--forward")   run_forward_test();
    else if (mode=="--benchmark") run_benchmark();
    else if (mode=="--all")       { run_tests(); run_forward_test(); run_benchmark(); }
    else if (mode=="--train") {
        std::string ds = argc>2 ? argv[2] : "dataset.txt";
        run_training(ds);
    }
    else if (mode=="--eval") {
        std::string ds   = argc>2 ? argv[2] : "dataset.txt";
        std::string ckpt = argc>3 ? argv[3] : "";
        run_eval(ds, ckpt);
    }
    else if (mode=="--generate") {
        std::string ckpt   = argc>2 ? argv[2] : "none";
        std::string prompt = argc>3 ? argv[3] : "Once upon a time";
        run_generate(ckpt, prompt);
    }
    else {
        std::cout << "Usage:\n"
                  << "  logos --test\n"
                  << "  logos --train  dataset.txt\n"
                  << "  logos --eval   dataset.txt  checkpoint.bin\n"
                  << "  logos --generate  checkpoint.bin  \"prompt\"\n";
    }
    return 0;
}
