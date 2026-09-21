// ============================================================
//  LOGOS — main.cpp
//  Entry Point: LOGOS Vedic-Physics Hybrid LLM
//
//  Modes:
//    --test      : Unit tests (Tensor, VedicGEMM, Tokenizer)
//    --forward   : Forward pass smoke test
//    --benchmark : VedicGEMM vs Reference speed
//    --all       : Sab tests ek saath
//    --train     : Training loop (dataset.txt chahiye)
//    --eval      : Evaluate perplexity + accuracy
//    --generate  : Text generation from trained model
//
//  Build:
//    g++ -std=c++20 -O3 -march=native -o logos src/main.cpp
//
//  Train example:
//    logos --train dataset.txt
//  Eval example:
//    logos --eval dataset.txt logos_checkpoint_step5000.bin
//  Generate example:
//    logos --generate logos_checkpoint_step5000.bin "Once upon a time"
// ============================================================

#include "../include/Tensor.hpp"
#include "Tokenizer.cpp"
#include "Model.cpp"
#include "Trainer.cpp"
#include "Checkpoint.cpp"
#include "DataLoader.cpp"
#include "Evaluate.cpp"
#include <iostream>
#include <string>
#include <chrono>

// ── Unit Tests ────────────────────────────────────────────────
void run_tests() {
    std::cout << "\n========== LOGOS Unit Tests ==========\n";

    std::cout << "\n[1] Tensor.hpp test...\n";
    Tensor A({4, 4}); A.fill_random(-1.f, 1.f);
    Tensor B = A + A;
    assert(std::abs(B.at(0,0) - 2.0f * A.at(0,0)) < 1e-5f);
    assert(!A.has_nan());
    A.print("Tensor A");
    std::cout << "✅ Tensor: PASS\n";

    std::cout << "\n[2] VedicGEMM test...\n";
    srand(42);
    Tensor M1({32, 64}); M1.fill_random(-1.f, 1.f);
    Tensor M2({64, 32}); M2.fill_random(-1.f, 1.f);
    Tensor C_ref   = reference_gemm(M1, M2);
    Tensor C_vedic = vedic_gemm(M1, M2);
    float max_err = 0.0f;
    for (int i = 0; i < C_ref.total_size; ++i)
        max_err = std::max(max_err, std::abs(C_ref[i] - C_vedic[i]));
    std::cout << "Max error (Vedic vs Reference): " << max_err << "\n";
    if (max_err < 1e-3f) std::cout << "✅ VedicGEMM: PASS\n";
    else                  std::cout << "❌ VedicGEMM: FAIL\n";

    std::cout << "\n[3] Tokenizer test...\n";
    Tokenizer tok;
    tok.build("hello world logos vedic math hello world hello logos", 300);
    tok.save("vocab.bin");
    Tokenizer tok2; tok2.load("vocab.bin");
    auto ids = tok2.encode("hello world");
    std::cout << "Encoded 'hello world': [";
    for (int i=0;i<(int)ids.size();++i) std::cout<<ids[i]<<(i+1<(int)ids.size()?",":"");
    std::cout << "]\n";
    std::string decoded = tok2.decode(ids);
    std::cout << "Decoded: '" << decoded << "'\n";
    if (decoded.find("hello") != std::string::npos)
        std::cout << "✅ Tokenizer: PASS\n";
    else
        std::cout << "❌ Tokenizer: FAIL\n";

    std::cout << "\n========== All Tests Complete ==========\n";
}

// ── Forward Pass Smoke Test ───────────────────────────────────
void run_forward_test() {
    std::cout << "\n========== Forward Pass Test ==========\n";
    ModelConfig cfg;
    cfg.d_model    = 64;
    cfg.num_heads  = 4;
    cfg.num_layers = 2;
    cfg.vocab_size = 256;
    cfg.max_seq_len= 32;

    LOGOSModel model(cfg);
    std::vector<int> dummy_input = {1, 5, 10, 20, 42, 100, 2};

    auto t_start = std::chrono::high_resolution_clock::now();
    Tensor logits = model.forward(dummy_input);
    auto t_end = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Output logits shape: (" << logits.rows() << ", " << logits.cols() << ")\n";
    std::cout << "Forward pass time: " << ms << " ms\n";
    logits.print("Logits (first 2 rows)");

    if (!logits.has_nan()) std::cout << "✅ Forward pass: PASS (no NaN)\n";
    else                    std::cout << "❌ Forward pass: FAIL (NaN)\n";

    std::cout << "\nGenerating 10 tokens (gibberish expected — random weights):\n";
    auto generated = model.generate({1}, 10);
    std::cout << "Generated IDs: [";
    for (int i=0;i<(int)generated.size();++i)
        std::cout<<generated[i]<<(i+1<(int)generated.size()?",":"");
    std::cout << "]\n";
    std::cout << "✅ MILESTONE: Token generation working\n";
}

// ── Speed Benchmark ───────────────────────────────────────────
void run_benchmark() {
    std::cout << "\n========== VedicGEMM Speed Benchmark ==========\n";
    int N = 512;
    Tensor A({N, N}); A.fill_random(-1.f, 1.f);
    Tensor B({N, N}); B.fill_random(-1.f, 1.f);
    int runs = 5;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) vedic_gemm(A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    float vedic_ms = std::chrono::duration<float,std::milli>(t2-t1).count() / runs;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) reference_gemm(A, B);
    auto t4 = std::chrono::high_resolution_clock::now();
    float ref_ms = std::chrono::duration<float,std::milli>(t4-t3).count() / runs;

    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "Vedic GEMM:     " << vedic_ms << " ms/op\n";
    std::cout << "Reference GEMM: " << ref_ms   << " ms/op\n";
    std::cout << "Speedup: " << (ref_ms / vedic_ms) << "x\n";
    if (vedic_ms < ref_ms)
        std::cout << "✅ Vedic GEMM FASTER — Research Metric 1: WIN\n";
    else
        std::cout << "⚠️  Same speed — tiling aur tune karo\n";
}

// ── Training Mode ─────────────────────────────────────────────
void run_training(const std::string& dataset_path) {
    std::cout << "\n========== LOGOS Training ==========\n";

    // Step 1: Tokenizer build karo
    std::cout << "[1/4] Building tokenizer from dataset...\n";
    Tokenizer tok;
    {
        std::ifstream f(dataset_path);
        if (!f) { std::cerr << "❌ Dataset not found: " << dataset_path << "\n"; return; }
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        // First 2MB se vocab build (speed ke liye)
        std::string vocab_text = text.substr(0, std::min((int)text.size(), 2*1024*1024));
        tok.build(vocab_text, 4096);
        tok.save("vocab.bin");
    }

    // Step 2: DataLoader
    std::cout << "[2/4] Loading dataset...\n";
    DataLoader loader(dataset_path, tok, 128, 1);  // seq=128 (CPU pe manageable)

    // Step 3: Model (small config for CPU training)
    std::cout << "[3/4] Initializing model...\n";
    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 128;   // CPU pe trainable size
    cfg.num_heads   = 4;
    cfg.num_layers  = 4;
    cfg.max_seq_len = 128;
    LOGOSModel model(cfg);

    // Step 4: Training loop
    std::cout << "[4/4] Training...\n";
    std::cout << "      Press Ctrl+C to stop — checkpoint auto-saved every 500 steps\n\n";

    Trainer trainer(model, 3e-4f);
    int step = 0;
    int total_epochs = 3;
    float best_loss = 999.f;

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        std::cout << "\n── Epoch " << epoch+1 << "/" << total_epochs << " ──\n";
        loader.current_pos = 0;
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            float loss = trainer.train_step(input_ids, target_ids);
            if (loss < 0) { std::cerr << "NaN detected — stopping\n"; return; }

            // Log every 100 steps
            if (step % 100 == 0) {
                std::cout << "Step " << std::setw(6) << step
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss
                          << " | T=" << trainer.optimizer.temperature << "\n";

                if (loss < best_loss) { best_loss = loss; }
            }

            // Checkpoint every 500 steps
            if (step > 0 && step % 500 == 0) {
                save_checkpoint(model, "logos_checkpoint", step);
                // Quick eval
                DataLoader eval_loader(dataset_path, tok, 128, 1);
                auto result = evaluate(model, eval_loader, 20);
                print_eval(result, step);
            }

            ++step;
        }
    }

    // Final checkpoint + evaluation
    std::cout << "\n========== Training Complete ==========\n";
    save_checkpoint(model, "logos_final", step);

    DataLoader eval_loader(dataset_path, tok, 128, 1);
    auto result = evaluate(model, eval_loader, 100);
    print_eval(result, step);

    std::cout << "Best loss seen: " << best_loss << "\n";
    std::cout << "Model saved: logos_final_step" << step << ".bin\n";
}

// ── Eval Mode ─────────────────────────────────────────────────
void run_eval(const std::string& dataset_path,
              const std::string& checkpoint_path) {
    std::cout << "\n========== LOGOS Evaluation ==========\n";

    Tokenizer tok;
    tok.load("vocab.bin");

    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 128;
    cfg.num_heads   = 4;
    cfg.num_layers  = 4;
    cfg.max_seq_len = 128;
    LOGOSModel model(cfg);

    if (!checkpoint_path.empty())
        load_checkpoint(model, checkpoint_path);

    DataLoader loader(dataset_path, tok, 128, 1);
    auto result = evaluate(model, loader, 200);
    print_eval(result, -1);
}

// ── Generate Mode ─────────────────────────────────────────────
void run_generate(const std::string& checkpoint_path,
                  const std::string& prompt_text) {
    std::cout << "\n========== LOGOS Text Generation ==========\n";

    Tokenizer tok;
    if (!tok.load("vocab.bin")) {
        std::cerr << "❌ vocab.bin nahi mila — pehle train karo\n"; return;
    }

    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 128;
    cfg.num_heads   = 4;
    cfg.num_layers  = 4;
    cfg.max_seq_len = 128;
    LOGOSModel model(cfg);

    if (!checkpoint_path.empty() && checkpoint_path != "none")
        load_checkpoint(model, checkpoint_path);

    std::cout << "Prompt: \"" << prompt_text << "\"\n\n";
    auto prompt_ids = tok.encode(prompt_text, 64);
    auto generated  = model.generate(prompt_ids, 100, 0.8f);  // temp=0.8

    std::cout << "Generated:\n" << tok.decode(generated) << "\n";
}

// ── Main ──────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║  LOGOS — Vedic-Physics Hybrid LLM    ║\n";
    std::cout << "║  C++20 | Vedic GEMM | Langevin Opt   ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    std::string mode = argc > 1 ? argv[1] : "--test";

    if      (mode == "--test")      { run_tests(); }
    else if (mode == "--forward")   { run_forward_test(); }
    else if (mode == "--benchmark") { run_benchmark(); }
    else if (mode == "--all")       { run_tests(); run_forward_test(); run_benchmark(); }
    else if (mode == "--train") {
        std::string dataset = argc > 2 ? argv[2] : "dataset.txt";
        run_training(dataset);
    }
    else if (mode == "--eval") {
        std::string dataset    = argc > 2 ? argv[2] : "dataset.txt";
        std::string checkpoint = argc > 3 ? argv[3] : "";
        run_eval(dataset, checkpoint);
    }
    else if (mode == "--generate") {
        std::string checkpoint = argc > 2 ? argv[2] : "none";
        std::string prompt     = argc > 3 ? argv[3] : "Once upon a time";
        run_generate(checkpoint, prompt);
    }
    else {
        std::cout << "Usage:\n";
        std::cout << "  logos --test\n";
        std::cout << "  logos --forward\n";
        std::cout << "  logos --benchmark\n";
        std::cout << "  logos --train  dataset.txt\n";
        std::cout << "  logos --eval   dataset.txt  checkpoint.bin\n";
        std::cout << "  logos --generate  checkpoint.bin  \"your prompt\"\n";
    }

    return 0;
}
