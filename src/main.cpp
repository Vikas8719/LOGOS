// ============================================================
//  LOGOS — main.cpp
//  Entry Point: LOGOS Vedic-Physics Hybrid LLM
//
//  Modes:
//    --test    : Run unit tests (Tensor, VedicGEMM, Tokenizer)
//    --forward : Forward pass smoke test (gibberish output)
//    --train   : Start training loop
//    --generate: Text generation from checkpoint
//
//  Build:
//    g++ -std=c++20 -O3 -march=native -o logos src/main.cpp
// ============================================================

#include "../include/Tensor.hpp"
#include "Tokenizer.cpp"
#include "Model.cpp"
#include "Trainer.cpp"
#include "Checkpoint.cpp"
#include <iostream>
#include <string>
#include <chrono>

// ── Unit Tests ────────────────────────────────────────────────
void run_tests() {
    std::cout << "\n========== LOGOS Unit Tests ==========\n";

    // Test 1: Tensor
    std::cout << "\n[1] Tensor.hpp test...\n";
    Tensor A({4, 4}); A.fill_random(-1.f, 1.f);
    Tensor B = A + A;
    assert(std::abs(B.at(0,0) - 2.0f * A.at(0,0)) < 1e-5f);
    assert(!A.has_nan());
    A.print("Tensor A");
    std::cout << "✅ Tensor: PASS\n";

    // Test 2: VedicGEMM
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
    if (max_err < 1e-3f)
        std::cout << "✅ VedicGEMM: PASS\n";
    else
        std::cout << "❌ VedicGEMM: FAIL (max_err=" << max_err << ")\n";

    // Test 3: Tokenizer
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
    cfg.d_model    = 64;   // Small for quick test
    cfg.num_heads  = 4;
    cfg.num_layers = 2;
    cfg.vocab_size = 256;
    cfg.max_seq_len= 32;

    LOGOSModel model(cfg);

    std::vector<int> dummy_input = {1, 5, 10, 20, 42, 100, 2};  // BOS...EOS
    std::cout << "Running forward pass with " << dummy_input.size() << " tokens...\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    Tensor logits = model.forward(dummy_input);
    auto t_end = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Output logits shape: (" << logits.rows() << ", " << logits.cols() << ")\n";
    std::cout << "Forward pass time: " << ms << " ms\n";
    logits.print("Logits (first 2 rows)");

    if (!logits.has_nan())
        std::cout << "✅ Forward pass: PASS (no NaN)\n";
    else
        std::cout << "❌ Forward pass: FAIL (NaN detected)\n";

    // Generate some tokens
    std::cout << "\nGenerating 10 tokens (random weights = gibberish expected):\n";
    auto generated = model.generate({1}, 10);
    std::cout << "Generated IDs: [";
    for (int i=0;i<(int)generated.size();++i)
        std::cout<<generated[i]<<(i+1<(int)generated.size()?",":"");
    std::cout << "]\n";
    std::cout << "✅ MILESTONE: Model can generate tokens (gibberish = correct at this stage)\n";
}

// ── Speed Benchmark ───────────────────────────────────────────
void run_benchmark() {
    std::cout << "\n========== VedicGEMM Speed Benchmark ==========\n";
    int N = 512;
    Tensor A({N, N}); A.fill_random(-1.f, 1.f);
    Tensor B({N, N}); B.fill_random(-1.f, 1.f);

    int runs = 5;
    // Vedic GEMM
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) vedic_gemm(A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    float vedic_ms = std::chrono::duration<float,std::milli>(t2-t1).count() / runs;

    // Reference GEMM
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) reference_gemm(A, B);
    auto t4 = std::chrono::high_resolution_clock::now();
    float ref_ms = std::chrono::duration<float,std::milli>(t4-t3).count() / runs;

    std::cout << "Matrix size: " << N << "×" << N << "\n";
    std::cout << "Vedic GEMM:     " << vedic_ms << " ms/op\n";
    std::cout << "Reference GEMM: " << ref_ms   << " ms/op\n";
    std::cout << "Speedup: " << (ref_ms / vedic_ms) << "×\n";
    if (vedic_ms < ref_ms)
        std::cout << "✅ Vedic GEMM is FASTER — Research metric 1: WIN\n";
    else
        std::cout << "⚠️  Vedic GEMM not faster at this size — investigate tiling\n";
}

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║  LOGOS — Vedic-Physics Hybrid LLM    ║\n";
    std::cout << "║  C++20 | 10M params | 512 ctx        ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    std::string mode = "--test";
    if (argc > 1) mode = argv[1];

    if (mode == "--test") {
        run_tests();
    } else if (mode == "--forward") {
        run_forward_test();
    } else if (mode == "--benchmark") {
        run_benchmark();
    } else if (mode == "--all") {
        run_tests();
        run_forward_test();
        run_benchmark();
    } else {
        std::cout << "Usage: logos [--test | --forward | --benchmark | --all]\n";
    }

    return 0;
}
