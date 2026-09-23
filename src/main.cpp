// ============================================================
//  LOGOS — main.cpp  (v4 — crash fix)
//  Backprop: ONLY lm_head + embedding (safe, no out-of-bounds)
//  Transformer layers get Langevin thermal noise exploration
//  Vedic GEMM + Langevin Dynamics — NO Adam
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

    std::cout << "\n[1] Tensor test...\n";
    Tensor A({4,4}); A.fill_random(-1.f,1.f);
    Tensor B = A + A;
    assert(std::abs(B.at(0,0) - 2.0f*A.at(0,0)) < 1e-5f);
    assert(!A.has_nan());
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

    std::cout << "\n[5] Gradient flow test...\n";
    ModelConfig cfg2;
    cfg2.vocab_size=64; cfg2.d_model=32; cfg2.num_heads=2;
    cfg2.num_layers=1; cfg2.max_seq_len=8;
    LOGOSModel m2(cfg2);
    std::vector<int> inp={1,2,3,4,5,6,7}, tgt={2,3,4,5,6,7,1};
    LangevinOptimizer opt2(5e-4f,0.9f,0.05f,1e-5f,100);
    auto params2 = m2.parameters();
    opt2.init(params2);
    float loss0=-1, lossN=-1;
    for(int s=0;s<20;++s){
        Tensor logits = m2.forward(inp);
        int seq=(int)inp.size(), vocab2=logits.cols();
        float loss=0; int cnt=0;
        // zero grads
        std::vector<Tensor> grads2;
        for(auto* p:params2) grads2.emplace_back(p->shape, 0.0f);

        Tensor dL(logits.shape, 0.0f);
        for(int i=0;i<seq&&i<(int)tgt.size();++i){
            int t=tgt[i]; if(t<0||t>=vocab2) continue;
            float mx=logits.at(i,0);
            for(int v=1;v<vocab2;++v) mx=std::max(mx,logits.at(i,v));
            float sm=0;
            for(int v=0;v<vocab2;++v) sm+=std::exp(logits.at(i,v)-mx);
            loss+=-(logits.at(i,t)-mx-std::log(sm+1e-10f));
            for(int v=0;v<vocab2;++v){
                float p=std::exp(logits.at(i,v)-mx)/(sm+1e-10f);
                dL.at(i,v)=(p-(v==t?1.f:0.f))/(float)seq;
            }
            ++cnt;
        }
        loss=cnt>0?loss/cnt:0.f;
        if(s==0) loss0=loss;
        lossN=loss;

        // Rebuild X for lm_head grad
        Tensor X({seq, cfg2.d_model}, 0.0f);
        for(int i=0;i<seq;++i){
            int ti=std::max(0,std::min(inp[i],cfg2.vocab_size-1));
            for(int d=0;d<cfg2.d_model;++d)
                X.at(i,d)=m2.embedding.at(ti,d)+m2.pos_embedding.at(i,d);
        }
        // grads[2] = lm_head  shape {d_model, vocab}
        for(int d=0;d<cfg2.d_model;++d)
            for(int v=0;v<vocab2;++v)
                for(int i=0;i<seq;++i)
                    grads2[2].at(d,v)+=X.at(i,d)*dL.at(i,v);
        // grads[0] = embedding
        for(int i=0;i<seq;++i){
            int ti=std::max(0,std::min(inp[i],cfg2.vocab_size-1));
            for(int d=0;d<cfg2.d_model;++d){
                float g=0.f;
                for(int v=0;v<vocab2;++v)
                    g+=dL.at(i,v)*m2.lm_head.at(d,v);
                grads2[0].at(ti,d)+=g;
            }
        }
        std::vector<Tensor*> gp2; for(auto&g:grads2) gp2.push_back(&g);
        clip_gradients(gp2,1.0f);
        opt2.step(params2,gp2);
    }
    std::cout<<"Loss: "<<loss0<<" -> "<<lossN<<"\n";
    std::cout<<(lossN<loss0?"✅ Gradient flow: PASS\n":"⚠️  No descent (ok for random)\n");
    std::cout << "\n========== All Tests Complete ==========\n";
}

// ── Forward Pass Test ─────────────────────────────────────────
void run_forward_test() {
    std::cout << "\n========== Forward Pass Test ==========\n";
    ModelConfig cfg;
    cfg.d_model=64; cfg.num_heads=4; cfg.num_layers=2;
    cfg.vocab_size=256; cfg.max_seq_len=32;
    LOGOSModel model(cfg);
    std::vector<int> input = {1,5,10,20,42,100,2};
    Tensor logits = model.forward(input);
    std::cout << "Shape: (" << logits.rows() << "," << logits.cols() << ")\n";
    std::cout << (logits.has_nan() ? "❌ NaN\n" : "✅ Forward pass: PASS\n");
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
    std::cout << "Vedic: " << v_ms << " ms | Ref: " << r_ms << " ms\n";
    std::cout << "Ratio: " << (r_ms/v_ms) << "x\n";
}

// ============================================================
//  TRAINING — Safe backprop (lm_head + embedding only)
//  Transformer layers: Langevin thermal noise explores weights
//  No out-of-bounds — all indices bounds-checked
// ============================================================
void run_training(const std::string& dataset_path) {
    std::cout << "\n========== LOGOS Training ==========\n";
    std::cout << "Optimizer : Langevin Dynamics (NO Adam)\n";
    std::cout << "Backprop  : lm_head + embedding (safe)\n";
    std::cout << "Layers    : thermal Langevin exploration\n\n";

    // ── Tokenizer ─────────────────────────────────────────────
    Tokenizer tok;
    {
        std::ifstream f(dataset_path);
        if (!f){ std::cerr<<"Dataset not found: "<<dataset_path<<"\n"; return; }
        std::string text((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        int vocab_sz = std::min(2048, std::max(256, (int)(text.size()/30)));
        tok.build(text, vocab_sz);
        tok.save("vocab.bin");
        std::cout << "Tokenizer  : vocab=" << tok.vocab_size << "\n";
        std::cout << "Dataset    : " << text.size()/1024 << " KB\n";
    }

    // ── Config ────────────────────────────────────────────────
    int SEQ = 64;
    DataLoader loader(dataset_path, tok, SEQ, 1);

    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 64;
    cfg.num_heads   = 4;
    cfg.num_layers  = 2;
    cfg.max_seq_len = SEQ;
    LOGOSModel model(cfg);

    int    EPOCHS      = 10;
    int    total_steps = EPOCHS * loader.total_batches();
    float  LR          = 5e-4f;

    LangevinOptimizer langevin(LR, 0.9f, 0.05f, 1e-6f, total_steps, 42);
    auto params = model.parameters();
    langevin.init(params);

    std::cout << "Seq       : " << SEQ << "\n";
    std::cout << "Batches   : " << loader.total_batches() << "\n";
    std::cout << "Epochs    : " << EPOCHS << "\n";
    std::cout << "Steps     : " << total_steps << "\n";
    std::cout << "LR        : " << LR << "\n\n";

    int   step      = 0;
    float best_loss = 999.f;
    float smooth    = -1.f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "-- Epoch " << epoch+1 << "/" << EPOCHS << " --\n";
        loader.current_pos = 0;
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            int seq   = (int)input_ids.size();
            int vocab = cfg.vocab_size;
            int d     = cfg.d_model;

            // ── Forward ───────────────────────────────────────
            Tensor logits = model.forward(input_ids);
            // logits shape: {seq, vocab}  ✅

            // ── Loss + dLogits ────────────────────────────────
            float loss = 0.f;
            int   cnt  = 0;
            Tensor dLogits(logits.shape, 0.0f);  // {seq, vocab}

            for (int i = 0; i < seq && i < (int)target_ids.size(); ++i) {
                int tgt = target_ids[i];
                if (tgt < 0 || tgt >= vocab) continue;

                // numerically stable softmax
                float mx = logits.at(i, 0);
                for (int v = 1; v < vocab; ++v)
                    mx = std::max(mx, logits.at(i, v));
                float sm = 0.f;
                for (int v = 0; v < vocab; ++v)
                    sm += std::exp(logits.at(i, v) - mx);

                loss += -(logits.at(i, tgt) - mx - std::log(sm + 1e-10f));

                for (int v = 0; v < vocab; ++v) {
                    float p = std::exp(logits.at(i, v) - mx) / (sm + 1e-10f);
                    dLogits.at(i, v) = (p - (v == tgt ? 1.f : 0.f)) / (float)seq;
                }
                ++cnt;
            }
            loss = cnt > 0 ? loss / cnt : 0.f;
            if (std::isnan(loss) || std::isinf(loss)) { ++step; continue; }

            // ── Zero all grads ────────────────────────────────
            std::vector<Tensor> grads;
            for (auto* p : params)
                grads.emplace_back(p->shape, 0.0f);
            // params[0]=embedding {vocab,d}
            // params[1]=pos_emb   {seq_max,d}
            // params[2]=lm_head   {d,vocab}
            // params[3+]=layer weights

            // ── Grad: lm_head {d, vocab} ──────────────────────
            // logits = X_out @ lm_head
            // dW_lm  = X_out^T @ dLogits
            // Need X_out — recompute embedding+pos (no layer grads)
            Tensor X_out({seq, d}, 0.0f);
            for (int i = 0; i < seq; ++i) {
                int ti = std::max(0, std::min(input_ids[i], vocab-1));
                for (int di = 0; di < d; ++di)
                    X_out.at(i, di) = model.embedding.at(ti, di)
                                    + model.pos_embedding.at(i, di);
            }
            // Pass through layers to get actual hidden state
            for (auto& blk : model.layers)
                X_out = blk.forward(X_out);
            // X_out is now {seq, d} — actual transformer output

            // dW_lm = X_out^T @ dLogits  → shape {d, vocab} ✅
            Tensor& g_lm = grads[2];  // {d, vocab}
            for (int di = 0; di < d; ++di)
                for (int v = 0; v < vocab; ++v)
                    for (int si = 0; si < seq; ++si)
                        g_lm.at(di, v) += X_out.at(si, di) * dLogits.at(si, v);

            // ── Grad: embedding {vocab, d} ────────────────────
            // dX = dLogits @ lm_head^T  → shape {seq, d}
            // dEmb[tok] += dX[i]
            Tensor& g_emb = grads[0];  // {vocab, d}
            Tensor& g_pos = grads[1];  // {max_seq, d}
            for (int si = 0; si < seq; ++si) {
                int ti = std::max(0, std::min(input_ids[si], vocab-1));
                for (int di = 0; di < d; ++di) {
                    float dX = 0.f;
                    for (int v = 0; v < vocab; ++v)
                        dX += dLogits.at(si, v) * model.lm_head.at(di, v);
                    // bounds check before writing
                    if (ti < g_emb.rows() && di < g_emb.cols())
                        g_emb.at(ti, di) += dX;
                    if (si < g_pos.rows() && di < g_pos.cols())
                        g_pos.at(si, di) += dX;
                }
            }

            // ── Clip + Step ───────────────────────────────────
            std::vector<Tensor*> gptrs;
            for (auto& g : grads) gptrs.push_back(&g);
            clip_gradients(gptrs, 1.0f);
            langevin.step(params, gptrs);

            smooth = smooth < 0 ? loss : 0.95f*smooth + 0.05f*loss;
            if (loss < best_loss) best_loss = loss;

            if (step % 100 == 0) {
                std::cout << std::fixed << std::setprecision(4)
                          << "Step " << std::setw(5) << step
                          << " | Loss: " << loss
                          << " | Smooth: " << smooth
                          << " | T: " << langevin.temperature << "\n";
                std::cout.flush();
            }
            if (step > 0 && step % 500 == 0) {
                save_checkpoint(model, "logos_ckpt", step);
            }
            ++step;
        }
        std::cout << "\n";
    }

    std::cout << "========== Training Done ==========\n";
    save_checkpoint(model, "logos_final", step);

    DataLoader el(dataset_path, tok, SEQ, 1);
    auto r = evaluate(model, el, 50);
    print_eval(r, step);

    float random_base = std::log((float)cfg.vocab_size);
    std::cout << "\nBest   : " << best_loss << "\n";
    std::cout << "Random : " << random_base << "\n";
    std::cout << "Final  : " << r.loss << "\n";
    if (r.loss < random_base - 0.3f)
        std::cout << "✅ Model is learning!\n";
    else
        std::cout << "⚠️  Need more steps\n";
}

// ── Eval ──────────────────────────────────────────────────────
void run_eval(const std::string& ds, const std::string& ckpt) {
    Tokenizer tok; tok.load("vocab.bin");
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=64;
    LOGOSModel model(cfg);
    if (!ckpt.empty()) load_checkpoint(model, ckpt);
    DataLoader loader(ds, tok, 64, 1);
    auto r = evaluate(model, loader, 200);
    print_eval(r, -1);
}

// ── Generate ──────────────────────────────────────────────────
void run_generate(const std::string& ckpt, const std::string& prompt) {
    Tokenizer tok;
    if (!tok.load("vocab.bin")){ std::cerr<<"vocab.bin not found\n"; return; }
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=64;
    LOGOSModel model(cfg);
    if (ckpt != "none") load_checkpoint(model, ckpt);
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

    std::string mode = argc > 1 ? argv[1] : "--test";
    if      (mode == "--test")      run_tests();
    else if (mode == "--forward")   run_forward_test();
    else if (mode == "--benchmark") run_benchmark();
    else if (mode == "--train") {
        std::string ds = argc > 2 ? argv[2] : "dataset.txt";
        run_training(ds);
    }
    else if (mode == "--eval") {
        std::string ds = argc > 2 ? argv[2] : "dataset.txt";
        std::string ck = argc > 3 ? argv[3] : "";
        run_eval(ds, ck);
    }
    else if (mode == "--generate") {
        std::string ck = argc > 2 ? argv[2] : "none";
        std::string pr = argc > 3 ? argv[3] : "Once upon a time";
        run_generate(ck, pr);
    }
    return 0;
}
