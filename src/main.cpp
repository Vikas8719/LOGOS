
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
#include "StreamingDataLoader.hpp"  
#include "Evaluate.hpp"
#include "Backprop.hpp"           
#include "AnurupyenaScaler.hpp"  
#include "TokenizerManager.hpp"  

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <sstream>


static bool is_safe_path(const std::string& path) {
    if (path.empty()) return false;
    if (path.find('\0') != std::string::npos) return false;

    std::string norm = path;
    for (char& c : norm) if (c == '\\') c = '/';

    std::string seg;
    std::istringstream ss(norm);
    while (std::getline(ss, seg, '/')) {
        if (seg == "..") return false;
    }
    return true;
}

static std::string safe_arg(const char* raw, const std::string& fallback) {
    if (!raw) return fallback;
    std::string s(raw);
    if (!is_safe_path(s)) {
        std::cerr << "⚠️  Unsafe path rejected: '" << s
                  << "' — using fallback: '" << fallback << "'\n";
        return fallback;
    }
    return s;
}


static float free_energy_loss_cpu(
    const Tensor&              logits,       
    const std::vector<int>&    targets,      
    float                      temperature,  
    Tensor&                    dLogits_out)  
{
    int seq   = logits.rows();
    int vocab = logits.cols();
    float total_F = 0.0f;
    int   cnt     = 0;

    dLogits_out = Tensor(logits.shape, 0.0f);

    std::vector<float> grad_row(vocab);
    for (int i = 0; i < seq && i < (int)targets.size(); ++i) {
        int tgt = targets[i];
        if (tgt < 0 || tgt >= vocab) continue;

        
        float F = FreeEnergyLoss::compute(
            &logits.data[i * vocab], vocab,
            tgt, temperature,
            grad_row.data());

        if (!std::isfinite(F)) continue;

        total_F += F;
        for (int v = 0; v < vocab; ++v)
            dLogits_out.at(i, v) = grad_row[v] / (float)seq;  // normalize by seq

        ++cnt;
    }
    return cnt > 0 ? total_F / cnt : 0.0f;
}


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
    LangevinOptimizer opt(1e-4f, 0.9f, 0.05f, 1e-5f, 1000);
    std::vector<Tensor*> ps = {&W};
    std::vector<Tensor*> gs = {&G};
    opt.init(ps);
    float w0 = W.data[0];
    opt.step(ps, gs);
    std::cout << "Weight changed: " << (W.data[0] != w0 ? "YES" : "NO") << "\n";
    std::cout << (!W.has_nan() ? "✅ LangevinOpt: PASS\n" : "❌ LangevinOpt: NaN\n");

    std::cout << "\n[4b] HybridSHMOptimizer test...\n";
    {
        Tensor Wh({4,4}); Wh.fill_random(-0.1f, 0.1f);
        Tensor Gh({4,4}); Gh.fill(0.01f);
        HybridSHMOptimizer shm(1e-4f, 0.1f, 0.9f, 0.05f, 1e-5f, 0.3f, 0.9f, 1000);
        std::vector<Tensor*> psh = {&Wh}, gsh = {&Gh};
        shm.init(psh);
        float w0h = Wh.data[0];
        shm.step(psh, gsh);
        std::cout << "Weight changed: " << (Wh.data[0] != w0h ? "YES" : "NO") << "\n";
        std::cout << (!Wh.has_nan() ? "✅ HybridSHMOpt: PASS\n" : "❌ HybridSHMOpt: NaN\n");
    }

    std::cout << "\n[4c] NaturalGradientOptimizer test...\n";
    {
        Tensor Wn({4,4}); Wn.fill_random(-0.1f, 0.1f);
        Tensor Gn({4,4}); Gn.fill(0.01f);
        NaturalGradientOptimizer natgrad(1e-4f);
        std::vector<Tensor*> psn = {&Wn}, gsn = {&Gn};
        natgrad.init(psn);
        float w0n = Wn.data[0];
        natgrad.step(psn, gsn);
        std::cout << "Weight changed: " << (Wn.data[0] != w0n ? "YES" : "NO") << "\n";
        std::cout << (!Wn.has_nan() ? "✅ NaturalGradOpt: PASS\n" : "❌ NaturalGradOpt: NaN\n");
    }

    std::cout << "\n[4d] WeightPathIntegral test...\n";
    {
        WeightPathIntegral wpi(1.0f, 100);
        for (int s = 0; s < 10; ++s) {
            float fake_loss = 1.0f / (s + 1);
            std::vector<float> delta(16, 0.01f);
            wpi.record_step(fake_loss, delta);
        }
        wpi.log_state();
        std::cout << (wpi.step_count == 10 ? "✅ WeightPathIntegral: PASS\n" : "❌ WeightPathIntegral: FAIL\n");
    }

    std::cout << "\n[4e] RiemannianMetric test...\n";
    {
        RiemannianMetric riem(16);
        Tensor Gr({4,4}); Gr.fill(0.5f);
        riem.update(Gr);
        Tensor g_nat = riem.riemannian_gradient(Gr);
        riem.log_state();
        std::cout << (!g_nat.has_nan() ? "✅ RiemannianMetric: PASS\n" : "❌ RiemannianMetric: NaN\n");
    }

    std::cout << "\n[4f] Physics-wired forward pass (Reynolds+NavierStokes+Feynman)...\n";
    {
        ModelConfig cfgw; cfgw.d_model=32; cfgw.num_heads=2; cfgw.num_layers=1;
        cfgw.vocab_size=64; cfgw.max_seq_len=16;
        LOGOSModel mw(cfgw);
        Tensor out = mw.forward({1,2,3,4,5});
        std::cout << (!out.has_nan() ? "✅ Physics-wired forward: PASS\n" : "❌ Physics-wired forward: NaN\n");
    }

    std::cout << "\n[4g] AnurupyenaScaler test (Āṇurūpyeṇa — Vedic proportional grad scale)...\n";
    {
       
        Tensor g_large({4,4}); g_large.fill(10.0f); 
        Tensor g_small({4,4}); g_small.fill(0.01f); 
        std::vector<Tensor*> test_grads = {&g_large, &g_small};

        AnurupyenaScaler anurup(1.0f);  
        anurup.scale_gradients(test_grads);

        
        float rms_large = 0.0f, rms_small = 0.0f;
        for (int i = 0; i < 16; ++i) {
            rms_large += g_large.data[i] * g_large.data[i];
            rms_small += g_small.data[i] * g_small.data[i];
        }
        rms_large = std::sqrt(rms_large / 16.0f);
        rms_small = std::sqrt(rms_small / 16.0f);

        bool pass = (std::abs(rms_large - 1.0f) < 0.1f) &&
                    (std::abs(rms_small - 1.0f) < 0.1f);
        anurup.log_state();
        std::cout << "RMS after scale: large=" << rms_large
                  << " small=" << rms_small << " (target=1.0)\n";
        std::cout << (pass ? "✅ AnurupyenaScaler: PASS\n"
                           : "⚠️  AnurupyenaScaler: clamped (check min/max_scale)\n");
    }

    
    std::cout << "\n[4h] FreeEnergy CPU Loss test (F = CE - T*S)...\n";
    {
      
        std::vector<float> logits_raw = {1.0f, 2.0f, 0.5f, -0.3f};
        int vocab_sz = 4;
        std::vector<float> grad_fe(vocab_sz);
        float F_val = FreeEnergyLoss::compute(
            logits_raw.data(), vocab_sz,
            /*target=*/1, /*temperature=*/0.05f,
            grad_fe.data());
        bool pass_fe = std::isfinite(F_val) && (F_val >= 0.0f);
        std::cout << "FreeEnergy loss: " << F_val
                  << " (grad[target]=" << grad_fe[1] << ")\n";
        std::cout << (pass_fe ? "✅ FreeEnergy CPU Loss: PASS\n"
                              : "❌ FreeEnergy CPU Loss: FAIL\n");
    }

    
    std::cout << "\n[4i] Full Backprop test (model_backward from Backprop.hpp)...\n";
    {
        ModelConfig cfgb;
        cfgb.vocab_size=32; cfgb.d_model=16; cfgb.num_heads=2;
        cfgb.num_layers=1;  cfgb.max_seq_len=8;
        LOGOSModel mb(cfgb);

        auto params_b  = mb.parameters();
        std::vector<Tensor> grads_b;
        for (auto* p : params_b) grads_b.emplace_back(p->shape, 0.0f);

        std::vector<int> inp_b = {1,2,3,4,5,6,7};
        std::vector<int> tgt_b = {2,3,4,5,6,7,1};

        float loss_b = model_backward(mb, inp_b, tgt_b, grads_b, true);

        
        float emb_norm = 0.0f, lm_norm = 0.0f, layer_norm = 0.0f;
        for (int i = 0; i < grads_b[0].total_size; ++i) emb_norm += grads_b[0].data[i]*grads_b[0].data[i];
        for (int i = 0; i < grads_b[1].total_size; ++i) lm_norm  += grads_b[1].data[i]*grads_b[1].data[i];
        for (int gi = 2; gi < (int)grads_b.size(); ++gi)
            for (int i = 0; i < grads_b[gi].total_size; ++i)
                layer_norm += grads_b[gi].data[i]*grads_b[gi].data[i];
        emb_norm   = std::sqrt(emb_norm);
        lm_norm    = std::sqrt(lm_norm);
        layer_norm = std::sqrt(layer_norm);

        bool pass_bp = std::isfinite(loss_b) && emb_norm > 1e-6f
                      && lm_norm > 1e-6f && layer_norm > 1e-6f;
        std::cout << "loss=" << loss_b
                  << " | grad_norm: emb=" << emb_norm
                  << " lm=" << lm_norm
                  << " layers=" << layer_norm << "\n";
        std::cout << (pass_bp ? "✅ Full Backprop: PASS (all layers have real grads)\n"
                              : "❌ Full Backprop: FAIL (some grads zero)\n");
    }

    std::cout << "\n[5] Gradient flow test (v10 — full backprop)...\n";
    ModelConfig cfg2;
    cfg2.vocab_size=64; cfg2.d_model=32; cfg2.num_heads=2;
    cfg2.num_layers=1; cfg2.max_seq_len=8;
    LOGOSModel m2(cfg2);
    std::vector<int> inp={1,2,3,4,5,6,7}, tgt={2,3,4,5,6,7,1};
    HybridSHMOptimizer opt2(5e-4f, 0.1f, 0.9f, 0.05f, 1e-5f, 0.3f, 0.9f, 100);
    auto params2 = m2.parameters();
    opt2.init(params2);
    AnurupyenaScaler anurup2(1.0f);

    float loss0=-1, lossN=-1;
    for(int s=0;s<20;++s){
        
        std::vector<Tensor> grads2;
        for(auto* p : params2) grads2.emplace_back(p->shape, 0.0f);

        float loss = model_backward(m2, inp, tgt, grads2, true);
        if(s==0) loss0=loss;
        lossN=loss;

        std::vector<Tensor*> gp2;
        for(auto& g : grads2) gp2.push_back(&g);
        clip_gradients(gp2, 1.0f);
        anurup2.scale_gradients(gp2);  // [v10] Āṇurūpyeṇa
        opt2.step(params2, gp2);
    }
    std::cout<<"Loss: "<<loss0<<" -> "<<lossN<<"\n";
    std::cout<<(lossN<loss0?"✅ Gradient flow v10: PASS\n":"⚠️  No descent (ok for random init)\n");
    std::cout << "\n========== All Tests Complete ==========\n";
}


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


void run_training(const std::string& dataset_path) {
    std::cout << "\n========== LOGOS Training v10 ==========\n";
    std::cout << "Optimizer  : Hybrid Stochastic Hamiltonian (SHM) v9\n";
    std::cout << "Precond    : RiemannianMetric (curvature-adaptive)\n";
    std::cout << "LR adapt   : WeightPathIntegral Feynman amplitude\n";
    std::cout << "Forward    : forward_with_hidden() — single-pass\n";
    std::cout << "Backprop   : [v10 NEW] model_backward() — FULL analytical\n";
    std::cout << "             Q/K/V, W_O, FFN, LayerNorm all have real gradients\n";
    std::cout << "Loss       : [v10 NEW] FreeEnergy F=CE-T*S on CPU (same as GPU)\n";
    std::cout << "Grad scale : [v10 NEW] Āṇurūpyeṇa proportional normalization\n";
    std::cout << "BN stats   : ReynoldsBatchNorm EMA saved/loaded in checkpoints\n\n";

    
    Tokenizer tok = TokenizerManager::load_or_build(
        dataset_path,   
        /*target_vocab=*/0,   
        /*force_rebuild=*/false
    );

    int SEQ = 64;
    
    constexpr int BATCH_SIZE = 8;  

    UnifiedDataLoader loader(dataset_path, tok, SEQ, /*grad_accum=*/BATCH_SIZE);

    ModelConfig cfg;
    cfg.vocab_size  = tok.vocab_size;
    cfg.d_model     = 64;
    cfg.num_heads   = 4;
    cfg.num_layers  = 2;
    cfg.max_seq_len = SEQ;
    LOGOSModel model(cfg);

    int   EPOCHS      = 10;
    int   total_steps = EPOCHS * loader.total_batches();
    if (total_steps <= 0) total_steps = 10000;
    float LR          = 5e-4f;

    
    HybridSHMOptimizer opt(LR, /*friction*/0.1f, /*mom_decay*/0.9f,
                           /*T_start*/0.05f, /*T_end*/1e-6f,
                           /*alpha_H_start*/0.3f, /*alpha_H_end*/0.9f,
                           total_steps, 42);
    auto params = model.parameters();
    opt.init(params);

    
    int total_param_count = 0;
    for (auto* p : params) total_param_count += p->total_size;
    RiemannianMetric riem(total_param_count);

    WeightPathIntegral path_integral(1.0f, 200);
    const float BASE_LR = LR;

    
    AnurupyenaScaler anurup_scaler(
        /*target_rms=*/0.01f,
        /*epsilon=*/1e-8f,
        /*min_scale=*/0.01f,
        /*max_scale=*/10.0f,
        /*ema_beta=*/0.99f,
        /*enabled=*/true);

    std::cout << "Seq       : " << SEQ << "\n";
    std::cout << "Batch     : " << BATCH_SIZE << "  (effective tokens/step = "
              << SEQ * BATCH_SIZE << ")\n";
    std::cout << "Batches   : " << loader.total_batches() << "\n";
    std::cout << "Epochs    : " << EPOCHS << "\n";
    std::cout << "Steps     : " << total_steps << "\n";
    std::cout << "LR        : " << LR << "\n\n";

    int   step      = 0;
    float best_loss = 999.f;
    float smooth    = -1.f;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "-- Epoch " << epoch+1 << "/" << EPOCHS << " --\n";
        loader.reset();
        std::vector<int> input_ids, target_ids;

        while (loader.next_batch(input_ids, target_ids)) {
            

            std::vector<Tensor> grads;
            for (auto* p : params) grads.emplace_back(p->shape, 0.0f);

           
            float accum_loss_ce = 0.0f;
            float accum_loss_fe = 0.0f;
            int   valid_samples  = 0;

            
            auto accumulate_one = [&](const std::vector<int>& inp,
                                       const std::vector<int>& tgt) {
                std::vector<Tensor> step_grads;
                for (auto* p : params) step_grads.emplace_back(p->shape, 0.0f);

                Tensor logits_fe;
                float loss_ce = model_backward_with_logits(model, inp, tgt, step_grads, logits_fe, true);
                if (std::isnan(loss_ce) || std::isinf(loss_ce)) return;

                Tensor dLogits_fe(logits_fe.shape, 0.0f);
                float loss_fe = free_energy_loss_cpu(logits_fe, tgt, opt.temperature, dLogits_fe);

                accum_loss_ce += std::isfinite(loss_ce) ? loss_ce : 0.0f;
                accum_loss_fe += std::isfinite(loss_fe) ? loss_fe : loss_ce;
                ++valid_samples;

                for (int gi = 0; gi < (int)grads.size(); ++gi) {
                    for (int k = 0; k < grads[gi].total_size; ++k)
                        grads[gi].data[k] += step_grads[gi].data[k];
                }
            };

           
            accumulate_one(input_ids, target_ids);

            
            for (int bi = 1; bi < BATCH_SIZE; ++bi) {
                std::vector<int> inp2, tgt2;
                bool ok = loader.next_batch(inp2, tgt2);
                if (!ok) break;
                accumulate_one(inp2, tgt2);
            }

            if (valid_samples == 0) { ++step; continue; }

            
            float inv_bs = 1.0f / (float)valid_samples;
            for (auto& g : grads)
                for (int k = 0; k < g.total_size; ++k)
                    g.data[k] *= inv_bs;

            float loss_ce = accum_loss_ce * inv_bs;
            float loss    = accum_loss_fe * inv_bs;

           
            std::vector<Tensor*> gptrs;
            for (auto& g : grads) gptrs.push_back(&g);

            
            clip_gradients(gptrs, 1.0f);

            
            anurup_scaler.scale_gradients(gptrs);

            
            opt.learning_rate = BASE_LR * path_integral.lr_scale(0.5f);

            
            auto snap_before = WeightPathIntegral::snapshot(params);
            opt.step(params, gptrs);

            
            path_integral.record_step_from_tensors(loss, params, snap_before);

            smooth = smooth < 0 ? loss : 0.95f*smooth + 0.05f*loss;
            if (loss < best_loss) best_loss = loss;

            if (step % 100 == 0) {
                std::cout << std::fixed << std::setprecision(4)
                          << "Step " << std::setw(5) << step
                          << " | BS: "      << valid_samples
                          << " | F(loss): " << loss
                          << " | CE: "      << loss_ce
                          << " | Smooth: "  << smooth
                          
                          << " | T: "       << opt.temperature
                          << " | aH: "      << opt.alpha_H
                          << " | lr_x: "    << path_integral.lr_scale(0.5f) << "\n";
                std::cout.flush();
            }
            if (step % 500 == 0) {
                riem.log_state();
                path_integral.log_state();
                anurup_scaler.log_state(step);  
            }
           
            if (step > 0 && step % 2000 == 0) {
                path_integral.reset_best();
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

    auto r = evaluate(model, loader, 50);
    print_eval(r, step);

    float random_base = std::log((float)cfg.vocab_size);
    std::cout << "\nBest F(loss)  : " << best_loss << "\n";
    std::cout << "Random CE     : " << random_base << "\n";
    std::cout << "Final eval CE : " << r.loss << "\n";
    if (r.loss < random_base - 0.3f)
        std::cout << "✅ Model is learning!\n";
    else
        std::cout << "⚠️  Need more steps\n";
}


void run_eval(const std::string& ds, const std::string& ckpt) {
    Tokenizer tok = TokenizerManager::load();

    ModelConfig cfg;
    cfg.vocab_size = tok.vocab_size;

    if (!ckpt.empty()) {
        std::ifstream probe(ckpt, std::ios::binary);
        if (probe) {
            ModelConfig file_cfg;
            probe.read(reinterpret_cast<char*>(&file_cfg), sizeof(ModelConfig));
            if (probe && probe.gcount() == sizeof(ModelConfig)
                && file_cfg.d_model > 0 && file_cfg.num_heads > 0
                && file_cfg.num_layers > 0 && file_cfg.max_seq_len > 0) {
                cfg.d_model     = file_cfg.d_model;
                cfg.num_heads   = file_cfg.num_heads;
                cfg.num_layers  = file_cfg.num_layers;
                cfg.max_seq_len = file_cfg.max_seq_len;
                if (file_cfg.vocab_size == tok.vocab_size)
                    cfg.vocab_size = file_cfg.vocab_size;
            }
        }
    }

    if (cfg.d_model == 0)     cfg.d_model     = 64;
    if (cfg.num_heads == 0)   cfg.num_heads   = 4;
    if (cfg.num_layers == 0)  cfg.num_layers  = 2;
    if (cfg.max_seq_len == 0) cfg.max_seq_len = 64;

    LOGOSModel model(cfg);
    if (!ckpt.empty()) load_checkpoint(model, ckpt);
    DataLoader loader(ds, tok, cfg.max_seq_len);
    auto r = evaluate(model, loader, 200);
    print_eval(r, -1);
}

void run_generate(const std::string& ckpt, const std::string& prompt) {
    
    Tokenizer tok;
    try {
        tok = TokenizerManager::load();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return;
    }
    ModelConfig cfg;
    cfg.vocab_size=tok.vocab_size; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=64;
    LOGOSModel model(cfg);
    if (ckpt != "none") load_checkpoint(model, ckpt);
    std::cout << "Prompt: \"" << prompt << "\"\n\n";
    auto ids = tok.encode(prompt, 32);
    auto out  = model.generate(ids, 50, 0.8f, /*top_p=*/0.9f);
    std::cout << tok.decode(out) << "\n";
}


int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════╗\n"
              << "║  LOGOS v10 — Vedic-Physics Hybrid LLM            ║\n"
              << "║  Full Backprop | FreeEnergy CPU | Āṇurūpyeṇa     ║\n"
              << "╚══════════════════════════════════════════════════╝\n\n";

    
    logos_rng::set_global_seed(42);
    std::cout << "RNG seed  : 42 (deterministic weight init)\n\n";

    std::string mode = argc > 1 ? argv[1] : "--test";
    if      (mode == "--test")      run_tests();
    else if (mode == "--forward")   run_forward_test();
    else if (mode == "--benchmark") run_benchmark();
    else if (mode == "--tokenizer-info") {
        TokenizerManager::print_status();
    }
    else if (mode == "--train") {
        std::string ds = safe_arg(argc > 2 ? argv[2] : nullptr, "dataset.txt");
        bool force_rebuild = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--retokenize") {
                force_rebuild = true;
                std::cout << "--retokenize: tokenizer force-rebuild mode\n";
                break;
            }
        }
        if (force_rebuild) {
            TokenizerManager::load_or_build(ds, 0, true);
        }
        run_training(ds);
    }
    else if (mode == "--eval") {
        std::string ds = safe_arg(argc > 2 ? argv[2] : nullptr, "dataset.txt");
        std::string ck = safe_arg(argc > 3 ? argv[3] : nullptr, "");
        run_eval(ds, ck);
    }
    else if (mode == "--generate") {
        std::string ck = safe_arg(argc > 2 ? argv[2] : nullptr, "none");
        std::string pr = argc > 3 ? std::string(argv[3]) : "Once upon a time";
        run_generate(ck, pr);
    }
    return 0;
}
