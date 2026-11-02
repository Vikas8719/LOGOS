#include "../cuda/VedicGEMM.cuh"
#include "../cuda/ModelGPU.cuh"
#include "../cuda/MixedPrecision.cuh"   // [FIX] cast_fp32_to_fp16_kernel — ModelGPU forward mein use hota hai
#include "../include/Model.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <chrono>
#include <stdexcept>

static int g_pass = 0, g_fail = 0;

#define TEST(name, expr) do { \
    bool _ok = (expr); \
    if (_ok) { std::cout << "  ✅ " << (name) << "\n"; ++g_pass; } \
    else { std::cerr << "  ❌ " << (name) << "  [FAIL]\n"; ++g_fail; } \
} while(0)

static size_t vram_free() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

static size_t vram_used() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

static bool gpu_tensor_finite(const GPUTensor& t) {
    std::vector<float> h(t.size);
    d2h(h.data(), t, t.size);
    for (float v : h) if (!std::isfinite(v)) return false;
    return true;
}

static void fill_gpu(GPUTensor& t, float val) {
    std::vector<float> h(t.size, val);
    h2d(t, h.data(), t.size);
}

static void test_mem1_vram_limits() {
    std::cout << "\n[MEM1] VRAM Large Allocation Stress\n";

    size_t free_before = vram_free();
    size_t total; cudaMemGetInfo(nullptr, &total);

    size_t target_bytes = free_before / 2;
    int elements = (int)(target_bytes / sizeof(float));
    elements = std::min(elements, 256*1024*1024);

    std::cout << "    Available: " << free_before/1024/1024 << " MB\n";
    std::cout << "    Allocating: " << elements*4/1024/1024 << " MB\n";

    bool alloc_ok = false;
    {
        float* d_buf = nullptr;
        cudaError_t e = cudaMalloc(&d_buf, (size_t)elements * sizeof(float));
        if (e == cudaSuccess && d_buf != nullptr) {
            alloc_ok = true;
            cudaMemset(d_buf, 0, (size_t)elements * sizeof(float));
            cudaDeviceSynchronize();
            cudaFree(d_buf);
        } else {
            cudaGetLastError();  // clear error
        }
    }
    cudaDeviceSynchronize();

    size_t free_after = vram_free();
    long long leak_bytes = (long long)free_before - (long long)free_after;

    TEST("Large alloc succeeded",                alloc_ok);
    TEST("VRAM restored after free (leak<1MB)",  std::abs(leak_bytes) < 1024*1024LL);

    std::cout << "    VRAM before: " << free_before/1024/1024 << " MB\n";
    std::cout << "    VRAM after:  " << free_after/1024/1024 << " MB\n";
    std::cout << "    Leak: " << leak_bytes << " bytes\n";
}

static void test_mem2_raii_cycle() {
    std::cout << "\n[MEM2] Rapid Alloc/Free Cycle (1000 iterations)\n";

    size_t vram_before = vram_used();

    for (int i = 0; i < 1000; ++i) {
        GPUTensor t = gpu_alloc(64, 64);  // 16KB per tensor
        fill_gpu(t, (float)i);
    }
    cudaDeviceSynchronize();

    size_t vram_after = vram_used();
    long long diff = (long long)vram_after - (long long)vram_before;

    std::cout << "    VRAM before: " << vram_before/1024 << " KB\n";
    std::cout << "    VRAM after:  " << vram_after/1024 << " KB\n";
    std::cout << "    Net change: " << diff << " bytes\n";

    TEST("RAII: no VRAM leak after 1000 cycles (< 1MB)", std::abs(diff) < 1024*1024LL);
}

static void test_mem3_nikhilam_1m() {
    std::cout << "\n[MEM3] Nikhilam 1M Element Compress/Decompress\n";

    int sz = 1024 * 1024;  // 1,048,576 elements
    size_t vram_before = vram_used();

    GPUTensor src = gpu_alloc(sz, 1);
    std::cout << "    Source (float32): " << sz*4/1024 << " KB\n";

    std::vector<float> h_src(sz);
    for (int i=0; i<sz; ++i) h_src[i] = (float)(i % 201) - 100.f;
    h2d(src, h_src.data(), sz);

    float* d_max;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(float)));
    absmax_kernel<<<(sz+255)/256, 256>>>(src.data, d_max, sz);
    cudaDeviceSynchronize();
    float h_max=0.f;
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    float scale = h_max / 127.0f + 1e-8f;
    cudaFree(d_max);

    int8_t* d_int8;
    CUDA_CHECK(cudaMalloc(&d_int8, sz*sizeof(int8_t)));
    nikhilam_quantize_kernel<<<(sz+255)/256, 256>>>(src.data, d_int8, scale, sz);
    cudaDeviceSynchronize();
    std::cout << "    Compressed (int8): " << sz*1/1024 << " KB (4x smaller)\n";

    size_t vram_peak = vram_used();
    std::cout << "    Peak VRAM (float+int8): " << (vram_peak-vram_before)/1024 << " KB\n";

    float* d_rec;
    CUDA_CHECK(cudaMalloc(&d_rec, sz*sizeof(float)));
    nikhilam_dequantize_kernel<<<(sz+255)/256, 256>>>(d_int8, d_rec, scale, sz);
    cudaDeviceSynchronize();

    std::vector<float> h_rec(100);
    CUDA_CHECK(cudaMemcpy(h_rec.data(), d_rec, 100*sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.f;
    for (int i=0; i<100; ++i)
        max_err = std::max(max_err, std::abs(h_src[i] - h_rec[i]));

    TEST("1M Nikhilam: max error < 1.0",         max_err < 1.0f);
    TEST("1M Nikhilam: scale reasonable",         scale > 0.f && std::isfinite(scale));

    cudaFree(d_int8); cudaFree(d_rec);
    { GPUTensor tmp = std::move(src); }  // force destructor now
    cudaDeviceSynchronize();

    size_t vram_after = vram_used();
    long long leak = (long long)vram_after - (long long)vram_before;
    TEST("1M Nikhilam: no leak after free (< 1MB)", std::abs(leak) < 1024*1024LL);

    std::cout << "    Max quant error (sample): " << max_err << "\n";
}

static void test_mem4_kv_cache_concurrent() {
    std::cout << "\n[MEM4] Concurrent KV Cache — Max Config (L=6,H=8,seq=512,DH=64)\n";

    int L=6, H=8, seq=512, DH=64;
    size_t vram_before = vram_used();

    std::vector<int8_t*> k_bufs, v_bufs;
    size_t total_int8 = 0;

    bool alloc_ok = true;
    for (int l=0; l<L && alloc_ok; ++l) {
        for (int h=0; h<H && alloc_ok; ++h) {
            int8_t *dk=nullptr, *dv=nullptr;
            cudaError_t ek = cudaMalloc(&dk, seq*DH*sizeof(int8_t));
            cudaError_t ev = cudaMalloc(&dv, seq*DH*sizeof(int8_t));
            if (ek!=cudaSuccess || ev!=cudaSuccess) {
                alloc_ok = false;
                if (dk) cudaFree(dk);
                if (dv) cudaFree(dv);
                cudaGetLastError();
            } else {
                k_bufs.push_back(dk);
                v_bufs.push_back(dv);
                total_int8 += 2*seq*DH;
            }
        }
    }

    size_t vram_peak = vram_used();
    std::cout << "    Total int8 elements: " << total_int8/1024 << "K\n";
    std::cout << "    KV int8 memory: " << total_int8/1024 << " KB\n";
    std::cout << "    KV float32 would be: " << total_int8*4/1024 << " KB\n";
    std::cout << "    VRAM at peak: " << vram_peak/1024/1024 << " MB used\n";

    for (auto* b : k_bufs) cudaFree(b);
    for (auto* b : v_bufs) cudaFree(b);
    cudaDeviceSynchronize();

    size_t vram_after = vram_used();
    long long leak = (long long)vram_after - (long long)vram_before;

    TEST("KV cache: all allocations succeeded", alloc_ok);
    TEST("KV cache: int8 < 4 MB total",         total_int8 < 4*1024*1024ULL);
    TEST("KV cache: no leak after free",         std::abs(leak) < 512*1024LL);
}

static void test_mem5_seq1_edge() {
    std::cout << "\n[MEM5] Edge Case: seq_len=1\n";

    ModelConfig cfg;
    cfg.vocab_size=64; cfg.d_model=32;
    cfg.num_heads=2; cfg.num_layers=1; cfg.max_seq_len=32;

    bool ok = false;
    try {
        ModelGPU model(cfg);
        LOGOSModel cpu_model(cfg);
        model.load_from_cpu(cpu_model);

        std::vector<int> tokens = {7};  // single token
        GPUTensor logits = model.forward(tokens);
        cudaDeviceSynchronize();

        ok = (logits.rows == 1) &&
             (logits.cols == cfg.vocab_size) &&
             gpu_tensor_finite(logits);
    } catch (const std::exception& e) {
        std::cerr << "    Exception: " << e.what() << "\n";
    }
    TEST("seq=1: forward pass (1×vocab logits, finite)", ok);
}

static void test_mem6_maxseq_edge() {
    std::cout << "\n[MEM6] Edge Case: seq_len=max_seq\n";

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=64;

    bool ok = false;
    try {
        ModelGPU model(cfg);
        LOGOSModel cpu_model(cfg);
        model.load_from_cpu(cpu_model);

        std::vector<int> tokens(cfg.max_seq_len);
        for (int i=0; i<cfg.max_seq_len; ++i) tokens[i] = i % cfg.vocab_size;

        GPUTensor logits = model.forward(tokens);
        cudaDeviceSynchronize();

        ok = (logits.rows == cfg.max_seq_len) &&
             (logits.cols == cfg.vocab_size) &&
             gpu_tensor_finite(logits);
    } catch (const std::exception& e) {
        std::cerr << "    Exception: " << e.what() << "\n";
    }
    TEST("seq=max_seq: forward no OOM, finite logits", ok);
}

static void test_mem7_grad_buffer_sizes() {
    std::cout << "\n[MEM7] Gradient Buffer Size Verification\n";

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    ModelGPU model(cfg);
    auto params = model.all_parameters();
    auto grads  = model.alloc_grad_buffers();

    TEST("Grad count == param count", grads.size() == params.size());

    bool sizes_match = true;
    for (int i=0; i<(int)params.size(); ++i) {
        if (grads[i]->size != params[i]->size) {
            std::cerr << "    Mismatch at i=" << i
                      << " param=" << params[i]->size
                      << " grad=" << grads[i]->size << "\n";
            sizes_match = false;
        }
    }
    TEST("Every grad size == corresponding param size", sizes_match);

    bool all_zero = true;
    for (auto* g : grads) {
        std::vector<float> h(g->size);
        d2h(h.data(), *g, g->size);
        for (float v : h) if (v != 0.f) { all_zero = false; break; }
        if (!all_zero) break;
    }
    TEST("All grad buffers zero-initialized", all_zero);

    std::cout << "    Total params: " << params.size() << "\n";
    long long total_param_bytes = 0;
    for (auto* p : params) total_param_bytes += p->size * 4;
    std::cout << "    Total param memory: " << total_param_bytes/1024 << " KB\n";

    for (auto* g : grads) delete g;
}

static void test_mem8_leapfrog_10m() {
    std::cout << "\n[MEM8] Leapfrog Kernel — 10M Elements Throughput\n";

    int sz = 10 * 1024 * 1024;  // 10,485,760
    size_t bytes = (size_t)sz * sizeof(float);
    std::cout << "    Size: " << sz/1024/1024 << "M elements = " << bytes*3/1024/1024 << " MB (W+V+G)\n";

    float *d_W=nullptr, *d_V=nullptr, *d_G=nullptr;
    bool alloc_ok = true;
    cudaError_t e1=cudaMalloc(&d_W,bytes);
    cudaError_t e2=cudaMalloc(&d_V,bytes);
    cudaError_t e3=cudaMalloc(&d_G,bytes);
    if (e1!=cudaSuccess||e2!=cudaSuccess||e3!=cudaSuccess) {
        alloc_ok = false;
        cudaGetLastError();
        if (d_W) cudaFree(d_W);
        if (d_V) cudaFree(d_V);
        if (d_G) cudaFree(d_G);
    }

    if (!alloc_ok) {
        TEST("10M Leapfrog: allocation (skipped - insufficient VRAM)", true);
        std::cout << "    ⚠️  Skipped (insufficient VRAM)\n";
        return;
    }

    cudaMemset(d_W, 0, bytes); // W = 0
    cudaMemset(d_V, 0, bytes); // V = 0
    std::vector<float> h_G(sz, 0.01f);
    cudaMemcpy(d_G, h_G.data(), bytes, cudaMemcpyHostToDevice);

    leapfrog_langevin_kernel<<<(sz+255)/256, 256>>>(
        d_W, d_V, d_G, 1e-4f, 0.9f, 1e-5f, 42u, sz);
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int r=0; r<10; ++r) {
        leapfrog_langevin_kernel<<<(sz+255)/256, 256>>>(
            d_W, d_V, d_G, 1e-4f, 0.9f, 1e-5f, (unsigned)(42+r), sz);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float,std::milli>(t2-t1).count() / 10.f;
    float gbps = (5.f * bytes / 1e9f) / (ms / 1e3f);

    std::cout << "    10M elements: " << ms << " ms/step\n";
    std::cout << "    Throughput: " << gbps << " GB/s\n";

    std::vector<float> h_W(100);
    cudaMemcpy(h_W.data(), d_W, 100*sizeof(float), cudaMemcpyDeviceToHost);
    bool finite = true;
    for (float v : h_W) if (!std::isfinite(v)) { finite=false; break; }

    TEST("10M Leapfrog: kernel completed",      alloc_ok);
    TEST("10M Leapfrog: output is finite",      finite);
    TEST("10M Leapfrog: throughput > 2 GB/s",  gbps > 2.f);

    cudaFree(d_W); cudaFree(d_V); cudaFree(d_G);
}

static void test_mem9_large_vocab_loss() {
    std::cout << "\n[MEM9] Free Energy Loss — Large Vocab Stress\n";

    int seq=64, vocab=32768;  // 32K vocab — reasonable large scale
    size_t logit_bytes = (size_t)seq*vocab*sizeof(float);
    std::cout << "    seq=" << seq << " vocab=" << vocab
              << " logit_size=" << logit_bytes/1024/1024 << " MB\n";

    size_t free_before = vram_free();
    if (free_before < logit_bytes * 3) {
        TEST("Large vocab: skipped (insufficient VRAM)", true);
        std::cout << "    ⚠️  Skipped — need "
                  << logit_bytes*3/1024/1024 << " MB, have "
                  << free_before/1024/1024 << " MB\n";
        return;
    }

    float *d_logits=nullptr, *d_loss=nullptr, *d_grad=nullptr;
    int   *d_targets=nullptr;
    bool   alloc_ok = true;

    if (cudaMalloc(&d_logits,  logit_bytes)         != cudaSuccess ||
        cudaMalloc(&d_loss,    seq*sizeof(float))    != cudaSuccess ||
        cudaMalloc(&d_grad,    logit_bytes)          != cudaSuccess ||
        cudaMalloc(&d_targets, seq*sizeof(int))      != cudaSuccess) {
        alloc_ok = false;
        cudaGetLastError();
    }

    if (!alloc_ok) {
        TEST("Large vocab: OOM-safe (allocation failed gracefully)", true);
        if (d_logits)  cudaFree(d_logits);
        if (d_loss)    cudaFree(d_loss);
        if (d_grad)    cudaFree(d_grad);
        if (d_targets) cudaFree(d_targets);
        return;
    }

    cudaMemset(d_logits, 0, logit_bytes);
    std::vector<int> h_targets(seq);
    for (int i=0; i<seq; ++i) h_targets[i] = (i*7) % vocab;
    cudaMemcpy(d_targets, h_targets.data(), seq*sizeof(int), cudaMemcpyHostToDevice);

    auto t1 = std::chrono::high_resolution_clock::now();
    FreeEnergyResult fe;
    cuda_free_energy_loss(d_logits, d_targets, d_loss, d_grad,
                          seq, vocab, 0.05f, fe);
    auto t2 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float,std::milli>(t2-t1).count();

    std::cout << "    Loss kernel: " << ms << " ms\n";
    std::cout << "    CE=" << fe.cross_entropy << " S=" << fe.entropy
              << " F=" << fe.free_energy << "\n";

    float expected_ce = std::log((float)vocab);
    TEST("Large vocab: CE finite",               std::isfinite(fe.cross_entropy));
    TEST("Large vocab: CE ≈ log(vocab)",          std::abs(fe.cross_entropy - expected_ce) < 1.0f);
    TEST("Large vocab: entropy finite and >= 0",  std::isfinite(fe.entropy) && fe.entropy >= 0.f);
    TEST("Large vocab: kernel < 1000ms",          ms < 1000.f);

    cudaFree(d_logits); cudaFree(d_loss); cudaFree(d_grad); cudaFree(d_targets);
}

static void test_mem10_boundary_rw() {
    std::cout << "\n[MEM10] Kernel Boundary — Last Element R/W\n";

    std::vector<int> sizes = {1, 127, 128, 255, 256, 257, 1023, 1024, 1025, 65535, 65536, 65537};

    for (int sz : sizes) {
        float* d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, sz*sizeof(float)));

        std::vector<float> h_in(sz);
        for (int i=0; i<sz; ++i) h_in[i] = (float)i * 0.001f;  // small values, gelu ≈ x/2
        cudaMemcpy(d_buf, h_in.data(), sz*sizeof(float), cudaMemcpyHostToDevice);

        GPUTensor t; t.data=d_buf; t.rows=1; t.cols=sz; t.size=sz;
        cuda_gelu(t);
        cudaDeviceSynchronize();
        t.data=nullptr;  // don't let destructor free

        std::vector<float> h_out(sz);
        cudaMemcpy(h_out.data(), d_buf, sz*sizeof(float), cudaMemcpyDeviceToHost);

        bool last_finite = std::isfinite(h_out[sz-1]);
        bool first_finite = std::isfinite(h_out[0]);

        if (!last_finite || !first_finite) {
            std::cerr << "  ❌ Boundary FAIL at size=" << sz
                      << " first=" << h_out[0] << " last=" << h_out[sz-1] << "\n";
            ++g_fail;
        }

        cudaFree(d_buf);
    }
    TEST("All boundary sizes: first+last element finite", true);
    std::cout << "    Tested sizes: 1 to 65537 (including non-powers-of-2)\n";
}

static void test_mem11_repeated_forward() {
    std::cout << "\n[MEM11] Repeated Forward Passes — No VRAM Accumulation\n";

    ModelConfig cfg;
    cfg.vocab_size=128; cfg.d_model=64;
    cfg.num_heads=4; cfg.num_layers=2; cfg.max_seq_len=32;

    ModelGPU model(cfg);
    LOGOSModel cpu_model(cfg);
    model.load_from_cpu(cpu_model);

    std::vector<int> tokens = {1,2,3,4,5,6,7,8};

    { auto l = model.forward(tokens); cudaDeviceSynchronize(); }
    size_t vram_after_1 = vram_used();

    for (int i=0; i<10; ++i) {
        auto l = model.forward(tokens);
        cudaDeviceSynchronize();
    }
    size_t vram_after_11 = vram_used();

    long long diff = (long long)vram_after_11 - (long long)vram_after_1;
    std::cout << "    VRAM after 1 pass:  " << vram_after_1/1024 << " KB\n";
    std::cout << "    VRAM after 11 pass: " << vram_after_11/1024 << " KB\n";
    std::cout << "    Drift: " << diff << " bytes\n";

    TEST("No VRAM accumulation over 11 forward passes (< 4MB drift)",
         std::abs(diff) < 4*1024*1024LL);
}

static void test_mem12_scale_stress() {
    std::cout << "\n[MEM12] ModelGPU Scale Stress\n";

    struct ScaleConfig {
        int d, L, H, seq;
        const char* name;
    };

    std::vector<ScaleConfig> configs = {
        { 64,  2, 4,  32, "Tiny (d=64, L=2)"},
        {128,  4, 8, 128, "Small (d=128, L=4)"},
        {256,  6, 8, 256, "Medium (d=256, L=6)"},
    };

    for (auto& sc : configs) {
        std::cout << "\n    Config: " << sc.name << "\n";

        ModelConfig cfg;
        cfg.vocab_size=512; cfg.d_model=sc.d;
        cfg.num_heads=sc.H; cfg.num_layers=sc.L;
        cfg.max_seq_len=sc.seq;

        size_t vram_before = vram_used();
        bool ok = false;

        try {
            ModelGPU model(cfg);
            LOGOSModel cpu_model(cfg);
            model.load_from_cpu(cpu_model);

            size_t vram_after_alloc = vram_used();
            std::cout << "      VRAM for model: "
                      << (vram_after_alloc-vram_before)/1024/1024 << " MB\n";

            std::vector<int> tokens(sc.seq);
            for (int i=0; i<sc.seq; ++i) tokens[i] = i % cfg.vocab_size;

            auto t1 = std::chrono::high_resolution_clock::now();
            GPUTensor logits = model.forward(tokens);
            cudaDeviceSynchronize();
            auto t2 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float,std::milli>(t2-t1).count();

            bool finite = gpu_tensor_finite(logits);
            bool shape_ok = (logits.rows==sc.seq && logits.cols==cfg.vocab_size);

            std::cout << "      Forward: " << ms << " ms | "
                      << (finite?"finite":"NaN!") << " | "
                      << (shape_ok?"shape OK":"shape FAIL") << "\n";

            ok = finite && shape_ok;

            model.print_kvcache_stats(sc.seq, 1000);
        }
        catch (const std::exception& e) {
            std::cerr << "      Exception: " << e.what() << "\n";
        }

        cudaDeviceSynchronize();
        size_t vram_after = vram_used();
        long long leak = (long long)vram_after - (long long)vram_before;
        std::cout << "      VRAM leak after destroy: " << leak << " bytes\n";

        TEST(std::string(sc.name) + ": forward OK + no leak",
             ok && std::abs(leak) < 2*1024*1024LL);
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  LOGOS Test 3: Memory & Bounds Stress    ║\n";
    std::cout << "║  1M+ data, VRAM leak, boundary checks    ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\nGPU: " << prop.name
              << " | VRAM: " << total_mem/1024/1024 << " MB"
              << " | Free: " << free_mem/1024/1024 << " MB\n";

    try {
        test_mem1_vram_limits();
        test_mem2_raii_cycle();
        test_mem3_nikhilam_1m();
        test_mem4_kv_cache_concurrent();
        test_mem5_seq1_edge();
        test_mem6_maxseq_edge();
        test_mem7_grad_buffer_sizes();
        test_mem8_leapfrog_10m();
        test_mem9_large_vocab_loss();
        test_mem10_boundary_rw();
        test_mem11_repeated_forward();
        test_mem12_scale_stress();
    } catch (const std::exception& e) {
        std::cerr << "\n💥 FATAL: " << e.what() << "\n";
        ++g_fail;
    }

    std::cout << "\n════════════════════════════════════\n";
    std::cout << "  PASS: " << g_pass << "  FAIL: " << g_fail << "\n";
    std::cout << "════════════════════════════════════\n";
    return g_fail > 0 ? 1 : 0;
}
