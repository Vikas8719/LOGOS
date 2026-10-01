// ============================================================
#ifndef LOGOS_CHECKPOINT_CPP
#define LOGOS_CHECKPOINT_CPP

#include "../include/Checkpoint.hpp"
#include "../include/Tensor.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdint>

// Keep the legacy positional-embedding block in the binary layout so existing
// checkpoints remain readable. RoPE has no learned position tensor, so newly
// saved checkpoints write zeros here and loaders discard the block.
static void write_legacy_position_slot(std::ofstream& f, const ModelConfig& cfg) {
    constexpr size_t CHUNK_BYTES = 64 * 1024;
    const std::vector<char> zeros(CHUNK_BYTES, 0);
    size_t remaining = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    while (remaining > 0 && f) {
        const size_t chunk = std::min(remaining, CHUNK_BYTES);
        f.write(zeros.data(), static_cast<std::streamsize>(chunk));
        remaining -= chunk;
    }
}

static bool discard_legacy_position_slot(std::ifstream& f,
                                         const ModelConfig& cfg,
                                         const std::string& path,
                                         bool allow_truncated) {
    constexpr size_t CHUNK_BYTES = 64 * 1024;
    std::vector<char> buffer(CHUNK_BYTES);
    size_t remaining = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, CHUNK_BYTES);
        f.read(buffer.data(), static_cast<std::streamsize>(chunk));
        const size_t read = static_cast<size_t>(f.gcount());
        if (read != chunk) {
            std::cerr << "❌ Checkpoint truncated in legacy position block: " << path << "\n";
            if (allow_truncated) {
                f.clear();
                return true;
            }
            return false;
        }
        remaining -= read;
    }
    return true;
}

// ── Internal: validate cfg fields from file ──────────────────
static bool validate_checkpoint_cfg(const ModelConfig& cfg, const std::string& path) {
    auto chk = [&](const char* name, int val, int lo, int hi) -> bool {
        if (val < lo || val > hi) {
            std::cerr << "❌ Checkpoint corrupt [" << path << "]: "
                      << name << "=" << val
                      << " out of range (" << lo << ".." << hi << ")\n";
            return false;
        }
        return true;
    };
    if (!chk("vocab_size",  cfg.vocab_size,  1, CKPT_MAX_VOCAB))   return false;
    if (!chk("d_model",     cfg.d_model,     1, CKPT_MAX_D_MODEL)) return false;
    if (!chk("num_layers",  cfg.num_layers,  1, CKPT_MAX_LAYERS))  return false;
    if (!chk("num_heads",   cfg.num_heads,   1, CKPT_MAX_HEADS))   return false;
    if (!chk("max_seq_len", cfg.max_seq_len, 1, CKPT_MAX_SEQ_LEN)) return false;

    if (cfg.d_model % cfg.num_heads != 0) {
        std::cerr << "❌ Checkpoint corrupt [" << path << "]: "
                  << "d_model (" << cfg.d_model
                  << ") not divisible by num_heads (" << cfg.num_heads << ")\n";
        return false;
    }

    size_t est_params =
        (size_t)cfg.vocab_size * cfg.d_model +
        (size_t)cfg.max_seq_len * cfg.d_model +
        (size_t)cfg.d_model * cfg.vocab_size +
        (size_t)cfg.num_layers * cfg.d_model * cfg.d_model * 10;
    size_t est_mb = (est_params * sizeof(float)) / (1024*1024);
    if (est_mb > CKPT_MAX_FILE_MB) {
        std::cerr << "❌ Checkpoint cfg implies " << est_mb
                  << " MB model — exceeds max " << CKPT_MAX_FILE_MB << " MB\n";
        return false;
    }
    return true;
}

static bool cfgs_match(const ModelConfig& from_file,
                        const ModelConfig& model_cfg,
                        const std::string& path)
{
    bool ok = true;
    auto chk = [&](const char* name, int fval, int mval) {
        if (fval != mval) {
            std::cerr << "❌ Checkpoint cfg mismatch [" << path << "]: "
                      << name << " file=" << fval << " model=" << mval << "\n";
            ok = false;
        }
    };
    chk("vocab_size",  from_file.vocab_size,  model_cfg.vocab_size);
    chk("d_model",     from_file.d_model,     model_cfg.d_model);
    chk("num_layers",  from_file.num_layers,  model_cfg.num_layers);
    chk("num_heads",   from_file.num_heads,   model_cfg.num_heads);
    chk("max_seq_len", from_file.max_seq_len, model_cfg.max_seq_len);
    if (!ok) {
        std::cerr << "   → Use load_checkpoint_force() to load anyway (partial, risky)\n";
        std::cerr << "   → Or rebuild model with matching config before loading\n";
    }
    return ok;
}

static bool read_tensor_safe(std::ifstream& f, Tensor& t, const std::string& path) {
    auto bytes = static_cast<std::streamsize>(t.total_size * sizeof(float));
    f.read(reinterpret_cast<char*>(t.data.data()), bytes);
    if (f.gcount() != bytes) {
        std::cerr << "❌ Checkpoint truncated mid-read: " << path << "\n";
        return false;
    }
    return true;
}

static bool read_tensor_clamped(std::ifstream& f, Tensor& t, const std::string& path) {
    auto bytes = static_cast<std::streamsize>(t.total_size * sizeof(float));
    f.read(reinterpret_cast<char*>(t.data.data()), bytes);
    if (f.gcount() < bytes) {
        size_t got = static_cast<size_t>(f.gcount());
        std::fill(t.data.begin() + got/sizeof(float), t.data.end(), 0.0f);
        std::cerr << "⚠️  Partial tensor read (" << got << "/" << bytes
                  << " bytes) in: " << path << " — zero-padded\n";
    }
    return true;
}

// ============================================================
//  save_checkpoint
// ============================================================
bool save_checkpoint(const LOGOSModel& model,
                     const std::string& path,
                     int step)
{
    std::string full_path = path + "_step" + std::to_string(step) + ".bin";
    std::ofstream f(full_path, std::ios::binary);
    if (!f) {
        std::cerr << "❌ Cannot save checkpoint: " << full_path << "\n";
        return false;
    }

    f.write(reinterpret_cast<const char*>(&model.cfg), sizeof(ModelConfig));

    auto wt = [&](const Tensor& t) {
        f.write(reinterpret_cast<const char*>(t.data.data()),
                t.total_size * sizeof(float));
    };

    wt(model.embedding);
    write_legacy_position_slot(f, model.cfg);
    wt(model.lm_head);

    auto wv = [&](const std::vector<float>& v) {
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(float)));
    };

    for (const auto& block : model.layers) {
        for (const auto& head : block.mha.heads) {
            wt(head.W_Q); wt(head.W_K); wt(head.W_V); wt(head.W_O);
        }
        wt(block.mha.W_proj);
        wt(block.ffn.W1); wt(block.ffn.b1);
        wt(block.ffn.W2); wt(block.ffn.b2);
        wt(block.ln1.gamma); wt(block.ln1.beta);
        wt(block.ln2.gamma); wt(block.ln2.beta);
        wv(block.ln1.running_mean); wv(block.ln1.running_var);
        wv(block.ln2.running_mean); wv(block.ln2.running_var);
    }

    if (!f) {
        std::cerr << "❌ Write error: " << full_path << "\n";
        return false;
    }
    std::cout << "✅ Checkpoint saved: " << full_path
              << " (step=" << step << ")\n";
    return true;
}

// ============================================================
//  load_checkpoint — STRICT
// ============================================================
bool load_checkpoint(LOGOSModel& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "❌ Cannot load: " << path << "\n";
        return false;
    }

    ModelConfig file_cfg;
    f.read(reinterpret_cast<char*>(&file_cfg), sizeof(ModelConfig));
    if (!f || f.gcount() != sizeof(ModelConfig)) {
        std::cerr << "❌ Checkpoint truncated (can't read cfg): " << path << "\n";
        return false;
    }

    if (!validate_checkpoint_cfg(file_cfg, path)) return false;
    if (!cfgs_match(file_cfg, model.cfg, path)) {
        std::cerr << "❌ load_checkpoint() aborted — cfg mismatch prevents safe load\n";
        return false;
    }

    auto rt = [&](Tensor& t) -> bool {
        return read_tensor_safe(f, t, path);
    };

    if (!rt(model.embedding))     return false;
    if (!discard_legacy_position_slot(f, file_cfg, path, false)) return false;
    if (!rt(model.lm_head))       return false;

    auto rv = [&](std::vector<float>& v, const std::string& lbl) -> bool {
        auto bytes = static_cast<std::streamsize>(v.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(v.data()), bytes);
        if (f.gcount() != bytes) {
            std::fill(v.begin(), v.end(), 0.0f);
            f.clear();
            std::cerr << "⚠️  BN running stat '" << lbl
                      << "' missing in checkpoint (older format) — zero-initialised\n";
        }
        return true;
    };

    for (auto& block : model.layers) {
        for (auto& head : block.mha.heads) {
            if (!rt(head.W_Q)) return false;
            if (!rt(head.W_K)) return false;
            if (!rt(head.W_V)) return false;
            if (!rt(head.W_O)) return false;
        }
        if (!rt(block.mha.W_proj)) return false;
        if (!rt(block.ffn.W1))     return false;
        if (!rt(block.ffn.b1))     return false;
        if (!rt(block.ffn.W2))     return false;
        if (!rt(block.ffn.b2))     return false;
        if (!rt(block.ln1.gamma))  return false;
        if (!rt(block.ln1.beta))   return false;
        if (!rt(block.ln2.gamma))  return false;
        if (!rt(block.ln2.beta))   return false;
        rv(block.ln1.running_mean, "ln1.running_mean");
        rv(block.ln1.running_var,  "ln1.running_var");
        rv(block.ln2.running_mean, "ln2.running_mean");
        rv(block.ln2.running_var,  "ln2.running_var");
        block.ln1.initialized = true;
        block.ln2.initialized = true;
    }

    if (model.embedding.has_nan()) {
        std::cerr << "❌ Checkpoint corrupt — NaN in embeddings: " << path << "\n";
        return false;
    }

    std::cout << "✅ Checkpoint loaded: " << path
              << " (vocab=" << file_cfg.vocab_size
              << " d=" << file_cfg.d_model
              << " L=" << file_cfg.num_layers << ")\n";
    return true;
}

// ============================================================
//  load_checkpoint_force — PERMISSIVE
// ============================================================
bool load_checkpoint_force(LOGOSModel& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "❌ Cannot load: " << path << "\n";
        return false;
    }

    ModelConfig file_cfg;
    f.read(reinterpret_cast<char*>(&file_cfg), sizeof(ModelConfig));
    if (!f || f.gcount() != sizeof(ModelConfig)) {
        std::cerr << "❌ Checkpoint truncated (can't read cfg): " << path << "\n";
        return false;
    }

    if (!validate_checkpoint_cfg(file_cfg, path)) return false;

    if (file_cfg.vocab_size  != model.cfg.vocab_size  ||
        file_cfg.d_model     != model.cfg.d_model     ||
        file_cfg.num_layers  != model.cfg.num_layers)
    {
        std::cerr << "⚠️  load_checkpoint_force: cfg mismatch — loading anyway\n";
        std::cerr << "   File: vocab=" << file_cfg.vocab_size << " d=" << file_cfg.d_model
                  << " L=" << file_cfg.num_layers << "\n";
        std::cerr << "   Model: vocab=" << model.cfg.vocab_size << " d=" << model.cfg.d_model
                  << " L=" << model.cfg.num_layers << "\n";
    }

    auto rt = [&](Tensor& t) -> bool {
        return read_tensor_clamped(f, t, path);
    };

    if (!rt(model.embedding))     return false;
    if (!discard_legacy_position_slot(f, file_cfg, path, true)) return false;
    if (!rt(model.lm_head))       return false;

    for (auto& block : model.layers) {
        for (auto& head : block.mha.heads) {
            if (!rt(head.W_Q)) return false;
            if (!rt(head.W_K)) return false;
            if (!rt(head.W_V)) return false;
            if (!rt(head.W_O)) return false;
        }
        if (!rt(block.mha.W_proj)) return false;
        if (!rt(block.ffn.W1))     return false;
        if (!rt(block.ffn.b1))     return false;
        if (!rt(block.ffn.W2))     return false;
        if (!rt(block.ffn.b2))     return false;
        if (!rt(block.ln1.gamma))  return false;
        if (!rt(block.ln1.beta))   return false;
        if (!rt(block.ln2.gamma))  return false;
        if (!rt(block.ln2.beta))   return false;
    }

    std::cout << "⚠️  Checkpoint force-loaded: " << path << "\n";
    return true;
}

// ============================================================
//  save_optimizer_state  (v21-OPTSTATE)
//  Saves GPUSHMOpt velocity buffers + WeightPathIntegral EMA
//  File: {base_path}_step{step}.optstate
// ============================================================
static constexpr uint32_t OPTSTATE_MAGIC   = 0x4F505431u;  // "OPT1"
static constexpr uint32_t OPTSTATE_VERSION = 1u;

bool save_optimizer_state(const OptimizerState& state,
                          const std::string& base_path,
                          int step)
{
    std::string path = base_path + "_step" + std::to_string(step) + ".optstate";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "⚠️  Cannot save optimizer state: " << path
                  << " (training continues, next resume will be cold)\n";
        return false;
    }

    // Header
    uint32_t magic   = OPTSTATE_MAGIC;
    uint32_t version = OPTSTATE_VERSION;
    int64_t  n_params = static_cast<int64_t>(state.velocities.size());
    f.write(reinterpret_cast<const char*>(&magic),    sizeof(magic));
    f.write(reinterpret_cast<const char*>(&version),  sizeof(version));
    f.write(reinterpret_cast<const char*>(&n_params), sizeof(n_params));

    // Per-parameter velocity vectors
    for (const auto& vel : state.velocities) {
        uint32_t sz = static_cast<uint32_t>(vel.size());
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        f.write(reinterpret_cast<const char*>(vel.data()),
                static_cast<std::streamsize>(sz * sizeof(float)));
    }

    // Footer: WeightPathIntegral EMA state
    f.write(reinterpret_cast<const char*>(&state.ema_action),    sizeof(float));
    f.write(reinterpret_cast<const char*>(&state.log_amplitude), sizeof(float));
    f.write(reinterpret_cast<const char*>(&state.step_count),    sizeof(int64_t));

    if (!f) {
        std::cerr << "❌ Write error: " << path << "\n";
        return false;
    }
    std::cout << "✅ Optimizer state saved: " << path
              << " (" << n_params << " param groups)\n";
    return true;
}

// ============================================================
//  load_optimizer_state  (v21-OPTSTATE)
//  Returns false gracefully if file missing (older checkpoint compat)
// ============================================================
bool load_optimizer_state(OptimizerState& state,
                          const std::string& base_path,
                          int step)
{
    std::string path = base_path + "_step" + std::to_string(step) + ".optstate";
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        // Older checkpoint — no optstate file, cold optimizer start
        std::cout << "  ℹ️  No optimizer state found (" << path
                  << ") — optimizer starts cold (normal for step 60000 checkpoint)\n";
        return false;
    }

    // Read + validate header
    uint32_t magic = 0, version = 0;
    int64_t  n_params = 0;
    f.read(reinterpret_cast<char*>(&magic),    sizeof(magic));
    f.read(reinterpret_cast<char*>(&version),  sizeof(version));
    f.read(reinterpret_cast<char*>(&n_params), sizeof(n_params));

    if (magic != OPTSTATE_MAGIC) {
        std::cerr << "❌ Optimizer state corrupt (bad magic): " << path << "\n";
        return false;
    }
    if (version != OPTSTATE_VERSION) {
        std::cerr << "⚠️  Optimizer state version mismatch (file=" << version
                  << " expected=" << OPTSTATE_VERSION << ") — skipping\n";
        return false;
    }
    if (n_params <= 0 || n_params > 100000) {
        std::cerr << "❌ Optimizer state corrupt (n_params=" << n_params << ")\n";
        return false;
    }

    // Per-parameter velocity vectors
    state.velocities.resize(static_cast<size_t>(n_params));
    for (auto& vel : state.velocities) {
        uint32_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        if (!f || sz == 0 || sz > 100000000u) {
            std::cerr << "❌ Optimizer state corrupt (vel size=" << sz << ")\n";
            return false;
        }
        vel.resize(sz);
        f.read(reinterpret_cast<char*>(vel.data()),
               static_cast<std::streamsize>(sz * sizeof(float)));
        if (static_cast<size_t>(f.gcount()) != sz * sizeof(float)) {
            std::cerr << "❌ Optimizer state truncated mid-velocity\n";
            return false;
        }
    }

    // Footer: WeightPathIntegral EMA
    f.read(reinterpret_cast<char*>(&state.ema_action),    sizeof(float));
    f.read(reinterpret_cast<char*>(&state.log_amplitude), sizeof(float));
    f.read(reinterpret_cast<char*>(&state.step_count),    sizeof(int64_t));

    if (!f) {
        // Footer missing (very old optstate) — weights loaded, EMA cold
        std::cerr << "⚠️  Optimizer EMA footer missing — velocities loaded, EMA cold\n";
        state.ema_action = 0.0f; state.log_amplitude = 0.0f; state.step_count = 0;
    }

    std::cout << "✅ Optimizer state loaded: " << path
              << " (" << n_params << " param groups"
              << " | ema_action=" << state.ema_action
              << " | steps=" << state.step_count << ")\n";
    return true;
}

#endif
