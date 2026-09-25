// ============================================================
//  LOGOS — Checkpoint.cpp
//
//  BUG 4 FIX: load_checkpoint() ab cfg mismatch pe ABORT karta hai
//    Pehle: mismatch warning print karta tha, phir bhi load karta tha
//           → d=256 checkpoint ko d=64 model mein load = silent overflow
//           → model ke embedding tensor sirf 64*vocab floats allocate hain
//             lekin 256*vocab floats padhe jaate hain → heap corruption
//    Ab:    strict match required. False return on mismatch.
//           load_checkpoint_force() available for deliberate partial loads.
//
//  BUG 3 FIX (carry forward): validate cfg bounds before any read
//  BUG 13 FIX (carry forward): single definition, declarations in .hpp
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
// BUG 3 FIX: All fields checked before ANY allocation or tensor access
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

// ── Internal: check cfg match between file and model ─────────
// BUG 4 FIX: This is the NEW strict check.
// Returns true only if all architecture params match exactly.
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

// ── Internal: read tensor with safe size check ───────────────
static bool read_tensor_safe(std::ifstream& f, Tensor& t, const std::string& path) {
    auto bytes = static_cast<std::streamsize>(t.total_size * sizeof(float));
    f.read(reinterpret_cast<char*>(t.data.data()), bytes);
    if (f.gcount() != bytes) {
        std::cerr << "❌ Checkpoint truncated mid-read: " << path << "\n";
        return false;
    }
    return true;
}

// ── Internal: read tensor with clamped size (force mode) ─────
// Reads min(file_available, tensor_size) — prevents overflow
static bool read_tensor_clamped(std::ifstream& f, Tensor& t, const std::string& path) {
    // In force mode we can't know exact file tensor size without re-computing from
    // file cfg. Best effort: read what model expects, skip if file is shorter.
    auto bytes = static_cast<std::streamsize>(t.total_size * sizeof(float));
    f.read(reinterpret_cast<char*>(t.data.data()), bytes);
    if (f.gcount() < bytes) {
        // Partial read — zero-fill remainder
        size_t got = static_cast<size_t>(f.gcount());
        size_t remaining = static_cast<size_t>(bytes) - got;
        std::fill(t.data.begin() + got/sizeof(float), t.data.end(), 0.0f);
        std::cerr << "⚠️  Partial tensor read (" << got << "/" << bytes
                  << " bytes) in: " << path << " — zero-padded\n";
        // Don't fail — force mode tolerates partial reads
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

    // Write config (used to validate on load)
    f.write(reinterpret_cast<const char*>(&model.cfg), sizeof(ModelConfig));

    auto wt = [&](const Tensor& t) {
        f.write(reinterpret_cast<const char*>(t.data.data()),
                t.total_size * sizeof(float));
    };

    wt(model.embedding);
    write_legacy_position_slot(f, model.cfg);
    wt(model.lm_head);

    for (const auto& block : model.layers) {
        for (const auto& head : block.mha.heads) {
            wt(head.W_Q); wt(head.W_K); wt(head.W_V); wt(head.W_O);
        }
        wt(block.mha.W_proj);
        wt(block.ffn.W1); wt(block.ffn.b1);
        wt(block.ffn.W2); wt(block.ffn.b2);
        wt(block.ln1.gamma); wt(block.ln1.beta);
        wt(block.ln2.gamma); wt(block.ln2.beta);
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
//  load_checkpoint — STRICT (BUG 4 FIX)
//  Requires model cfg to exactly match file cfg.
//  Returns false on any mismatch — no silent partial loads.
// ============================================================
bool load_checkpoint(LOGOSModel& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "❌ Cannot load: " << path << "\n";
        return false;
    }

    // Step 1: Read cfg from file
    ModelConfig file_cfg;
    f.read(reinterpret_cast<char*>(&file_cfg), sizeof(ModelConfig));
    if (!f || f.gcount() != sizeof(ModelConfig)) {
        std::cerr << "❌ Checkpoint truncated (can't read cfg): " << path << "\n";
        return false;
    }

    // Step 2: Validate cfg fields (BUG 3 FIX — bounds check)
    if (!validate_checkpoint_cfg(file_cfg, path)) return false;

    // Step 3: BUG 4 FIX — STRICT cfg match check
    // Pehle: mismatch pe warning tha, phir bhi load hota tha → silent overflow
    // Ab:    mismatch pe false return — caller must fix their model config
    if (!cfgs_match(file_cfg, model.cfg, path)) {
        std::cerr << "❌ load_checkpoint() aborted — cfg mismatch prevents safe load\n";
        return false;  // Hard abort — no silent buffer overflow
    }

    // Step 4: Read tensors — sizes are safe (cfg verified to match)
    auto rt = [&](Tensor& t) -> bool {
        return read_tensor_safe(f, t, path);
    };

    if (!rt(model.embedding))     return false;
    if (!discard_legacy_position_slot(f, file_cfg, path, false)) return false;
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
//  load_checkpoint_force — PERMISSIVE (use with caution)
//  Allows cfg mismatch — reads into existing model tensors.
//  Tensors read safely (no overflow) — may zero-pad if file is shorter.
//  Use for: vocab resize, architecture experiments, partial weight loading.
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

    // Warn about mismatch but proceed
    if (file_cfg.vocab_size  != model.cfg.vocab_size  ||
        file_cfg.d_model     != model.cfg.d_model     ||
        file_cfg.num_layers  != model.cfg.num_layers)
    {
        std::cerr << "⚠️  load_checkpoint_force: cfg mismatch — loading anyway\n";
        std::cerr << "   File: vocab=" << file_cfg.vocab_size << " d=" << file_cfg.d_model
                  << " L=" << file_cfg.num_layers << "\n";
        std::cerr << "   Model: vocab=" << model.cfg.vocab_size << " d=" << model.cfg.d_model
                  << " L=" << model.cfg.num_layers << "\n";
        std::cerr << "   Result may be incorrect — use only for experimentation\n";
    }

    // Read with clamping — no overflow possible
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

#endif
