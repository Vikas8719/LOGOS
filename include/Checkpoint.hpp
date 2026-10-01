#pragma once
// ============================================================
//  LOGOS — Checkpoint.hpp
//
//  BUG 4 FIX: Config mismatch ab HARD ERROR hai (silent overflow band)
//    Pehle: cfg read karta tha file se, validate karta tha,
//           lekin phir model.embedding (jo model ki current size pe tha)
//           use karta tha cfg ki jagah. Agar d=256,L=6 checkpoint ko
//           d=64,L=2 model mein load karo → silent buffer overflow/underflow.
//           Crash nahi hota, galat data padha jaata hai — production mein
//           silently wrong model weights!
//    Ab:    load_checkpoint() REQUIRES cfg match. Mismatch pe:
//           - ERROR print karta hai exact mismatch details ke saath
//           - false return karta hai (load abort)
//           - Caller ko manually model rebuild karna hoga matching cfg se
//           Use load_checkpoint_force() agar aap deliberately mismatch allow karna chahte ho
//           (e.g. vocab resize after tokenizer update — partial load).
//
//  BUG 3 FIX (v5 se carry forward): Validate cfg before ANY tensor access
//    Malicious .bin → heap overflow → arbitrary code execution
//    Ab: strict bounds check before allocation/read
//
//  BUG 13 FIX (v5 se carry forward): Single definition pattern
//    Declarations only here, definitions in Checkpoint.cpp
//
//  v21-OPTSTATE: Optimizer state save/load
//    Problem: GPUSHMOpt velocity (momentum) GPU buffers save nahi hote the
//             → resume ke baad optimizer zero se shuru → CE 5.x se 8.x pe wapis
//    Fix:     .optstate file alongside .bin checkpoint
//             velocities + WeightPathIntegral EMA save/load
// ============================================================
#include "Model.hpp"
#include <string>
#include <vector>
#include <cstdint>

// ── Validation bounds (sane max values) ──────────────────────
static constexpr int    CKPT_MAX_VOCAB    = 100000;
static constexpr int    CKPT_MAX_D_MODEL  = 8192;
static constexpr int    CKPT_MAX_LAYERS   = 128;
static constexpr int    CKPT_MAX_HEADS    = 256;
static constexpr int    CKPT_MAX_SEQ_LEN  = 32768;
static constexpr size_t CKPT_MAX_FILE_MB  = 10000;  // 10GB

// ── Save ──────────────────────────────────────────────────────
// Returns true on success
bool save_checkpoint(const LOGOSModel& model,
                     const std::string& path,
                     int step);

// ── Load (strict — requires cfg match) ───────────────────────
// BUG 4 FIX: Returns false if cfg in file != model.cfg
// Caller must ensure model is built with matching config before loading.
// Error message prints exact mismatch for diagnosis.
bool load_checkpoint(LOGOSModel& model, const std::string& path);

// ── Load (force — allows cfg mismatch, partial load) ─────────
// Use ONLY when you know what you're doing (e.g. vocab resize).
// Tensors read up to min(file_size, model_size) — no overflow.
// May produce wrong results if architectures differ significantly.
bool load_checkpoint_force(LOGOSModel& model, const std::string& path);

// ── Optimizer State Save/Load (v21-OPTSTATE) ─────────────────
// GPUSHMOpt velocity (momentum) GPU buffers + WeightPathIntegral EMA
// Saved as separate .optstate file alongside .bin checkpoint
//
// File format:
//   [magic=0x4F505431 (4B)][version=1 (4B)][n_params (8B)]
//   per-param: [size (4B)][float*size velocity data]
//   footer:    [ema_action (4B)][log_amplitude (4B)][step_count (8B)]
//
// Kaggle workflow:
//   Save → logos_gpu_ckpt_step60000.bin
//          logos_gpu_ckpt_step60000.optstate   ← NEW
//   Load → Add BOTH files to Kaggle dataset input
//          LOGOS_CKPT=/path/logos_gpu_ckpt_step60000.bin  (same as before)
//          .optstate auto-detected from same path
//
// Backward compat:
//   .optstate missing → load_optimizer_state() returns false gracefully
//   → optimizer starts from zeros (old behaviour, no crash)
struct OptimizerState {
    // Per-parameter velocity vectors (CPU copy of GPU d_velocity buffers)
    std::vector<std::vector<float>> velocities;
    // WeightPathIntegral EMA fields
    float   ema_action    = 0.0f;
    float   log_amplitude = 0.0f;
    int64_t step_count    = 0;
};

// Save: writes logos_gpu_ckpt_step{step}.optstate
bool save_optimizer_state(const OptimizerState& state,
                          const std::string& base_path,
                          int step);

// Load: reads logos_gpu_ckpt_step{step}.optstate
// Returns false gracefully if file not found (older checkpoint compat)
bool load_optimizer_state(OptimizerState& state,
                          const std::string& base_path,
                          int step);
