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
// ============================================================
#include "Model.hpp"
#include <string>

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
