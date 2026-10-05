#pragma once
// ============================================================
//  LOGOS — TrainingState.hpp  [v29-FULLRESUME]
//
//  Problem: Resume pe sirf weights + optimizer velocities restore hote the.
//  Ye sab RESET ho jaate the:
//    ❌ Data position  → dataset shuru se padhna
//    ❌ prev_train_ce  → overfit detection broken
//    ❌ prev_val_ce    → overfit streak reset
//    ❌ best_loss      → wrong "best" checkpoint after resume
//    ❌ loss_scaler    → AMP scale reset to 65536
//    ❌ gnorm history  → grad clipping context lost
//
//  Fix: TrainingState struct — har 1000 steps pe .trainstate file mein
//  save hoti hai alongside .bin aur .optstate. Resume pe load karke
//  sab kuch wahan se shuru hota hai jahan chhooda tha.
//
//  File format:
//    [magic=0x54524E31 (4B)] [version=2 (4B)]
//    [step (8B)] [best_loss (4B)]
//    [prev_train_ce (4B)] [prev_val_ce (4B)]
//    [overfit_streak (4B)]
//    [amp_scale (4B)] [amp_window (4B)] [amp_overflows (4B)]
//    [amp_scale_ups (4B)] [amp_scale_downs (4B)]
//    [loader_shard (4B)] [loader_byte_pos (8B)]
//    [loader_tokens_seen (8B)] [loader_epoch (4B)]
//    [val_loader_byte_pos (8B)]
//    [gnorm_ema (4B)]           ← EMA of recent gradient norms
//    [train_ce_ema (4B)]        ← EMA of recent train CE (for smooth display)
//    [val_ce_ema (4B)]          ← EMA of recent val CE
// ============================================================

#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cmath>

static constexpr uint32_t TRAINSTATE_MAGIC   = 0x54524E31u; // "TRN1"
static constexpr uint32_t TRAINSTATE_VERSION = 2u;

struct TrainingState {
    // ── Step & loss tracking ─────────────────────────────────
    int64_t step         = 0;
    float   best_loss    = 999.f;

    // ── Overfit detection ────────────────────────────────────
    float   prev_train_ce    = 999.f;
    float   prev_val_ce      = 999.f;
    int     overfit_streak   = 0;

    // ── AMP loss scaler ──────────────────────────────────────
    float   amp_scale        = 65536.f;
    int     amp_window       = 0;      // steps_since_last_overflow
    int     amp_overflows    = 0;
    int     amp_scale_ups    = 0;
    int     amp_scale_downs  = 0;

    // ── Data loader position ─────────────────────────────────
    int     loader_shard        = 0;
    int64_t loader_byte_pos     = 0;   // bytes consumed in current shard
    int64_t loader_tokens_seen  = 0;
    int     loader_epoch        = 0;
    int64_t val_loader_byte_pos = 0;   // val loader byte position

    // ── Smooth metrics (EMA) ─────────────────────────────────
    float   gnorm_ema    = 0.f;
    float   train_ce_ema = 999.f;
    float   val_ce_ema   = 999.f;

    // ── Save ─────────────────────────────────────────────────
    bool save(const std::string& base_path, int step_n) const {
        std::string path = base_path + "_step" + std::to_string(step_n) + ".trainstate";
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "⚠️  Cannot save training state: " << path << "\n";
            return false;
        }

        uint32_t magic   = TRAINSTATE_MAGIC;
        uint32_t version = TRAINSTATE_VERSION;
        f.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Step & loss
        f.write(reinterpret_cast<const char*>(&step),      sizeof(step));
        f.write(reinterpret_cast<const char*>(&best_loss), sizeof(best_loss));

        // Overfit detection
        f.write(reinterpret_cast<const char*>(&prev_train_ce),  sizeof(prev_train_ce));
        f.write(reinterpret_cast<const char*>(&prev_val_ce),    sizeof(prev_val_ce));
        f.write(reinterpret_cast<const char*>(&overfit_streak), sizeof(overfit_streak));

        // AMP scaler
        f.write(reinterpret_cast<const char*>(&amp_scale),       sizeof(amp_scale));
        f.write(reinterpret_cast<const char*>(&amp_window),      sizeof(amp_window));
        f.write(reinterpret_cast<const char*>(&amp_overflows),   sizeof(amp_overflows));
        f.write(reinterpret_cast<const char*>(&amp_scale_ups),   sizeof(amp_scale_ups));
        f.write(reinterpret_cast<const char*>(&amp_scale_downs), sizeof(amp_scale_downs));

        // Loader position
        f.write(reinterpret_cast<const char*>(&loader_shard),       sizeof(loader_shard));
        f.write(reinterpret_cast<const char*>(&loader_byte_pos),    sizeof(loader_byte_pos));
        f.write(reinterpret_cast<const char*>(&loader_tokens_seen), sizeof(loader_tokens_seen));
        f.write(reinterpret_cast<const char*>(&loader_epoch),       sizeof(loader_epoch));
        f.write(reinterpret_cast<const char*>(&val_loader_byte_pos),sizeof(val_loader_byte_pos));

        // Smooth metrics
        f.write(reinterpret_cast<const char*>(&gnorm_ema),    sizeof(gnorm_ema));
        f.write(reinterpret_cast<const char*>(&train_ce_ema), sizeof(train_ce_ema));
        f.write(reinterpret_cast<const char*>(&val_ce_ema),   sizeof(val_ce_ema));

        if (!f) {
            std::cerr << "❌ TrainingState write error: " << path << "\n";
            return false;
        }
        std::cout << "✅ Training state saved: " << path << "\n";
        return true;
    }

    // ── Load ─────────────────────────────────────────────────
    bool load(const std::string& base_path, int step_n) {
        std::string path = base_path + "_step" + std::to_string(step_n) + ".trainstate";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cout << "  ℹ️  No training state found (" << path << ") — fresh metrics\n";
            return false;
        }

        uint32_t magic = 0, version = 0;
        f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != TRAINSTATE_MAGIC) {
            std::cerr << "❌ TrainingState corrupt (bad magic): " << path << "\n";
            return false;
        }
        if (version != TRAINSTATE_VERSION) {
            std::cerr << "⚠️  TrainingState version mismatch (file=" << version
                      << " expected=" << TRAINSTATE_VERSION << ") — fresh metrics\n";
            return false;
        }

        // Step & loss
        f.read(reinterpret_cast<char*>(&step),      sizeof(step));
        f.read(reinterpret_cast<char*>(&best_loss), sizeof(best_loss));

        // Overfit detection
        f.read(reinterpret_cast<char*>(&prev_train_ce),  sizeof(prev_train_ce));
        f.read(reinterpret_cast<char*>(&prev_val_ce),    sizeof(prev_val_ce));
        f.read(reinterpret_cast<char*>(&overfit_streak), sizeof(overfit_streak));

        // AMP scaler
        f.read(reinterpret_cast<char*>(&amp_scale),       sizeof(amp_scale));
        f.read(reinterpret_cast<char*>(&amp_window),      sizeof(amp_window));
        f.read(reinterpret_cast<char*>(&amp_overflows),   sizeof(amp_overflows));
        f.read(reinterpret_cast<char*>(&amp_scale_ups),   sizeof(amp_scale_ups));
        f.read(reinterpret_cast<char*>(&amp_scale_downs), sizeof(amp_scale_downs));

        // Loader position
        f.read(reinterpret_cast<char*>(&loader_shard),       sizeof(loader_shard));
        f.read(reinterpret_cast<char*>(&loader_byte_pos),    sizeof(loader_byte_pos));
        f.read(reinterpret_cast<char*>(&loader_tokens_seen), sizeof(loader_tokens_seen));
        f.read(reinterpret_cast<char*>(&loader_epoch),       sizeof(loader_epoch));
        f.read(reinterpret_cast<char*>(&val_loader_byte_pos),sizeof(val_loader_byte_pos));

        // Smooth metrics
        f.read(reinterpret_cast<char*>(&gnorm_ema),    sizeof(gnorm_ema));
        f.read(reinterpret_cast<char*>(&train_ce_ema), sizeof(train_ce_ema));
        f.read(reinterpret_cast<char*>(&val_ce_ema),   sizeof(val_ce_ema));

        if (!f) {
            // Partial read — older format, use what we got
            std::cerr << "⚠️  TrainingState partial read — some fields default\n";
        }

        // Sanity checks
        if (!std::isfinite(best_loss)    || best_loss    < 0.f) best_loss    = 999.f;
        if (!std::isfinite(prev_train_ce)|| prev_train_ce< 0.f) prev_train_ce= 999.f;
        if (!std::isfinite(prev_val_ce)  || prev_val_ce  < 0.f) prev_val_ce  = 999.f;
        if (!std::isfinite(amp_scale)    || amp_scale    < 1.f) amp_scale    = 65536.f;
        if (overfit_streak < 0) overfit_streak = 0;
        if (loader_byte_pos < 0) loader_byte_pos = 0;
        if (val_loader_byte_pos < 0) val_loader_byte_pos = 0;

        std::cout << "✅ Training state loaded: " << path << "\n";
        std::cout << "   step="        << step
                  << " | best_F="      << best_loss
                  << " | train_CE="    << prev_train_ce
                  << " | val_CE="      << prev_val_ce
                  << " | amp_scale="   << amp_scale
                  << "\n";
        std::cout << "   loader_shard=" << loader_shard
                  << " | byte_pos="     << loader_byte_pos
                  << " | tokens_seen="  << loader_tokens_seen
                  << " | epoch="        << loader_epoch
                  << "\n";
        return true;
    }
};
