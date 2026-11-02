#pragma once
#include "Model.hpp"
#include <string>
#include <vector>
#include <cstdint>

static constexpr int    CKPT_MAX_VOCAB    = 100000;
static constexpr int    CKPT_MAX_D_MODEL  = 8192;
static constexpr int    CKPT_MAX_LAYERS   = 128;
static constexpr int    CKPT_MAX_HEADS    = 256;
static constexpr int    CKPT_MAX_SEQ_LEN  = 32768;
static constexpr size_t CKPT_MAX_FILE_MB  = 10000;  // 10GB

bool save_checkpoint(const LOGOSModel& model,
                     const std::string& path,
                     int step);

bool load_checkpoint(LOGOSModel& model, const std::string& path);

bool load_checkpoint_force(LOGOSModel& model, const std::string& path);

struct OptimizerState {
    std::vector<std::vector<float>> velocities;
    float   ema_action    = 0.0f;
    float   log_amplitude = 0.0f;
    int64_t step_count    = 0;
};

bool save_optimizer_state(const OptimizerState& state,
                          const std::string& base_path,
                          int step);

bool load_optimizer_state(OptimizerState& state,
                          const std::string& base_path,
                          int step);
