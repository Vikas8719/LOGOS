#pragma once
// ============================================================
//  LOGOS — Checkpoint.hpp
// ============================================================
#include "Model.hpp"
#include <string>

void save_checkpoint(const LOGOSModel& model,
                     const std::string& path,
                     int step);

bool load_checkpoint(LOGOSModel& model,
                     const std::string& path);
