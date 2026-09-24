// ============================================================
//  LOGOS — PhysicsOpt.cpp
//
//  ⚠️  DEAD FILE — NOT COMPILED — DO NOT USE ⚠️
//
//  CMakeLists.txt sirf main.cpp compile karta hai.
//  Actual LangevinOptimizer include/PhysicsOpt.hpp mein hai (header-only).
//
//  Ye file kyun exist karti hai:
//    - Historical reference — physics equations ka backup explanation
//    - Git history ke liye (purani development ka record)
//
//  BUG 1 HISTORY (do NOT copy this to .hpp):
//    Ye file previously ek alag LangevinOptimizer define karti thi jisme:
//      - T_start = 10.0f (100x zyada! → noise explosion)
//      - lr      = 3e-4f (.hpp se different)
//      - vel formula: γ·v  ← YAHI SAHI THA
//      - noise_scale: √(2γT) × lr inside vel (bina 0.01 dampener ke)
//
//    .hpp mein vel formula (1-γ)·v tha — GALAT.
//    BUG 1 FIX ne .hpp ko γ·v pe le aaya (is file ki tarah).
//
//  ✅ AUTHORITATIVE implementation: include/PhysicsOpt.hpp
//  ❌ Ye file kisi bhi build mein compile/link nahi hoti.
// ============================================================

// Intentionally empty — see include/PhysicsOpt.hpp for live code.
