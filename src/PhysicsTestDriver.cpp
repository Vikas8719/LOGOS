// ============================================================
//  LOGOS — src/PhysicsTestDriver.cpp
//
//  PURPOSE: MULL mutation testing ke liye physics header code
//  ko binary mein compile karna.
//
//  Kyun zaruri hai:
//    PhysicsOpt.hpp, LayerNorm.hpp, FeedForward.hpp, Attention.hpp
//    sab header-only hain. MULL sirf compiled .o files ke LLVM IR
//    pe mutations apply kar sakta hai. Agar code binary mein nahi
//    hai toh mutation possible hi nahi hoti.
//
//    Yeh file un headers ko explicitly include karti hai + ek
//    thin instantiation layer provide karti hai taaki:
//      - Templates instantiate hon (mull unhe mutate kar sake)
//      - Physics functions binary mein exist karen
//      - test_math_unit.cpp ke M11-M16 tests in mutations ko cover karen
//
//  MULL mein yeh file add hoti hai:
//    mull_math_unit target mein src/PhysicsTestDriver.cpp link hota hai
//    → physics code compiled → mutations apply hoti hain
//    → test_math_unit binary unhe exercise karta hai (M11-M16)
//    → mutation score physics code ko bhi reflect karta hai
//
//  Note: Yeh file koi test nahi karti — sirf compilation driver hai.
//  Actual tests test_math_unit.cpp mein hain (M11-M16).
// ============================================================

// ── Physics optimizer headers ────────────────────────────────
#include "../include/Tensor.hpp"
#include "../include/PhysicsOpt.hpp"    // FreeEnergyLoss, LangevinOptimizer,
                                         // HybridSHMOptimizer, NaturalGradientOptimizer,
                                         // WeightPathIntegral, RiemannianMetric
#include "../include/LayerNorm.hpp"     // ReynoldsBatchNorm, layernorm_cpu
#include "../include/FeedForward.hpp"   // FeynmanDropout, FeedForwardLayer
#include "../include/Attention.hpp"     // navier_stokes_attention, boltzmann_softmax

// ── Explicit instantiations / force-compile ──────────────────
// Compiler "as-if" rule pe symbols optimize out ho sakte hain.
// Volatile pointers se ensure karte hain ki code survive kare.

namespace logos_physics_driver {

// FreeEnergyLoss::compute — M3, M10 mutations cover karte hain
float force_free_energy(const float* logits, int vocab, int target,
                        float T, float* grad) {
    return FreeEnergyLoss::compute(logits, vocab, target, T, grad);
}

// LangevinOptimizer::step — M4 (leapfrog Langevin) mutations
void force_langevin_step(LangevinOptimizer& opt,
                          const std::vector<Tensor*>& ps,
                          const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}

// HybridSHMOptimizer::step + anneal — M11 alpha_H/alpha_L annealing
void force_shm_step(HybridSHMOptimizer& opt,
                    const std::vector<Tensor*>& ps,
                    const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}

// NaturalGradientOptimizer — M11 Fisher diagonal EMA
void force_natural_step(NaturalGradientOptimizer& opt,
                         const std::vector<Tensor*>& ps,
                         const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}

// WeightPathIntegral — M16 action accumulation
void force_wpi(WeightPathIntegral& wpi, float loss,
               const std::vector<float>& delta) {
    wpi.record_step(loss, delta);
}

// RiemannianMetric — M15 curved parameter space
float force_riemannian(RiemannianMetric& rm,
                        const Tensor& t1, const Tensor& t2) {
    return rm.riemannian_distance(t1, t2);
}

// ReynoldsBatchNorm — M13 laminar/turbulent blend
Tensor force_reynolds(ReynoldsBatchNorm& rbn, const Tensor& X) {
    return rbn.forward(X, true);
}

// FeynmanDropout — M14 path integral amplitudes
float force_feynman(FeynmanDropout& fd) {
    return fd.sample_weight();
}

// navier_stokes_attention — M12 fluid flow attention
Tensor force_ns_attention(const Tensor& Q, const Tensor& K,
                           const Tensor& V, const Tensor& mask,
                           float T, float eta, float nu) {
    return navier_stokes_attention(Q, K, V, mask, T, eta, nu);
}

// boltzmann_softmax — M5 temperature scaling
Tensor force_boltzmann(const Tensor& scores, float T) {
    return boltzmann_softmax(scores, T);
}

}  // namespace logos_physics_driver
