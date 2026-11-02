
#include "../include/Tensor.hpp"
#include "../include/PhysicsOpt.hpp"    
                                         
                                         
#include "../include/LayerNorm.hpp"     
#include "../include/FeedForward.hpp"   
#include "../include/Attention.hpp"    



namespace logos_physics_driver {


float force_free_energy(const float* logits, int vocab, int target,
                        float T, float* grad) {
    return FreeEnergyLoss::compute(logits, vocab, target, T, grad);
}


void force_langevin_step(LangevinOptimizer& opt,
                          const std::vector<Tensor*>& ps,
                          const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}


void force_shm_step(HybridSHMOptimizer& opt,
                    const std::vector<Tensor*>& ps,
                    const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}


void force_natural_step(NaturalGradientOptimizer& opt,
                         const std::vector<Tensor*>& ps,
                         const std::vector<Tensor*>& gs) {
    opt.step(ps, gs);
}


void force_wpi(WeightPathIntegral& wpi, float loss,
               const std::vector<float>& delta) {
    wpi.record_step(loss, delta);
}


float force_riemannian(RiemannianMetric& rm,
                        const Tensor& t1, const Tensor& t2) {
    return rm.riemannian_distance(t1, t2);
}


Tensor force_reynolds(ReynoldsBatchNorm& rbn, const Tensor& X) {
    return rbn.forward(X, true);
}


float force_feynman(FeynmanDropout& fd) {
    return fd.sample_weight();
}

Tensor force_ns_attention(const Tensor& Q, const Tensor& K,
                           const Tensor& V, const Tensor& mask,
                           float T, float eta, float nu) {
    return navier_stokes_attention(Q, K, V, mask, T, eta, nu);
}


Tensor force_boltzmann(const Tensor& scores, float T) {
    return boltzmann_softmax(scores, T);
}

}  
