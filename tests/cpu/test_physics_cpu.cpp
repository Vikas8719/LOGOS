// T21: CPU-side physics tests — WeightPathIntegral (action / amplitude / adaptive LR).
// Expected values are computed independently in double precision from the written formulas.
#include "test_common.hpp"
#include "PathIntegral.hpp"
#include <limits>

static void test_hand_computed() {
    TC_SECTION("WeightPathIntegral: hand-computed values");
    WeightPathIntegral p(2.0f);                       // hbar = 2
    p.record_step_norm(3.0f, 0.5f);                   // action = 1.5
    TC_NEAR(p.last_action, 1.5, 1e-6, "action = loss * norm");
    TC_NEAR(p.cumulative_action, 1.5, 1e-6, "cumulative_action after step 1");
    TC_NEAR(p.log_amplitude, -0.75, 1e-6, "log_A = -action/hbar");
    TC_NEAR(p.ema_action, 1.5, 1e-6, "ema initialised to first action");
    TC_NEAR(p.amplitude(), std::exp(-0.75), 1e-6, "amplitude = exp(log_A)");

    p.record_step_norm(1.0f, 1.0f);                   // action = 1
    double ema2 = 0.98 * 1.5 + 0.02 * 1.0;            // 1.49
    TC_NEAR(p.ema_action, ema2, 1e-6, "ema update 0.98/0.02");
    TC_NEAR(p.log_amplitude, -0.75 - 0.5, 1e-6, "log_A accumulates");
    TC_NEAR(p.cumulative_action, 2.5, 1e-6, "cumulative_action accumulates");
    TC_CHECK(p.step_count == 2, "step_count == 2");
    // last(1.0) < ema(1.49) -> rel clipped to 1 -> scale 1
    TC_NEAR(p.lr_scale_ema(0.5f), 1.0, 1e-6, "calm step -> lr scale 1.0");

    p.record_step_norm(10.0f, 1.0f);                  // spike: action = 10
    double ema3 = 0.98 * ema2 + 0.02 * 10.0;
    double rel  = std::min(1.0, std::exp(-(10.0 - ema3) / (ema3 + 1e-8)));
    TC_NEAR(p.lr_scale_ema(0.5f), 0.5 + 0.5 * rel, 1e-5, "spike -> lr scale matches closed form");
    TC_CHECK(p.lr_scale_ema(0.5f) < 0.51f, "big spike pushes lr scale to ~floor");
    TC_NEAR(p.lr_scale_ema(0.2f), 0.2 + 0.8 * rel, 1e-5, "lr_min_frac respected");
}

static void test_hbar() {
    TC_SECTION("hbar scaling + guard");
    WeightPathIntegral a(1.0f), b(4.0f);
    for (int i = 0; i < 10; ++i) { a.record_step_norm(2.0f, 0.3f); b.record_step_norm(2.0f, 0.3f); }
    TC_NEAR(a.log_amplitude, 4.0 * b.log_amplitude, 1e-4, "log_A scales as 1/hbar");
    WeightPathIntegral z(0.0f);
    z.record_step_norm(1.0f, 1.0f);
    TC_CHECK(std::isfinite(z.log_amplitude) && std::isfinite(z.amplitude()), "hbar=0 clamped (no inf/NaN)");
    WeightPathIntegral n(-5.0f);
    n.record_step_norm(1.0f, 1.0f);
    TC_CHECK(std::isfinite(n.log_amplitude), "negative hbar clamped");
}

static void test_robustness() {
    TC_SECTION("NaN / Inf / negative inputs never poison state");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    WeightPathIntegral p(1.0f);
    p.record_step_norm(1.0f, 1.0f);
    const float la = p.log_amplitude;
    const int sc = p.step_count;
    p.record_step_norm(nan, 1.0f);
    p.record_step_norm(1.0f, nan);
    p.record_step_norm(inf, 1.0f);
    p.record_step_norm(1.0f, inf);
    p.record_step_norm(inf, 0.0f);          // inf*0 = NaN
    TC_CHECK(p.log_amplitude == la, "non-finite steps ignored (log_A unchanged)");
    TC_CHECK(p.step_count == sc, "non-finite steps not counted");
    TC_CHECK(std::isfinite(p.ema_action) && std::isfinite(p.cumulative_action), "ema/cumulative finite");
    p.record_step_norm(-5.0f, 1.0f);        // negative loss -> action 0
    TC_NEAR(p.last_action, 0.0, 0.0, "negative loss clamps to 0 action");
    TC_CHECK(p.log_amplitude == la, "zero action leaves log_A unchanged");
    p.record_step_norm(2.0f, -3.0f);        // negative norm -> action 0
    TC_NEAR(p.last_action, 0.0, 0.0, "negative norm clamps to 0 action");
    TC_CHECK(p.log_amplitude <= la, "log_A never increases");
}

static void test_amplitude_bounds() {
    TC_SECTION("amplitude / relative_amplitude bounds");
    WeightPathIntegral p(1e-3f);
    TC_NEAR(p.relative_amplitude(), 1.0, 0.0, "relative_amplitude==1 before reset_best");
    p.record_step_norm(1.0f, 1.0f);
    p.reset_best();
    TC_CHECK(p.best_step == p.step_count, "reset_best records step");
    TC_NEAR(p.relative_amplitude(), 1.0, 1e-6, "relative_amplitude==1 right after reset_best");
    p.record_step_norm(1.0f, 1.0f);
    TC_CHECK(p.relative_amplitude() < 1.0f, "relative_amplitude < 1 after more action");
    for (int i = 0; i < 100; ++i) p.record_step_norm(100.0f, 100.0f);   // log_A -> very negative
    TC_CHECK(p.log_amplitude < -88.0f, "log_A below clamp");
    TC_CHECK(p.amplitude() > 0.0f && std::isfinite(p.amplitude()), "amplitude clamped: >0 and finite");
    TC_CHECK(p.relative_amplitude() > 0.0f, "relative_amplitude clamped: >0");
    TC_CHECK(p.amplitude() <= 1.0f, "amplitude <= 1");
}

static void test_lr_scale_behaviour() {
    TC_SECTION("adaptive LR behaviour");
    WeightPathIntegral p(1.0f);
    for (int i = 0; i < 200; ++i) p.record_step_norm(2.0f, 0.5f);       // steady
    TC_NEAR(p.lr_scale_ema(0.5f), 1.0, 1e-5, "steady training -> scale 1");
    float prev = 1.0f;
    for (float spike : {2.0f, 4.0f, 8.0f, 16.0f}) {
        WeightPathIntegral q(1.0f);
        for (int i = 0; i < 100; ++i) q.record_step_norm(1.0f, 1.0f);
        q.record_step_norm(spike, 1.0f);
        float s = q.lr_scale_ema(0.5f);
        TC_CHECK(s <= prev + 1e-6f, "scale non-increasing as spike grows");
        TC_CHECK(s >= 0.5f - 1e-6f && s <= 1.0f + 1e-6f, "scale within [floor,1]");
        prev = s;
    }
}

int main() {
    std::printf("LOGOS T21: physics (CPU) tests\n");
    test_hand_computed();
    test_hbar();
    test_robustness();
    test_amplitude_bounds();
    test_lr_scale_behaviour();
    return tc::finish("T21_Physics_CPU");
}
