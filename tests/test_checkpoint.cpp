#include "Checkpoint.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <sstream>

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 8;
    cfg.d_model = 4;
    cfg.num_heads = 2;
    cfg.num_layers = 1;
    cfg.max_seq_len = 3;

    LOGOSModel source(cfg);
    const std::string base = "logos_checkpoint_roundtrip_test";
    const std::string file = base + "_step7.bin";
    std::remove(file.c_str());

    if (std::filesystem::exists(file)) {
        std::cerr << "file should not exist before save\n";
        return 1;
    }

    if (!save_checkpoint(source, base, 7)) {
        std::cerr << "checkpoint save failed\n";
        return 1;
    }

    if (!std::filesystem::exists(file)) {
        std::cerr << "save_checkpoint did not create file (void call side-effect missing)\n";
        return 1;
    }
    if (std::filesystem::file_size(file) == 0) {
        std::cerr << "save_checkpoint wrote empty file\n";
        return 1;
    }

    size_t expected_bytes = sizeof(ModelConfig) +
        static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    for (const Tensor* parameter : source.parameters())
        expected_bytes += static_cast<size_t>(parameter->total_size) * sizeof(float);
    expected_bytes += static_cast<size_t>(cfg.num_layers) * 4 *
                      static_cast<size_t>(cfg.d_model) * sizeof(float);
    // footer: final LayerNorm gamma + beta
    expected_bytes += 2 * static_cast<size_t>(cfg.d_model) * sizeof(float);


    if (sizeof(ModelConfig) == 0) {
        std::cerr << "sizeof(ModelConfig) must be > 0\n";
        return 1;
    }

    size_t pos_block_bytes = static_cast<size_t>(cfg.max_seq_len) *
                             static_cast<size_t>(cfg.d_model) * sizeof(float);
    if (pos_block_bytes != 3 * 4 * sizeof(float)) {
        std::cerr << "pos block bytes wrong: " << pos_block_bytes << "\n";
        return 1;
    }

    size_t rbn_extra_bytes = static_cast<size_t>(cfg.num_layers) * 4 *
                             static_cast<size_t>(cfg.d_model) * sizeof(float);
    if (rbn_extra_bytes != 1 * 4 * 4 * sizeof(float)) {
        std::cerr << "RBN extra bytes wrong: " << rbn_extra_bytes << "\n";
        return 1;
    }

    if (expected_bytes <= sizeof(ModelConfig)) {
        std::cerr << "expected_bytes too small (arithmetic wrong): " << expected_bytes << "\n";
        return 1;
    }

    if (expected_bytes < sizeof(ModelConfig) + pos_block_bytes) {
        std::cerr << "expected_bytes misses pos block\n";
        return 1;
    }

    {
        // size pehle padho, file baad me hatao (pehle ulta tha: remove ke baad file_size() throw karta tha)
        const size_t got_bytes = std::filesystem::file_size(file);
        if (got_bytes != expected_bytes) {
            std::remove(file.c_str());
            std::cerr << "checkpoint size mismatch: got " << got_bytes
                      << " expected " << expected_bytes << "\n";
            return 1;
        }
    }

    {
        std::fstream legacy(file, std::ios::binary | std::ios::in | std::ios::out);
        const auto position_offset = static_cast<std::streamoff>(sizeof(ModelConfig)) +
            static_cast<std::streamoff>(source.embedding.total_size * sizeof(float));
        legacy.seekp(position_offset);
        std::vector<float> old_positions(
            static_cast<size_t>(cfg.max_seq_len) * cfg.d_model, 0.25f);
        legacy.write(reinterpret_cast<const char*>(old_positions.data()),
                     static_cast<std::streamsize>(old_positions.size() * sizeof(float)));
        if (!legacy) {
            std::remove(file.c_str());
            std::cerr << "could not prepare legacy checkpoint fixture\n";
            return 1;
        }
    }

    LOGOSModel restored(cfg);
    if (!load_checkpoint(restored, file)) {
        std::remove(file.c_str());
        std::cerr << "checkpoint load failed\n";
        return 1;
    }

    const auto expected = source.parameters();
    const auto actual = restored.parameters();

    if (expected.size() != actual.size()) {
        std::remove(file.c_str());
        std::cerr << "parameter count mismatch: src=" << expected.size()
                  << " restored=" << actual.size() << "\n";
        return 1;
    }
    if (expected.size() < 2) {
        std::remove(file.c_str());
        std::cerr << "too few parameters: " << expected.size() << "\n";
        return 1;
    }

    if (expected[0]->total_size != cfg.vocab_size * cfg.d_model) {
        std::remove(file.c_str());
        std::cerr << "embedding total_size wrong: " << expected[0]->total_size
                  << " expected " << (cfg.vocab_size * cfg.d_model) << "\n";
        return 1;
    }

    if (expected[1]->total_size != cfg.d_model * cfg.vocab_size) {
        std::remove(file.c_str());
        std::cerr << "lm_head total_size wrong: " << expected[1]->total_size
                  << " expected " << (cfg.d_model * cfg.vocab_size) << "\n";
        return 1;
    }

    bool equal = (expected.size() == actual.size());
    for (size_t i = 0; equal && i < expected.size(); ++i) {
        equal = expected[i]->shape == actual[i]->shape &&
                expected[i]->data == actual[i]->data;
    }

    // A non-finite value in a non-embedding parameter must be rejected by the
    // strict loader. This is the corruption class exercised by fuzz_checkpoint.
    {
        const std::string nan_base = "logos_checkpoint_non_finite";
        const std::string nan_file = nan_base + "_step0.bin";
        std::remove(nan_file.c_str());
        if (!save_checkpoint(source, nan_base, 0)) {
            std::remove(file.c_str());
            std::cerr << "non-finite fixture save failed\n";
            return 1;
        }
        {
            std::fstream corrupt(nan_file, std::ios::binary | std::ios::in | std::ios::out);
            const auto first_head_offset = static_cast<std::streamoff>(sizeof(ModelConfig)) +
                static_cast<std::streamoff>(source.embedding.total_size * sizeof(float)) +
                static_cast<std::streamoff>(source.pos_embedding.total_size * sizeof(float)) +
                static_cast<std::streamoff>(source.lm_head.total_size * sizeof(float));
            corrupt.seekp(first_head_offset);
            const float nan = std::numeric_limits<float>::quiet_NaN();
            corrupt.write(reinterpret_cast<const char*>(&nan), sizeof(nan));
            if (!corrupt) {
                std::remove(file.c_str());
                std::remove(nan_file.c_str());
                std::cerr << "could not corrupt non-embedding checkpoint parameter\n";
                return 1;
            }
        }
        LOGOSModel nan_target(cfg);
        if (load_checkpoint(nan_target, nan_file)) {
            std::remove(file.c_str());
            std::remove(nan_file.c_str());
            std::cerr << "strict loader accepted NaN in a non-embedding parameter\n";
            return 1;
        }
        std::remove(nan_file.c_str());
    }
    std::remove(file.c_str());
    if (!equal) {
        std::cerr << "checkpoint round-trip changed model parameters\n";
        return 1;
    }

    std::cout << "checkpoint round-trip passed\n";

    {
        ModelConfig cfg2;
        cfg2.vocab_size  = 16;
        cfg2.d_model     = 8;
        cfg2.num_heads   = 2;
        cfg2.num_layers  = 2;
        cfg2.max_seq_len = 4;

        LOGOSModel m2(cfg2);
        const std::string base2 = "logos_ckpt_formula_test";
        const std::string file2 = base2 + "_step1.bin";
        std::remove(file2.c_str());

        if (!save_checkpoint(m2, base2, 1)) {
            std::cerr << "formula test: save failed\n";
            return 1;
        }

        size_t expected2 = sizeof(ModelConfig) +
            static_cast<size_t>(cfg2.max_seq_len) * cfg2.d_model * sizeof(float);
        for (const Tensor* p : m2.parameters())
            expected2 += static_cast<size_t>(p->total_size) * sizeof(float);
        expected2 += static_cast<size_t>(cfg2.num_layers) * 4 *
                     static_cast<size_t>(cfg2.d_model) * sizeof(float);
        expected2 += 2 * static_cast<size_t>(cfg2.d_model) * sizeof(float);   // final LN footer

        size_t actual2 = std::filesystem::file_size(file2);
        std::remove(file2.c_str());

        if (actual2 != expected2) {
            std::cerr << "formula test: size mismatch: got " << actual2
                      << " expected " << expected2 << "\n";
            return 1;
        }

        if (sizeof(ModelConfig) == 0) {
            std::cerr << "sizeof(ModelConfig) must be > 0\n";
            return 1;
        }

        size_t pos_block = static_cast<size_t>(cfg2.max_seq_len) * cfg2.d_model * sizeof(float);
        if (pos_block != 4 * 8 * sizeof(float)) {
            std::cerr << "legacy pos block size wrong: " << pos_block << "\n";
            return 1;
        }

        size_t rbn_bytes = static_cast<size_t>(cfg2.num_layers) * 4 *
                           static_cast<size_t>(cfg2.d_model) * sizeof(float);
        if (rbn_bytes != 2 * 4 * 8 * sizeof(float)) {
            std::cerr << "RBN bytes wrong: " << rbn_bytes << "\n";
            return 1;
        }

        std::cout << "formula test passed: expected=" << expected2
                  << " actual=" << actual2 << "\n";
    }

    {
        ModelConfig cfg3;
        cfg3.vocab_size  = 8;
        cfg3.d_model     = 4;
        cfg3.num_heads   = 2;
        cfg3.num_layers  = 1;
        cfg3.max_seq_len = 3;

        LOGOSModel src3(cfg3);
        {
            int idx = 0;
            for (Tensor* p : src3.parameters())
                for (float& v : p->data) v = (float)(++idx) * 0.01f;
        }
        // final LayerNorm (footer) non-default values so the round-trip really exercises it
        for (size_t i = 0; i < src3.final_ln.gamma.data.size(); ++i)
            src3.final_ln.gamma.data[i] = 1.5f + 0.25f * static_cast<float>(i);
        for (size_t i = 0; i < src3.final_ln.beta.data.size(); ++i)
            src3.final_ln.beta.data[i] = -0.5f + 0.125f * static_cast<float>(i);
        const std::string base3 = "logos_ckpt_nontrivial";
        const std::string file3 = base3 + "_step5.bin";
        std::remove(file3.c_str());

        if (!save_checkpoint(src3, base3, 5)) {
            std::cerr << "nontrivial save failed\n";
            return 1;
        }

        LOGOSModel rst3(cfg3);
        if (!load_checkpoint(rst3, file3)) {
            std::remove(file3.c_str());
            std::cerr << "nontrivial load failed\n";
            return 1;
        }
        std::remove(file3.c_str());

        const auto ps3 = src3.parameters();
        const auto pr3 = rst3.parameters();
        bool match3 = (ps3.size() == pr3.size());
        for (size_t i = 0; match3 && i < ps3.size(); ++i)
            match3 = (ps3[i]->data == pr3[i]->data);
        match3 = match3 &&
                 src3.final_ln.gamma.data == rst3.final_ln.gamma.data &&
                 src3.final_ln.beta.data  == rst3.final_ln.beta.data;

        if (!match3) {
            std::cerr << "nontrivial round-trip: parameter mismatch\n";
            return 1;
        }
        std::cout << "nontrivial round-trip passed\n";
    }

    {
        ModelConfig cfg4;
        cfg4.vocab_size  = 4;
        cfg4.d_model     = 4;
        cfg4.num_heads   = 1;
        cfg4.num_layers  = 1;
        cfg4.max_seq_len = 2;

        LOGOSModel src4(cfg4);
        const std::string base4 = "logos_ckpt_offsets";
        const std::string file4 = base4 + "_step3.bin";
        std::remove(file4.c_str());

        if (!save_checkpoint(src4, base4, 3)) {
            std::cerr << "offset test save failed\n";
            return 1;
        }

        {
            std::ifstream f(file4, std::ios::binary);
            ModelConfig read_cfg;
            f.read(reinterpret_cast<char*>(&read_cfg), sizeof(ModelConfig));
            if (read_cfg.vocab_size  != cfg4.vocab_size  ||
                read_cfg.d_model     != cfg4.d_model     ||
                read_cfg.num_heads   != cfg4.num_heads   ||
                read_cfg.num_layers  != cfg4.num_layers  ||
                read_cfg.max_seq_len != cfg4.max_seq_len) {
                std::remove(file4.c_str());
                std::cerr << "offset test: ModelConfig mismatch in file\n";
                return 1;
            }

            size_t emb_floats = static_cast<size_t>(cfg4.vocab_size) * cfg4.d_model;
            std::vector<float> emb(emb_floats);
            f.read(reinterpret_cast<char*>(emb.data()),
                   static_cast<std::streamsize>(emb_floats * sizeof(float)));
            if (!f) {
                std::remove(file4.c_str());
                std::cerr << "offset test: could not read embedding block\n";
                return 1;
            }
            bool emb_match = true;
            for (size_t i = 0; i < emb_floats; ++i)
                if (emb[i] != src4.embedding.data[i]) emb_match = false;
            if (!emb_match) {
                std::remove(file4.c_str());
                std::cerr << "offset test: embedding data mismatch at file offset\n";
                return 1;
            }
        }
        std::remove(file4.c_str());
        std::cout << "offset arithmetic test passed\n";
    }

    {
        const std::string opt_base = "logos_ckpt_optstate_test";
        const std::string opt_file = opt_base + "_step11.optstate";
        std::remove(opt_file.c_str());

        OptimizerState saved;
        saved.velocities = {{0.25f, -0.5f}, {1.5f}};
        saved.ema_action = 2.75f;
        saved.log_amplitude = -0.125f;
        saved.step_count = 42;
        if (!save_optimizer_state(saved, opt_base, 11)) {
            std::cerr << "optimizer state save failed\n";
            return 1;
        }

        OptimizerState loaded;
        if (!load_optimizer_state(loaded, opt_base, 11) ||
            loaded.velocities != saved.velocities ||
            loaded.ema_action != saved.ema_action ||
            loaded.log_amplitude != saved.log_amplitude ||
            loaded.step_count != saved.step_count) {
            std::remove(opt_file.c_str());
            std::cerr << "optimizer state round-trip mismatch\n";
            return 1;
        }

        const auto write_header = [&](uint32_t magic, uint32_t version,
                                      int64_t count, uint32_t vector_size) {
            std::ofstream f(opt_file, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<const char*>(&version), sizeof(version));
            f.write(reinterpret_cast<const char*>(&count), sizeof(count));
            if (count > 0)
                f.write(reinterpret_cast<const char*>(&vector_size), sizeof(vector_size));
        };
        constexpr uint32_t magic = 0x4F505431u;
        write_header(0, 1, 1, 1);
        loaded.velocities.clear();
        loaded.step_count = 0;
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "bad optimizer magic was accepted\n";
            return 1;
        }
        if (!loaded.velocities.empty()) {
            std::remove(opt_file.c_str());
            std::cerr << "velocities populated despite bad magic rejection\n";
            return 1;
        }

        write_header(magic ^ 1u, 1, 1, 1);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "near-magic value was accepted (kills != -> == mutation)\n";
            return 1;
        }
        if (!loaded.velocities.empty()) {
            std::remove(opt_file.c_str());
            std::cerr << "velocities populated despite near-magic rejection\n";
            return 1;
        }

        write_header(magic, 2, 1, 1);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "bad optimizer version was accepted\n";
            return 1;
        }
        if (!loaded.velocities.empty()) {
            std::remove(opt_file.c_str());
            std::cerr << "velocities populated despite version rejection\n";
            return 1;
        }

        write_header(magic, 0, 1, 1);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "version 0 was accepted (kills != -> == version check)\n";
            return 1;
        }
        if (!loaded.velocities.empty()) {
            std::remove(opt_file.c_str());
            std::cerr << "velocities populated despite version-0 rejection\n";
            return 1;
        }

        write_header(magic, 1, 0, 1);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "empty optimizer state was accepted\n";
            return 1;
        }
        if (!loaded.velocities.empty()) {
            std::remove(opt_file.c_str());
            std::cerr << "velocities populated despite n_params=0 rejection\n";
            return 1;
        }

        write_header(magic, 1, static_cast<int64_t>(-1), 1);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "negative n_params was accepted (kills <= -> >= mutation)\n";
            return 1;
        }

        write_header(magic, 1, 1, 0);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "empty optimizer velocity was accepted\n";
            return 1;
        }
        write_header(magic, 1, 1, 2);
        loaded.velocities.clear();
        if (load_optimizer_state(loaded, opt_base, 11)) {
            std::remove(opt_file.c_str());
            std::cerr << "truncated optimizer velocity was accepted\n";
            return 1;
        }

        {
            std::ofstream f(opt_file, std::ios::binary | std::ios::trunc);
            const uint32_t version = 1, size = 1;
            const int64_t count = 1;
            const float velocity = 3.5f;
            f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<const char*>(&version), sizeof(version));
            f.write(reinterpret_cast<const char*>(&count), sizeof(count));
            f.write(reinterpret_cast<const char*>(&size), sizeof(size));
            f.write(reinterpret_cast<const char*>(&velocity), sizeof(velocity));
        }
        loaded.ema_action = 9.0f;
        loaded.log_amplitude = 8.0f;
        loaded.step_count = 7;
        if (!load_optimizer_state(loaded, opt_base, 11) ||
            loaded.velocities.size() != 1 || loaded.velocities[0].size() != 1 ||
            loaded.velocities[0][0] != 3.5f || loaded.ema_action != 0.0f ||
            loaded.log_amplitude != 0.0f || loaded.step_count != 0) {
            std::remove(opt_file.c_str());
            std::cerr << "legacy optimizer state was not restored safely\n";
            return 1;
        }
        std::remove(opt_file.c_str());
    }

    {
        const std::string bad_file = "logos_ckpt_invalid_config.bin";
        LOGOSModel target(cfg);
        const auto load_rejects_with = [&](const ModelConfig& file_cfg,
                                           const std::string& expected_error) {
            {
                std::ofstream f(bad_file, std::ios::binary | std::ios::trunc);
                f.write(reinterpret_cast<const char*>(&file_cfg), sizeof(file_cfg));
            }
            std::ostringstream captured;
            std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
            const bool loaded = load_checkpoint(target, bad_file);
            std::cerr.rdbuf(previous);
            return !loaded && captured.str().find(expected_error) != std::string::npos;
        };
        {
            std::ofstream f(bad_file, std::ios::binary | std::ios::trunc);
            const int partial = 8;
            f.write(reinterpret_cast<const char*>(&partial), sizeof(partial));
        }
        {
            std::ostringstream captured;
            std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
            const bool loaded = load_checkpoint(target, bad_file);
            std::cerr.rdbuf(previous);
            if (loaded || captured.str().find("truncated (can't read cfg)") == std::string::npos) {
                std::remove(bad_file.c_str());
                std::cerr << "short checkpoint config was not rejected at the header\n";
                return 1;
            }
        }
        ModelConfig invalid = cfg;
        invalid.d_model = 3; // not divisible by num_heads
        if (!load_rejects_with(invalid, "not divisible by num_heads")) {
            std::remove(bad_file.c_str());
            std::cerr << "invalid checkpoint dimensions were not rejected by validation\n";
            return 1;
        }

        invalid = cfg;
        invalid.vocab_size = 100001; // above the supported bound
        if (!load_rejects_with(invalid, "out of range")) {
            std::remove(bad_file.c_str());
            std::cerr << "out-of-range checkpoint dimensions were not rejected by validation\n";
            return 1;
        }

        invalid = cfg;
        invalid.vocab_size = 100000;
        invalid.d_model = 8192;
        invalid.num_heads = 256;
        invalid.num_layers = 2;
        invalid.max_seq_len = 1;
        if (!load_rejects_with(invalid, "exceeds max")) {
            std::remove(bad_file.c_str());
            std::cerr << "oversized checkpoint dimensions were not rejected by validation\n";
            return 1;
        }
        std::remove(bad_file.c_str());
    }

    {
        const std::string mismatch_base = "logos_ckpt_mismatch";
        const std::string mismatch_file = mismatch_base + "_step2.bin";
        std::remove(mismatch_file.c_str());
        if (!save_checkpoint(source, mismatch_base, 2)) {
            std::cerr << "mismatch fixture save failed\n";
            return 1;
        }

        // --- mutation gap 1: vocab_size != → == (cfgs_match kills this mutant) ---
        // If != is mutated to == the check inverts: matching vocab_size would REJECT
        // and mismatching would ACCEPT. We verify both directions explicitly.
        {
            // Exact match must SUCCEED (would fail if != were left as ==)
            LOGOSModel exact_match_model(cfg);
            if (!load_checkpoint(exact_match_model, mismatch_file)) {
                std::remove(mismatch_file.c_str());
                std::cerr << "cfgs_match gap1: exact vocab_size rejected (killed != -> == mutant)\n";
                return 1;
            }
            // vocab_size+1 must FAIL (would pass under the mutant)
            ModelConfig vbad = cfg; vbad.vocab_size += 1;
            LOGOSModel vbad_model(vbad);
            if (load_checkpoint(vbad_model, mismatch_file)) {
                std::remove(mismatch_file.c_str());
                std::cerr << "cfgs_match gap1: mismatched vocab_size accepted\n";
                return 1;
            }
            // vocab_size-1 must also FAIL
            ModelConfig vbad2 = cfg; vbad2.vocab_size -= 1;
            LOGOSModel vbad2_model(vbad2);
            if (load_checkpoint(vbad2_model, mismatch_file)) {
                std::remove(mismatch_file.c_str());
                std::cerr << "cfgs_match gap1: vocab_size-1 accepted (kills off-by-one mutant)\n";
                return 1;
            }
        }

        // --- mutation gap 2: pos_embedding block == → != ---
        // The loaded pos_embedding data must equal what was saved. We verify this by
        // loading a fresh model and checking that its pos_embedding block matches the source.
        {
            LOGOSModel pos_check(cfg);
            if (!load_checkpoint(pos_check, mismatch_file)) {
                std::remove(mismatch_file.c_str());
                std::cerr << "pos_embedding check: load failed\n";
                return 1;
            }
            // After a clean save+load the pos embedding must match source
            if (pos_check.pos_embedding.data != source.pos_embedding.data) {
                std::remove(mismatch_file.c_str());
                std::cerr << "cfgs_match gap2: pos_embedding data mismatch after round-trip (killed == -> != mutant)\n";
                return 1;
            }
        }

        // Full mismatch sweep (all five fields)
        std::vector<ModelConfig> mismatches;
        ModelConfig changed = cfg;
        changed.vocab_size += 1; mismatches.push_back(changed);
        changed = cfg; changed.d_model += 2; mismatches.push_back(changed);
        changed = cfg; changed.num_layers += 1; mismatches.push_back(changed);
        changed = cfg; changed.num_heads = 1; mismatches.push_back(changed);
        changed = cfg; changed.max_seq_len += 1; mismatches.push_back(changed);
        for (const ModelConfig& mismatch_cfg : mismatches) {
            LOGOSModel mismatch_model(mismatch_cfg);
            if (load_checkpoint(mismatch_model, mismatch_file)) {
                std::remove(mismatch_file.c_str());
                std::cerr << "strict loader accepted a mismatched config\n";
                return 1;
            }
        }
        std::remove(mismatch_file.c_str());
        // Verify file was actually cleaned up (temp-file cleanup test)
        if (std::filesystem::exists(mismatch_file)) {
            std::cerr << "mismatch fixture file not removed after test\n";
            return 1;
        }
        std::cout << "cfgs_match mutation gaps (vocab_size !=, pos_embedding ==) + cleanup passed\n";
    }

    {
        const std::string legacy_base = "logos_ckpt_legacy_stats";
        const std::string legacy_file = legacy_base + "_step4.bin";
        std::remove(legacy_file.c_str());
        if (!save_checkpoint(source, legacy_base, 4)) {
            std::cerr << "legacy stats fixture save failed\n";
            return 1;
        }
        const size_t stats_bytes = static_cast<size_t>(cfg.num_layers) * 4 *
                                   static_cast<size_t>(cfg.d_model) * sizeof(float);
        // legacy = no BN stats AND no final-LN footer (footer ab file ke sabse end me hai)
        const size_t footer_bytes = 2 * static_cast<size_t>(cfg.d_model) * sizeof(float);
        const size_t full_size = std::filesystem::file_size(legacy_file);
        std::filesystem::resize_file(legacy_file, full_size - stats_bytes - footer_bytes);

        LOGOSModel legacy_target(cfg);
        for (auto& block : legacy_target.layers) {
            std::fill(block.ln1.running_mean.begin(), block.ln1.running_mean.end(), 9.0f);
            std::fill(block.ln1.running_var.begin(), block.ln1.running_var.end(), 9.0f);
            std::fill(block.ln2.running_mean.begin(), block.ln2.running_mean.end(), 9.0f);
            std::fill(block.ln2.running_var.begin(), block.ln2.running_var.end(), 9.0f);
        }
        if (!load_checkpoint(legacy_target, legacy_file)) {
            std::remove(legacy_file.c_str());
            std::cerr << "legacy checkpoint without BN stats failed to load\n";
            return 1;
        }
        for (const auto& block : legacy_target.layers) {
            const auto all_zero = [](const std::vector<float>& values) {
                return std::all_of(values.begin(), values.end(),
                                   [](float value) { return value == 0.0f; });
            };
            if (!block.ln1.initialized || !block.ln2.initialized ||
                !all_zero(block.ln1.running_mean) || !all_zero(block.ln1.running_var) ||
                !all_zero(block.ln2.running_mean) || !all_zero(block.ln2.running_var)) {
                std::remove(legacy_file.c_str());
                std::cerr << "missing BN statistics were not zero-initialized\n";
                return 1;
            }
        }
        for (float g : legacy_target.final_ln.gamma.data)
            if (g != 1.0f) {
                std::remove(legacy_file.c_str());
                std::cerr << "legacy checkpoint: final_ln.gamma not default (1.0)\n";
                return 1;
            }
        for (float b : legacy_target.final_ln.beta.data)
            if (b != 0.0f) {
                std::remove(legacy_file.c_str());
                std::cerr << "legacy checkpoint: final_ln.beta not default (0.0)\n";
                return 1;
            }
        std::remove(legacy_file.c_str());
    }

    {
        // checkpoint with BN stats but WITHOUT the final-LN footer (pre-footer format):
        // load must succeed and leave final_ln at defaults (gamma=1, beta=0)
        const std::string nf_base = "logos_ckpt_no_footer";
        const std::string nf_file = nf_base + "_step8.bin";
        std::remove(nf_file.c_str());
        if (!save_checkpoint(source, nf_base, 8)) {
            std::cerr << "no-footer fixture save failed\n";
            return 1;
        }
        const size_t footer_bytes = 2 * static_cast<size_t>(cfg.d_model) * sizeof(float);
        std::filesystem::resize_file(nf_file, std::filesystem::file_size(nf_file) - footer_bytes);

        LOGOSModel nf_target(cfg);
        if (!load_checkpoint(nf_target, nf_file)) {
            std::remove(nf_file.c_str());
            std::cerr << "checkpoint without final-LN footer failed to load\n";
            return 1;
        }
        std::remove(nf_file.c_str());
        for (float g : nf_target.final_ln.gamma.data)
            if (g != 1.0f) { std::cerr << "no-footer: final_ln.gamma not default\n"; return 1; }
        for (float b : nf_target.final_ln.beta.data)
            if (b != 0.0f) { std::cerr << "no-footer: final_ln.beta not default\n"; return 1; }
        std::cout << "no-footer (pre final-LN) checkpoint loads with defaults\n";
    }

    {
        const std::string force_base = "logos_ckpt_force_read";
        const std::string force_file = force_base + "_step6.bin";
        std::remove(force_file.c_str());
        if (!save_checkpoint(source, force_base, 6)) {
            std::cerr << "force-load fixture save failed\n";
            return 1;
        }

        LOGOSModel exact_target(cfg);
        if (!load_checkpoint_force(exact_target, force_file)) {
            std::remove(force_file.c_str());
            std::cerr << "force-load rejected a complete matching checkpoint\n";
            return 1;
        }
        const auto exact_expected = source.parameters();
        const auto exact_actual = exact_target.parameters();
        if (exact_expected.size() != exact_actual.size()) {
            std::remove(force_file.c_str());
            std::cerr << "force-load parameter count mismatch\n";
            return 1;
        }
        for (size_t i = 0; i < exact_expected.size(); ++i) {
            if (exact_expected[i]->data != exact_actual[i]->data) {
                std::remove(force_file.c_str());
                std::cerr << "force-load changed a complete tensor\n";
                return 1;
            }
        }

        const size_t position = sizeof(ModelConfig) +
            static_cast<size_t>(source.embedding.total_size) * sizeof(float) +
            static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
        const size_t partial_bytes = static_cast<size_t>(source.lm_head.total_size) *
                                    sizeof(float) / 2 + 1;
        std::filesystem::resize_file(force_file, position + partial_bytes);

        LOGOSModel partial_target(cfg);
        std::fill(partial_target.lm_head.data.begin(), partial_target.lm_head.data.end(), 9.0f);
        if (!load_checkpoint_force(partial_target, force_file)) {
            std::remove(force_file.c_str());
            std::cerr << "force-load rejected a truncated tensor\n";
            return 1;
        }
        const size_t complete_floats = partial_bytes / sizeof(float);
        for (size_t i = 0; i < partial_target.lm_head.data.size(); ++i) {
            const float expected_value = i < complete_floats
                ? source.lm_head.data[i] : 0.0f;
            if (partial_target.lm_head.data[i] != expected_value) {
                std::remove(force_file.c_str());
                std::cerr << "force-load did not preserve/zero-fill the partial tensor correctly\n";
                return 1;
            }
        }
        std::remove(force_file.c_str());
    }

    {
        // Atomic save: a stale "<file>.tmp" from a killed run must not matter, no ".tmp" may remain after a
        // successful save, and saving over an existing checkpoint (rename over existing file) must work.
        const std::string at_base = "logos_ckpt_atomic";
        const std::string at_file = at_base + "_step9.bin";
        const std::string at_tmp  = at_file + ".tmp";
        std::remove(at_file.c_str());
        {
            std::ofstream stale(at_tmp, std::ios::binary | std::ios::trunc);
            stale << "half-written garbage from a killed run";
        }
        if (!save_checkpoint(source, at_base, 9)) {
            std::remove(at_tmp.c_str());
            std::cerr << "atomic test: save over a stale .tmp failed\n";
            return 1;
        }
        if (std::filesystem::exists(at_tmp)) {
            std::remove(at_file.c_str());
            std::remove(at_tmp.c_str());
            std::cerr << "atomic test: .tmp file left behind after a successful save\n";
            return 1;
        }
        if (!save_checkpoint(source, at_base, 9)) {
            std::remove(at_file.c_str());
            std::cerr << "atomic test: overwriting an existing checkpoint failed\n";
            return 1;
        }
        LOGOSModel at_target(cfg);
        const bool at_loaded = load_checkpoint(at_target, at_file);
        std::remove(at_file.c_str());
        if (!at_loaded) {
            std::cerr << "atomic test: checkpoint saved via tmp+rename did not load\n";
            return 1;
        }

        OptimizerState at_opt;
        at_opt.velocities = {{1.0f, 2.0f}};
        const std::string at_opt_file = at_base + "_step9.optstate";
        const bool at_opt_ok = save_optimizer_state(at_opt, at_base, 9);
        const bool at_opt_tmp_left = std::filesystem::exists(at_opt_file + ".tmp");
        std::remove(at_opt_file.c_str());
        std::remove((at_opt_file + ".tmp").c_str());
        if (!at_opt_ok || at_opt_tmp_left) {
            std::cerr << "atomic test: optimizer state save failed or left a .tmp file\n";
            return 1;
        }
        std::cout << "atomic save (checkpoint + optimizer state) passed\n";
    }

    {
        // A file that ends INSIDE the BN running stats or the final-LN footer is corrupt (e.g. an interrupted
        // copy), not an "older format". Cuts exactly on the legacy boundaries must keep loading.
        const std::string tr_base = "logos_ckpt_trunc_src";
        const std::string tr_src  = tr_base + "_step10.bin";
        const std::string tr_case = "logos_ckpt_trunc_case.bin";
        std::remove(tr_src.c_str());
        if (!save_checkpoint(source, tr_base, 10)) {
            std::cerr << "truncation fixture save failed\n";
            return 1;
        }
        std::string blob;
        {
            std::ifstream in(tr_src, std::ios::binary);
            std::ostringstream ss;
            ss << in.rdbuf();
            blob = ss.str();
        }
        std::remove(tr_src.c_str());

        const size_t vec    = static_cast<size_t>(cfg.d_model) * sizeof(float);      // one gamma/beta/stat vector
        const size_t footer = 2 * vec;                                                // final-LN gamma + beta
        const size_t stats  = static_cast<size_t>(cfg.num_layers) * 4 * vec;          // BN running stats
        struct Cut { size_t removed; bool ok; const char* what; };
        const Cut cuts[] = {
            {0,                 true,  "complete file"},
            {footer,            true,  "no final-LN footer (older format)"},
            {footer + stats,    true,  "no BN stats and no footer (legacy format)"},
            {1,                 false, "final_ln.beta cut inside"},
            {vec,               false, "final_ln.beta missing while gamma is present"},
            {vec + 1,           false, "final_ln.gamma cut inside"},
            {footer + 1,        false, "ln2.running_var cut inside"},
            {footer + vec,      false, "ln2.running_var missing after earlier stats were read"},
            {footer + stats - 1, false, "ln1.running_mean cut after its first byte"},
            {footer + stats + 1, false, "ln2.beta tensor cut inside"},
        };
        for (const Cut& c : cuts) {
            {
                std::ofstream out(tr_case, std::ios::binary | std::ios::trunc);
                out.write(blob.data(), static_cast<std::streamsize>(blob.size() - c.removed));
            }
            LOGOSModel tr_target(cfg);
            std::ostringstream quiet;                       // expected warnings/errors stay out of the log
            std::streambuf* prev_err = std::cerr.rdbuf(quiet.rdbuf());
            std::streambuf* prev_out = std::cout.rdbuf(quiet.rdbuf());
            const bool loaded = load_checkpoint(tr_target, tr_case);
            std::cerr.rdbuf(prev_err);
            std::cout.rdbuf(prev_out);
            if (loaded != c.ok) {
                std::remove(tr_case.c_str());
                std::cerr << "truncation case '" << c.what << "' (-" << c.removed << " bytes): expected "
                          << (c.ok ? "load OK" : "reject") << ", got "
                          << (loaded ? "load OK" : "reject") << "\n";
                return 1;
            }
        }
        std::remove(tr_case.c_str());
        std::cout << "truncation inside BN stats / final-LN footer is rejected; legacy boundaries still load\n";
    }

    return 0;
}
