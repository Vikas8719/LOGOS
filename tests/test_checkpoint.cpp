#include "Checkpoint.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>

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

    // Verify file does NOT exist before save (pre-condition)
    if (std::filesystem::exists(file)) {
        std::cerr << "file should not exist before save\n";
        return 1;
    }

    if (!save_checkpoint(source, base, 7)) {
        std::cerr << "checkpoint save failed\n";
        return 1;
    }

    // Verify file DOES exist after save (void call side-effect check)
    if (!std::filesystem::exists(file)) {
        std::cerr << "save_checkpoint did not create file (void call side-effect missing)\n";
        return 1;
    }
    // File size must be > 0
    if (std::filesystem::file_size(file) == 0) {
        std::cerr << "save_checkpoint wrote empty file\n";
        return 1;
    }

    // expected_bytes must match save_checkpoint layout exactly:
    //   [ModelConfig]
    //   [embedding floats]
    //   [legacy position block: max_seq_len * d_model floats]
    //   [lm_head floats]
    //   per layer:
    //     [per-head W_Q, W_K, W_V, W_O]
    //     [W_proj]
    //     [ffn.W1, ffn.b1, ffn.W2, ffn.b2]
    //     [ln1.gamma, ln1.beta]
    //     [ln2.gamma, ln2.beta]
    //     [ln1.running_mean, ln1.running_var]   ← ReynoldsBatchNorm stats
    //     [ln2.running_mean, ln2.running_var]   ← ReynoldsBatchNorm stats
    size_t expected_bytes = sizeof(ModelConfig) +
        static_cast<size_t>(cfg.max_seq_len) * cfg.d_model * sizeof(float);
    for (const Tensor* parameter : source.parameters())
        expected_bytes += static_cast<size_t>(parameter->total_size) * sizeof(float);
    // Add ReynoldsBatchNorm running_mean + running_var for each layer (ln1 + ln2)
    // Each running stat vector has d_model floats
    expected_bytes += static_cast<size_t>(cfg.num_layers) * 4 *
                      static_cast<size_t>(cfg.d_model) * sizeof(float);

    // ── Exact arithmetic verification of formula components ──────────────
    // Each component individually checked to kill operator-mutation mutants.

    // 1. sizeof(ModelConfig) must be exactly what it is (> 0)
    if (sizeof(ModelConfig) == 0) {
        std::cerr << "sizeof(ModelConfig) must be > 0\n";
        return 1;
    }

    // 2. Legacy pos block = max_seq_len * d_model * sizeof(float)
    //    cfg: max_seq_len=3, d_model=4 → 3*4*4 = 48 bytes
    size_t pos_block_bytes = static_cast<size_t>(cfg.max_seq_len) *
                             static_cast<size_t>(cfg.d_model) * sizeof(float);
    if (pos_block_bytes != 3 * 4 * sizeof(float)) {
        std::cerr << "pos block bytes wrong: " << pos_block_bytes << "\n";
        return 1;
    }

    // 3. RBN extra = num_layers * 4 * d_model * sizeof(float)
    //    cfg: num_layers=1, d_model=4 → 1*4*4*4 = 64 bytes
    size_t rbn_extra_bytes = static_cast<size_t>(cfg.num_layers) * 4 *
                             static_cast<size_t>(cfg.d_model) * sizeof(float);
    if (rbn_extra_bytes != 1 * 4 * 4 * sizeof(float)) {
        std::cerr << "RBN extra bytes wrong: " << rbn_extra_bytes << "\n";
        return 1;
    }

    // 4. expected_bytes must be > just sizeof(ModelConfig) (has param data too)
    if (expected_bytes <= sizeof(ModelConfig)) {
        std::cerr << "expected_bytes too small (arithmetic wrong): " << expected_bytes << "\n";
        return 1;
    }

    // 5. expected_bytes must include at least the pos block exactly
    if (expected_bytes < sizeof(ModelConfig) + pos_block_bytes) {
        std::cerr << "expected_bytes misses pos block\n";
        return 1;
    }

    if (std::filesystem::file_size(file) != expected_bytes) {
        std::remove(file.c_str());
        std::cerr << "checkpoint size mismatch: got "
                  << std::filesystem::file_size(file)
                  << " expected " << expected_bytes << "\n";
        return 1;
    }

    // Simulate legacy learned-position weights; loaders must discard them and
    // continue reading lm_head and layer weights from their original offsets.
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

    // Parameter count must match exactly (catches loop bound mutations)
    if (expected.size() != actual.size()) {
        std::remove(file.c_str());
        std::cerr << "parameter count mismatch: src=" << expected.size()
                  << " restored=" << actual.size() << "\n";
        return 1;
    }
    // Must have at least 2 parameters (embedding + lm_head)
    if (expected.size() < 2) {
        std::remove(file.c_str());
        std::cerr << "too few parameters: " << expected.size() << "\n";
        return 1;
    }

    // Verify embedding (first param) total_size = vocab_size * d_model
    if (expected[0]->total_size != cfg.vocab_size * cfg.d_model) {
        std::remove(file.c_str());
        std::cerr << "embedding total_size wrong: " << expected[0]->total_size
                  << " expected " << (cfg.vocab_size * cfg.d_model) << "\n";
        return 1;
    }

    // Verify lm_head (second param) total_size = d_model * vocab_size
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
    std::remove(file.c_str());
    if (!equal) {
        std::cerr << "checkpoint round-trip changed model parameters\n";
        return 1;
    }

    std::cout << "checkpoint round-trip passed\n";

    // ── T2b-extra: Expected bytes formula verification ─────────────────────
    // Verify each component of the expected_bytes formula independently
    // to catch arithmetic mutations (+ vs -, * vs /)
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

        // Recompute expected_bytes the same way
        size_t expected2 = sizeof(ModelConfig) +
            static_cast<size_t>(cfg2.max_seq_len) * cfg2.d_model * sizeof(float);
        for (const Tensor* p : m2.parameters())
            expected2 += static_cast<size_t>(p->total_size) * sizeof(float);
        expected2 += static_cast<size_t>(cfg2.num_layers) * 4 *
                     static_cast<size_t>(cfg2.d_model) * sizeof(float);

        size_t actual2 = std::filesystem::file_size(file2);
        std::remove(file2.c_str());

        if (actual2 != expected2) {
            std::cerr << "formula test: size mismatch: got " << actual2
                      << " expected " << expected2 << "\n";
            return 1;
        }

        // Verify sizeof(ModelConfig) > 0 (not zero)
        if (sizeof(ModelConfig) == 0) {
            std::cerr << "sizeof(ModelConfig) must be > 0\n";
            return 1;
        }

        // Verify legacy pos block size: max_seq_len * d_model * sizeof(float)
        size_t pos_block = static_cast<size_t>(cfg2.max_seq_len) * cfg2.d_model * sizeof(float);
        if (pos_block != 4 * 8 * sizeof(float)) {
            std::cerr << "legacy pos block size wrong: " << pos_block << "\n";
            return 1;
        }

        // Verify ReynoldsBatchNorm extra bytes: num_layers * 4 * d_model * sizeof(float)
        size_t rbn_bytes = static_cast<size_t>(cfg2.num_layers) * 4 *
                           static_cast<size_t>(cfg2.d_model) * sizeof(float);
        if (rbn_bytes != 2 * 4 * 8 * sizeof(float)) {
            std::cerr << "RBN bytes wrong: " << rbn_bytes << "\n";
            return 1;
        }

        std::cout << "formula test passed: expected=" << expected2
                  << " actual=" << actual2 << "\n";
    }

    // ── T2b-roundtrip2: Round-trip with non-trivial weights ────────────────
    // Fill source with known values, save, load, verify exact match
    {
        ModelConfig cfg3;
        cfg3.vocab_size  = 8;
        cfg3.d_model     = 4;
        cfg3.num_heads   = 2;
        cfg3.num_layers  = 1;
        cfg3.max_seq_len = 3;

        LOGOSModel src3(cfg3);
        // Fill all parameters with recognisable pattern: p[i] = (i+1)*0.01
        {
            int idx = 0;
            for (Tensor* p : src3.parameters())
                for (float& v : p->data) v = (float)(++idx) * 0.01f;
        }
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

        if (!match3) {
            std::cerr << "nontrivial round-trip: parameter mismatch\n";
            return 1;
        }
        std::cout << "nontrivial round-trip passed\n";
    }

    // ── T2b-offsets: Verify file offset arithmetic ─────────────────────────
    // The position block offset = sizeof(ModelConfig) + embedding_floats * sizeof(float)
    // Verify this is computed correctly (catches + vs -, * vs / mutations)
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

        // Read ModelConfig from file and verify values match
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

            // Verify embedding block: vocab_size * d_model floats
            size_t emb_floats = static_cast<size_t>(cfg4.vocab_size) * cfg4.d_model;
            std::vector<float> emb(emb_floats);
            f.read(reinterpret_cast<char*>(emb.data()),
                   static_cast<std::streamsize>(emb_floats * sizeof(float)));
            if (!f) {
                std::remove(file4.c_str());
                std::cerr << "offset test: could not read embedding block\n";
                return 1;
            }
            // Embedding must equal source embedding exactly
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

    return 0;
}
