# LOGOS — Physics-Informed Hybrid LLM

**Language:** C++20 / CUDA &nbsp;|&nbsp; **Parameters:** ~219 M &nbsp;|&nbsp; **Context:** 8 192 tokens &nbsp;|&nbsp; **Version:** v30

> A research language model that replaces the standard Adam optimizer with a **Symplectic Hamiltonian–Langevin (SHM)** hybrid, adds a **Feynman path-integral adaptive LR**, and trains under a **Free Energy loss** F = CE − T·S. Built and tested entirely in C++20/CUDA with zero Python in the hot path.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Physics Innovations](#physics-innovations)
3. [Project Layout](#project-layout)
4. [Build & Run](#build--run)
5. [Hyperparameter Sweep Tool](#hyperparameter-sweep-tool)
6. [Environment Variables](#environment-variables)
7. [Test Suite & Mutation Coverage](#test-suite--mutation-coverage)
8. [Version History](#version-history)
9. [Research Goals](#research-goals)

---

## Architecture

```
d=1024 | L=16 | H=16 | DH=64 | seq=8 192 | vocab=8 192
```

```
INPUT TEXT ──► BPE Tokenizer (vocab=8 192, byte-level GPT-2 style)
           ──► Hyperbolic Embedding  (expmap₀, Poincaré ball)
           ──► 16 × Transformer Block
               ├─ LN₁ (Reynolds-adaptive) ──► MHA (Flash Attn, causal)
               │                                └─ NS Q-advect + V-diffuse
               └─ LN₂ (Reynolds-adaptive) ──► FFN (GELU + Feynman β-dropout)
           ──► Final LayerNorm
           ──► LM Head  (cuBLAS GEMM)
           ──► Free Energy Loss  F = CE − T·S
           ──► SHM Optimizer  (α_H + α_L = 1)
                └─ WeightPathIntegral adaptive LR scale
```

### VRAM Budget — Tesla T4 (16 GB)

| Component | Size |
|---|---|
| Weights FP32 | ~876 MB |
| FP16 AMP shadow | ~438 MB |
| Activations (grad\_accum=4) | ~680 MB |
| SHM velocity buffers | ~876 MB |
| **Total** | **~2.9 GB / 15 GB ✅** |

---

## Physics Innovations

| Component | Description |
|---|---|
| **SHM Optimizer** | Symplectic Hamiltonian + Langevin hybrid; α_H + α_L = 1 enforced at every step |
| **Free Energy Loss** | F = CE − T·S — temperature-weighted entropy bonus encourages exploration |
| **Feynman Dropout** | β-amplitude dropout; backward mask chain is fully differentiable |
| **Navier-Stokes Attention** | Q-advection (η = 0.1) + V-diffusion (ν = 0.05) in every attention head |
| **Reynolds LayerNorm** | Adaptive normalization via turbulence Reynolds number estimate |
| **Flash Attention** | Custom CUDA kernel — O(seq·DH) VRAM vs O(seq²) for vanilla attention |
| **Hyperbolic Embedding** | Poincaré ball via expmap₀ with full Jacobian backward pass |
| **Path Integral LR** | `WeightPathIntegral` — EMA of action (loss × ‖Δθ‖) drives adaptive LR scaling |

---

## Project Layout

```
LOGOS/
├── include/                   # Header-only C++ library
│   ├── Checkpoint.hpp         # Atomic save/load, legacy format backward-compat
│   ├── Model.hpp              # LOGOSModel, ModelConfig, weight structs
│   ├── PathIntegral.hpp       # WeightPathIntegral adaptive LR
│   ├── StreamingDataLoader.hpp
│   ├── Tensor.hpp
│   ├── TokenBin.hpp           # Pre-tokenised .bin format + loader
│   ├── Tokenizer.hpp          # Byte-level BPE, persistent piece cache
│   └── TrainingState.hpp
├── src/                       # GPU training + inference binary
├── cuda/                      # Custom CUDA kernels
├── tests/
│   ├── test_checkpoint.cpp    # Checkpoint round-trip, mutation-gap tests
│   ├── cpu/
│   │   ├── test_tokenizer.cpp # BPE unit + cache boundary mutation tests
│   │   ├── test_rapidcheck.cpp# Property-based tests (RapidCheck)
│   │   ├── test_fuzz.cpp      # Seeded deterministic fuzz (3 000 iters default)
│   │   └── test_integration.cpp
│   └── fuzz/
│       ├── fuzz_checkpoint.cpp# libFuzzer target + strict invariant assertions
│       ├── fuzz_tokenizer.cpp
│       ├── fuzz_tokenbin.cpp
│       └── fuzz_vocab.cpp
├── tools/
│   └── sweep_lr_clip.py       # LR × CLIP hyperparameter sweep (Kaggle/RunPod)
├── paper/                     # Research notes
├── mull.yml                   # Mull mutation-testing config (cxx_all scope)
└── CMakeLists.txt
```

---

## Build & Run

> **Requires:** CUDA 11.8+, CMake ≥ 3.20, GCC/Clang with C++20 support. Tested on Kaggle (Tesla T4) and RunPod (A100/H100).

```bash
# Configure + build (GPU binary)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Train from scratch
./build/logos_gpu --train /path/to/dataset.txt

# Resume from checkpoint
LOGOS_CKPT=logos_gpu_ckpt_step40000.bin ./build/logos_gpu --train /path/to/dataset.txt

# Inference / generation
./build/logos_gpu --generate logos_gpu_ckpt_step60000.bin \
    "Once upon a time" \
    128   `# max tokens`  \
    4     `# top-k`       \
    1.0   `# temperature` \
    50    `# repetition window`
```

### CPU Tests (no GPU needed)

```bash
cmake -B build_cpu -DCMAKE_BUILD_TYPE=Debug -DLOGOS_CPU_TESTS=ON
cmake --build build_cpu --parallel
ctest --test-dir build_cpu --output-on-failure
```

---

## Hyperparameter Sweep Tool

`tools/sweep_lr_clip.py` runs a systematic **LR × CLIP grid sweep** directly on the `logos_gpu` binary (Kaggle / RunPod environment). Each configuration is a fully isolated run (separate working directory) so checkpoints and vocab files never collide.

```bash
# Stage 1 — find best learning rate (fixed clip=0.3)
python tools/sweep_lr_clip.py \
    --binary build/logos_gpu \
    --dataset /kaggle/working/dataset.txt \
    --stage lr \
    --steps 400

# Stage 2 — sweep clip values (uses best LR from stage 1)
python tools/sweep_lr_clip.py \
    --binary build/logos_gpu \
    --dataset /kaggle/working/dataset.txt \
    --stage clip \
    --steps 400

# Full grid (expensive)
python tools/sweep_lr_clip.py --stage grid --steps 400
```

**Features:**
- Resumes interrupted sweeps automatically (`sweep_out/results.json` persisted after every run)
- First run builds `vocab.bin`; subsequent runs reuse it
- Heavy binary files (`.bin`, `.optstate`) deleted after each run to save disk
- Results ranked by validation CE; best config printed at the end
- CSV export at `sweep_out/results.csv`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOGOS_LR` | `2e-3` | Learning rate |
| `LOGOS_CLIP` | `0.3` | Gradient clip norm |
| `LOGOS_STEPS` | `500 000` | Total training steps |
| `LOGOS_SEQ_LEN` | `8192` (T4: `4096`) | Sequence length |
| `LOGOS_GRAD_ACCUM` | `4` (T4) / `16` (A100) / `32` (H100) | Gradient accumulation steps |
| `LOGOS_CHUNK_MB` | `4` (T4) / `32` (A100) / `64` (H100) | Data prefetch chunk size |
| `LOGOS_CKPT_FREQ` | `2000` (T4) / `5000` (H100) | Checkpoint save frequency (steps) |
| `LOGOS_LOG_FREQ` | `500` | Training log print frequency (steps) |
| `LOGOS_CKPT` | _(none)_ | Resume checkpoint path |
| `LOGOS_VOCAB` | `vocab.bin` | Tokenizer vocabulary file |
| `LOGOS_START_STEP` | `0` | Resume step offset |
| `LOGOS_BEST_F` | _(none)_ | Resume best Free Energy value |
| `LOGOS_DROPOUT_SEED` | `1234` | Feynman dropout RNG seed (sweep: fixed for fair comparison) |
| `LOGOS_FUZZ_ITERS` | `3000` | `test_fuzz` iteration count |
| `LOGOS_FUZZ_SEED` | `1337` | `test_fuzz` RNG seed (print on failure to reproduce) |

---

## Test Suite & Mutation Coverage

LOGOS uses a **four-layer testing strategy** aimed at >85% mutation-kill rate across `include/` and `src/`.

### Layer 1 — Unit Tests (`tests/test_checkpoint.cpp`, `tests/cpu/test_tokenizer.cpp`)

| Test | What it kills |
|---|---|
| `cfgs_match` vocab_size `!=→==` | Mismatch check inverts — exact match would be rejected |
| `cfgs_match` pos_embedding `==→!=` | Loaded pos weights silently diverge from saved |
| Temp-file cleanup assertion | `.tmp` left on disk after atomic save |
| `CACHE_MAX_ENTRIES >=→>` | Cache never evicts; memory grows unboundedly |
| `cache_.clear()` void-call | Cache fills forever; eviction semantics broken |
| Cache boundary idempotence | Post-evict encoding differs from cold encoding |

### Layer 2 — Property-Based Tests (`tests/cpu/test_rapidcheck.cpp`)

RapidCheck runs 1 000 examples per property by default (`RC_PARAMS=max_success=1000`).

| Property | Mutation killed |
|---|---|
| `decode(encode_raw(x)) == x` for arbitrary / WS / UTF-8 inputs | Any byte-mapping off-by-one |
| Piece concat == input, each 1–128 bytes | Pre-tokeniser cap logic |
| Encoding additive at piece boundaries | Merge-rank lookup mutation |
| `encode_raw(x, m)` is prefix of full encoding | `max_tokens` fence-post |
| `encode(x, L)` BOS/EOS framing | Special-token insertion mutations |
| `save()/load()` fingerprint stable | Serialisation field-swap |
| Cache-state independence (evict + re-encode) | `cache_.clear()` noop + `>=→>` |
| `WeightPathIntegral` monotone / NaN / range invariants | Six PathIntegral mutants |
| TokenBin `pretokenize_file` + loader shift-by-1 | TokenBin header arithmetic |

### Layer 3 — Seeded Fuzz (`tests/cpu/test_fuzz.cpp`)

Deterministic, reproducible with `LOGOS_FUZZ_SEED`. Each violation counter targets one mutant class:

| Counter | Mutant killed |
|---|---|
| `v_monotone` | `log_amplitude -= action` → `+=` |
| `v_cum_mono` | `cumulative_action += action` → `-=` |
| `v_step_mono` | `++step_count` removed |
| `v_lr_hi` | `+1e-8f` → `-1e-8f` in `lr_scale_ema` denom |
| `v_denom_pos` | `std::max(ema_action, 0.0f)` removed |
| `v_amp_range` | `exp(log_amplitude)` sign flip |
| `v_nan_*` | `isfinite` guard on action removed |

### Layer 4 — libFuzzer (`tests/fuzz/fuzz_checkpoint.cpp`)

libFuzzer with structure-aware mutations (`fz::mutate_good`). After a **successful** load, six invariants are checked and `std::abort()` on violation (fuzzer saves the input as a crash):

1. `parameters()` non-empty, every tensor `total_size > 0`
2. All loaded weights are finite (no NaN/Inf)
3. `cfg` fields match the model's expected config
4. `layers.size() == cfg.num_layers`
5. `pos_embedding.size() == max_seq_len * d_model`
6. `final_ln.{gamma,beta}` sizes correct and values finite

### Running Mutation Tests (Mull)

```bash
# Install Mull (requires LLVM/Clang toolchain)
# Config: mull.yml (scope = include/*.hpp + src/*.cpp, cxx_all mutators)
mull-runner ./build_cpu/logos_cpu_tests
```

---

## Version History

| Version | Key Change |
|---|---|
| v30 | Dataset auto-resolve + `vocab.bin` skip-rebuild; mutation test gaps fixed |
| v29 | Full resume: `DataState` + `TrainingState` + `OptState` |
| v27 | H100 tweaks: adaptive chunk/ckpt, prefetch safety, RAII leak fixes |
| v26 | Security: path-traversal fix, env-var range clamp |
| v25 | AMP double-scale bug fix, RAII training guard |
| v24 | Manual AMP loss scaler wired end-to-end |
| v22 | Scale 17 M → 219 M (d=1024, L=16, H=16, seq=8192) |
| v17 | 1.5 GB Wikipedia Hi+En dataset, bias gradients fixed |
| v15 | Feynman dropout bwd + Hyperbolic expmap bwd (100% wired) |

---

## Research Goals

1. **Physics Stability** — Compare SHM (Hamiltonian + Langevin) vs Adam on training loss curves.
2. **Free Energy Training** — Does F = CE − T·S generalise better than pure CE loss?
3. **Hindi + English Bilingual** — 219 M parameters trained on 1.5 GB Wikipedia Hi+En.
