# LOGOS — Vedic-Physics Hybrid LLM

**Language:** C++20 / CUDA | **Parameters:** ~219M | **Context:** 8192 Tokens | **Version:** v30

## Architecture (Production)

```
d=1024 | L=16 | H=16 | DH=64 | seq=8192 | vocab=8192
```

```
INPUT TEXT → [BPE Tokenizer: 8192 vocab]
           → [Hyperbolic Embedding: expmap₀ Poincaré ball]
           → [16× Transformer Block]:
               LN1 (Reynolds-adaptive) → MHA (Flash Attn + Shunyam sparse + Nikhilam INT8 KV)
                                       → NS Q-advect + V-diffuse
               LN2 (Reynolds-adaptive) → FFN (GELU + Feynman β-dropout)
           → [LM Head: Vedic GEMM]
           → [Free Energy Loss: F = CE − T·S]
           → TOKEN
```

## VRAM Budget (Tesla T4 16 GB)
| Component | Size |
|-----------|------|
| Weights FP32 | ~876 MB |
| FP16 shadow (AMP) | ~438 MB |
| Nikhilam INT8 KV | ~268 MB |
| Activations (grad_accum=4) | ~680 MB |
| Optimizer velocity | ~876 MB |
| **Total estimate** | **~3.1 GB / 15 GB** ✅ |

## Physics Innovations
| Component | Description |
|-----------|-------------|
| **Vedic GEMM** | Urdhva Tiryagbhyam matrix multiply + cuBLAS fallback |
| **SHM Optimizer** | Hamiltonian + Langevin hybrid (α_H + α_L = 1) |
| **Free Energy Loss** | F = CE − T·S (temperature-weighted entropy bonus) |
| **Feynman Dropout** | β-amplitude dropout with backward mask chain |
| **Navier-Stokes Attn** | Q-advection (η=0.1) + V-diffusion (ν=0.05) |
| **Reynolds LayerNorm** | Adaptive normalization via turbulence Re number |
| **Nikhilam KV** | INT8 KV cache (4× VRAM reduction) |
| **Shunyam Sparse** | O(seq × window) attention (128× vs O(seq²)) |
| **Flash Attention** | Custom CUDA kernel: O(seq·DH) memory (vs O(seq²)) |
| **Hyperbolic Embedding** | Poincaré ball via expmap₀ with Jacobian backward |
| **Path Integral LR** | WeightPathIntegral adaptive LR scaling |

## Build (GPU)
```bash
# Kaggle / RunPod — CUDA required
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run training
./build/logos_gpu --train dataset.txt

# Run inference
./build/logos_gpu --generate logos_gpu_ckpt_step60000.bin "Once upon a time" 128 4 1.0 50
```

## Key Env Overrides
| Variable | Default | Description |
|----------|---------|-------------|
| `LOGOS_LR` | 2e-3 | Learning rate |
| `LOGOS_CLIP` | 0.3 | Gradient clip norm |
| `LOGOS_STEPS` | 150000 | Total training steps |
| `LOGOS_SEQ_LEN` | 8192 (4096 on T4) | Sequence length |
| `LOGOS_GRAD_ACCUM` | 4 (T4), 16 (A100), 32 (H100) | Gradient accumulation |
| `LOGOS_CHUNK_MB` | 4 (T4), 32 (A100), 64 (H100) | Data prefetch chunk |
| `LOGOS_CKPT_FREQ` | 2000 (T4), 5000 (H100) | Checkpoint save frequency |
| `LOGOS_LOG_FREQ` | 500 | Training log print frequency |
| `LOGOS_CKPT` | — | Resume checkpoint path |
| `LOGOS_VOCAB` | vocab.bin | Tokenizer vocab file |
| `LOGOS_START_STEP` | 0 | Resume step offset |
| `LOGOS_BEST_F` | — | Resume best Free Energy |

## Version History
| Version | Key Change |
|---------|-----------|
| v30 | Dataset auto-resolve + vocab.bin skip-rebuild (v33-FAST) |
| v29 | Full resume: DataState + TrainingState + OptState |
| v27 | H100 tweaks: adaptive chunk/ckpt, prefetch safety, RAII leaks |
| v26 | Security: path traversal fix, env range clamp |
| v25 | AMP double-scale bug fix, RAII training guard |
| v24 | Manual AMP loss scaler wired end-to-end |
| v22 | Scale 17M → 219M (d=1024 L=16 H=16 seq=8192) |
| v17 | 1.5 GB Wikipedia Hi+En dataset, bias grads fixed |
| v15 | Feynman dropout bwd + Hyperbolic expmap bwd (100% wired) |

## Research Goals
1. **Vedic GEMM Speed**: Urdhva Tiryagbhyam vs cuBLAS (same model size)
2. **Physics Stability**: SHM (Hamiltonian+Langevin) vs Adam (loss curve)
3. **Free Energy Training**: F=CE−T·S vs pure CE loss (generalization)
4. **Hindi+English Bilingual**: 219M trained on Wikipedia Hi+En (1.5 GB)
