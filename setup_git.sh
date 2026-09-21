#!/bin/bash
# ============================================================
#  LOGOS — GitHub repo setup (Linux/macOS)
#  Sirf ek baar chalao — pehli baar push ke liye
# ============================================================

set -e  # Koi bhi error → script band

echo ""
echo "===== LOGOS Git Setup ====="
echo ""

# Git check
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git install nahi hai!"
    echo "Ubuntu: sudo apt install git"
    echo "macOS:  brew install git"
    exit 1
fi

echo "[1/5] Git repo initialize..."
git init

echo "[2/5] Saari files stage..."
git add .

echo "[3/5] Pehla commit..."
git commit -m "feat: LOGOS Vedic-Physics LLM — initial project structure

- Tensor.hpp: Row-Major data store
- VedicGEMM.cpp: Urdhva Tiryagbhyam tiled GEMM
- Tokenizer.cpp: BPE tokenizer
- Attention.cpp: MultiHead + Boltzmann softmax
- FeedForward.cpp: GELU + Vedic GEMM
- TransformerBlock.cpp: Full transformer layer
- PhysicsOpt.cpp: Langevin Dynamics optimizer
- Model.cpp: Complete 10M param model
- Trainer.cpp + Checkpoint.cpp
- CI/CD: GitHub Actions (Linux/Windows/macOS)"

echo "[4/5] Branch → main..."
git branch -M main

echo ""
echo "===== ABHI YE KARO ====="
echo ""
echo "1. GitHub pe naya repo banao:"
echo "   https://github.com/new"
echo "   Name: LOGOS"
echo "   README add mat karo (already hai)"
echo ""
read -p "GitHub repo URL daalo (e.g. https://github.com/user/LOGOS): " REPO_URL

echo "[5/5] Remote add + push..."
git remote add origin "$REPO_URL"
git push -u origin main

echo ""
echo "===== DONE! ====="
echo "GitHub Actions ab automatically build karega."
echo "Check karo: ${REPO_URL}/actions"
echo ""
