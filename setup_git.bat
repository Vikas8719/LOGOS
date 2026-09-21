@echo off
REM ============================================================
REM  LOGOS — GitHub repo setup (Windows)
REM  Sirf ek baar chalao — pehli baar push ke liye
REM  Baad mein: git add . && git commit -m "msg" && git push
REM ============================================================

echo.
echo ===== LOGOS Git Setup =====
echo.

REM Check git installed hai ya nahi
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git install nahi hai!
    echo Download: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/5] Git repo initialize kar raha hai...
git init

echo [2/5] Saari files stage kar raha hai...
git add .

echo [3/5] Pehla commit...
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

echo [4/5] Branch ka naam 'main' rakho...
git branch -M main

echo.
echo ===== ABHI YE KARO =====
echo.
echo 1. GitHub pe naya repo banao:
echo    https://github.com/new
echo    Name: LOGOS
echo    Private ya Public — tumhari choice
echo    README add mat karo (already hai)
echo.
echo 2. Phir yahan repo URL daalo:
set /p REPO_URL="GitHub repo URL daalo (e.g. https://github.com/username/LOGOS): "

echo [5/5] Remote add karke push kar raha hai...
git remote add origin %REPO_URL%
git push -u origin main

echo.
echo ===== DONE! =====
echo GitHub Actions automatically build shuru ho jayega.
echo Check karo: %REPO_URL%/actions
echo.
pause
