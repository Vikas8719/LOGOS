@echo off
REM ============================================================
REM  LOGOS arXiv Paper — PDF Build Script (Windows / MiKTeX)
REM  Usage: build_paper.bat
REM  Requirements: MiKTeX or TeX Live for Windows
REM ============================================================
cd /d "%~dp0"

echo ============================================
echo   LOGOS arXiv Paper -- Building PDF
echo ============================================

echo [1/4] pdflatex pass 1...
pdflatex -interaction=nonstopmode logos_arxiv.tex
if errorlevel 1 goto error

echo [2/4] bibtex...
bibtex logos_arxiv

echo [3/4] pdflatex pass 2...
pdflatex -interaction=nonstopmode logos_arxiv.tex

echo [4/4] pdflatex pass 3...
pdflatex -interaction=nonstopmode logos_arxiv.tex

echo.
echo ✅ Build complete! Output: logos_arxiv.pdf
echo.
echo Next steps for arXiv:
echo   1. Review logos_arxiv.pdf
echo   2. https://arxiv.org/submit
echo   3. Category: cs.CL or cs.LG
echo   4. Upload logos_arxiv.tex + logos_arxiv.bbl
goto end

:error
echo.
echo ❌ Build failed. Check logos_arxiv.log for details.

:end
pause
