#!/usr/bin/env bash
# ============================================================
#  LOGOS arXiv Paper — PDF Build Script
#  Usage: bash build_paper.sh
#  Requirements: pdflatex + bibtex (TeX Live / MiKTeX)
# ============================================================
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  LOGOS arXiv Paper — Building PDF"
echo "============================================"

# First pass
echo "[1/4] pdflatex pass 1..."
pdflatex -interaction=nonstopmode logos_arxiv.tex

# Bibliography
echo "[2/4] bibtex..."
bibtex logos_arxiv || true

# Second pass (resolve refs)
echo "[3/4] pdflatex pass 2..."
pdflatex -interaction=nonstopmode logos_arxiv.tex

# Third pass (finalise)
echo "[4/4] pdflatex pass 3..."
pdflatex -interaction=nonstopmode logos_arxiv.tex

echo ""
echo "✅ Build complete! Output: logos_arxiv.pdf"
echo ""
echo "Word count (approx):"
pdftotext logos_arxiv.pdf - 2>/dev/null | wc -w || true

echo ""
echo "Next steps for arXiv submission:"
echo "  1. Review logos_arxiv.pdf"
echo "  2. Create account at https://arxiv.org"
echo "  3. Submit to cs.CL (Computation and Language)"
echo "     or cs.LG (Machine Learning)"
echo "  4. Upload: logos_arxiv.tex + logos_arxiv.bbl"
echo "  5. Wait 1-2 business days for moderation"
