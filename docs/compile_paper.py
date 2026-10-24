#!/usr/bin/env python3
"""Compile the IEEE LaTeX paper to PDF."""
import subprocess
import sys
from pathlib import Path

paper = Path(__file__).parent / "paper_ieee.tex"

if not paper.exists():
    print(f"Error: {paper} not found")
    sys.exit(1)

# Check for pdflatex
try:
    subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Error: pdflatex not found. Install TeX Live or MiKTeX.")
    print("  Windows: https://miktex.org/download")
    print("  Linux: apt-get install texlive-full")
    print("  macOS: brew install mactex")
    sys.exit(1)

# Compile sequence for IEEE with bibliography: pdflatex -> bibtex -> pdflatex -> pdflatex
print("\n=== Pass 1/4: pdflatex ===")
result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory=.", str(paper)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Compilation failed:")
    print(result.stdout[-3000:])
    sys.exit(1)

print("\n=== Pass 2/4: bibtex ===")
result = subprocess.run(
    ["bibtex", str(paper.with_suffix(""))],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("BibTeX warning:", result.stdout[-1000:])

print("\n=== Pass 3/4: pdflatex ===")
result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory=.", str(paper)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Compilation failed:")
    print(result.stdout[-3000:])
    sys.exit(1)

print("\n=== Pass 4/4: pdflatex (final) ===")
result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory=.", str(paper)],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Compilation failed:")
    print(result.stdout[-3000:])
    sys.exit(1)

pdf = paper.with_suffix(".pdf")
if pdf.exists():
    print(f"\n✓ PDF generated: {pdf}")
    print(f"  Size: {pdf.stat().st_size / 1024:.1f} KB")
else:
    print("Error: PDF not generated")
    sys.exit(1)