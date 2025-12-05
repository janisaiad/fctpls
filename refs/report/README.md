# FEPLS Analysis Report

This directory contains the LaTeX report for the Functional Extreme Partial Least Squares (FEPLS) analysis applied to financial data.

## Structure

- `report.tex` - Main LaTeX source file
- `report.pdf` - Compiled PDF report
- `figures/` - Directory containing all figures used in the report
  - `4ig_akko_tau_comparison.png` - Comparison plot for 4IG → AKKO across tau values
  - `4ig_akko_tau_0.0.png` - Detailed analysis for 4IG → AKKO with tau = 0.0
  - `4ig_akko_tau_-0.5.png` - Detailed analysis for 4IG → AKKO with tau = -0.5
  - `akko_4ig_tau_comparison.png` - Comparison plot for AKKO → 4IG across tau values
  - `akko_4ig_tau_-1.0.png` - Detailed analysis for AKKO → 4IG with tau = -1.0

## Compilation

To compile the report:

```bash
pdflatex report.tex
pdflatex report.tex  # Run twice for proper cross-references
```

## Content

The report includes:
- Theoretical framework of FEPLS
- Two main theorems (explicit solution and consistency)
- Detailed explanations of bounds and the parameter ρ
- Empirical analysis of 4IG and AKKO stock pairs
- Visualizations and interpretations

## Data Source

The figures are generated from the analysis in `notebooks/new/large_scale_tau_traintest.py` and stored in `results/tau_comparison_plots/`.

