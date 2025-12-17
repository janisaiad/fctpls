# FEPLS: Functional Extreme Partial Least Squares

**Exploring predictive patterns in extreme events through functional data analysis**

What patterns in high-dimensional or functional covariates best predict rare, extreme events? This project implements and empirically validates Functional Extreme Partial Least Squares (FEPLS), a method designed to identify predictive features in functional data associated with extreme responses. We apply FEPLS to financial time series, investigating how intraday return patterns from one asset anticipate large, infrequent moves in another.

To get started, just run `./launch.sh`.

**Implementation details:** This project provides a comprehensive implementation of the FEPLS framework from Girard & Pakzad (2023), including theoretical analysis, parameter calibration procedures ($\tau$ and $k$ tuning), and extensive empirical validation on both medium-frequency (5-minute OHLC) and high-frequency (tick-by-tick) financial data. The work includes detailed diagnostic tools, consistency validation, and practical guidelines for applying FEPLS in real-world settings.

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [Tests](#tests)
- [License](#license)
- [Contact](#contact)

## About

Functional Extreme Partial Least Squares (FEPLS) extends classical Partial Least Squares (PLS) into the domain of extremes, targeting the features of high-dimensional or infinite-dimensional covariates that are most informative for explaining or predicting the occurrence of extreme events in a response variable.

The central research question is:
> *What is the most likely shape of the covariate $X$ when the response $Y$ is extreme?*

FEPLS addresses this by finding a direction $w$ in the space of functional covariates that maximizes the covariance between the projection $\langle w, X \rangle$ and $Y$, *conditionally* on $Y$ exceeding a high threshold $y$. This differs from standard PLS, which maximizes covariance over all data points (average behavior), by focusing specifically on the tail regime that matters most for risk prediction.

**Key theoretical contributions:**
- Second-order regular variation framework for bias-variance tradeoff
- Signal dominance conditions for identifiability
- Optimal convergence rates under heavy-tailed regimes
- Practical parameter calibration procedures ($\tau$ and $k$ tuning)

**Main experimental file:** See `notebooks/new/hypothesis_verif.py` and `notebooks/new/large_scale_hypothesisverif.py` for comprehensive empirical validation.

**Report:** See `refs/report/report.tex` for a detailed theoretical and empirical analysis of the FEPLS framework.

## Installation

To install dependencies using uv, follow these steps:

1. **Install uv:**

   **macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or using wget:
   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, you can install uv using:
   - pipx (recommended): `pipx install uv`
   - pip: `pip install uv`
   - Homebrew: `brew install uv`
   - WinGet: `winget install --id=astral-sh.uv -e`
   - Scoop: `scoop install main/uv`

2. **Using uv in this project:**

   - Initialize a new virtual environment:
   ```bash
   uv venv
   ```

   - Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On Unix
   .venv\Scripts\activate     # On Windows
   ```

   - Install dependencies from requirements.txt:
   ```bash
   uv add -r requirements.txt
   ```

   - Add a new package:
   ```bash
   uv add package_name
   ```

   - Remove a package:
   ```bash
   uv remove package_name
   ```

   - Update a package:
   ```bash
   uv pip install --upgrade package_name
   ```

   - Generate requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

   - List installed packages:
   ```bash
   uv pip list
   ```

## Usage

The main scripts for empirical validation are located in `notebooks/new/`:

- **Hypothesis verification:** `notebooks/new/hypothesis_verif.py` - Validates FEPLS consistency and parameter calibration
- **Large-scale analysis:** `notebooks/new/large_scale_hypothesisverif.py` - Comprehensive empirical evaluation across multiple stock pairs
- **Statistical analysis:** `notebooks/new/stats_analysis.py` - Diagnostic tools and visualization
- **Subsampling analysis:** `notebooks/new/subsampling_apple.py` - High-frequency data analysis using subsampling methodology

Example workflow:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run hypothesis verification
python notebooks/new/hypothesis_verif.py

# Run large-scale analysis
python notebooks/new/large_scale_hypothesisverif.py
```

The analysis uses $X$ as the daily return curve from a stock A and $Y$ as the next day's maximum return from a stock B, reproducing the analysis from the paper using open-source data from [Stooq](https://stooq.com/db/h/).

## Features

- **Theoretical framework:** Complete implementation of FEPLS with second-order regular variation assumptions
- **Parameter calibration:** Automated $\tau$ and $k$ tuning procedures for optimal bias-variance tradeoff
- **Empirical validation:** Extensive experiments on medium-frequency (5-minute OHLC) and high-frequency (tick-by-tick) financial data
- **Diagnostic tools:** Hill estimator, Q-Q plots, conditional quantile analysis, and consistency validation
- **Cross-asset analysis:** Investigation of how intraday patterns in one asset predict extremes in another
- **Reproducible workflow:** Transparent statistical procedures for FEPLS model validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Tests

To run the test suite:

```bash
uv run tests/test_env.py
```

Or using pytest:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Authors:** Janis Aiad, Simon Elis
- **Institution:** Master MVA - Statistical Learning with Extreme Values, ENS Paris Saclay - Ecole Polytechnique - ENS ULM
- **Repository:** [github.com/janisaiad/fctpls](https://github.com/janisaiad/fctpls)

## Warning

If you're using macOS or Python 3, replace `pip` with `pip3` in line 1 of `launch.sh`.

Replace with your project folder name (which means the name of the library you are developing) in `tests/test_env.py`.
