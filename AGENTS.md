# AGENTS.md

## Project Context
This repository implements a Monte Carlo simulation engine based on Geometric Brownian Motion (GBM) for equity-price scenario analysis. The project focuses on estimating ranges of probable outcomes, not deterministic prediction.

Primary technologies:
- Python
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `yfinance`
- `scipy`

## Project Structure
The repository is intentionally small and centered on three top-level files:

- `monte_carlo_simulatio_proyect.py`: main executable script for downloading data, running GBM simulations, and plotting results.
- `MONTE_CARLO_SIMULATIO_PROYECT.ipynb`: notebook for exploratory analysis and presentation.
- `README.md`: essay-style overview, methodology, and supporting context.

There is no `src/`, `tests/`, or `assets/` directory yet. If reusable logic grows, extract it into small helper modules instead of expanding the top-level script further.

## Environment Setup
This project has no build system. Typical local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib yfinance scipy seaborn jupyter
```

Runtime note:
- Internet access is required for `yfinance` downloads.

Common entry points:

```bash
python monte_carlo_simulatio_proyect.py
jupyter notebook MONTE_CARLO_SIMULATIO_PROYECT.ipynb
```

## Coding Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for variables, functions, and module names.
- Prefer English identifiers in code, even if surrounding documentation remains bilingual.
- Keep ticker lists, simulation parameters, data acquisition, and plotting logic clearly separated.
- If formatting or linting is introduced, prefer `black` and `ruff`.

## Testing And Validation
There is no automated test suite yet.

For new logic:
- Add focused `pytest` tests under `tests/`.
- Use names like `test_simulation.py`.
- Prefer deterministic helper tests by seeding NumPy RNGs.
- Avoid tests that depend on plot rendering.

Before submitting changes:
- Run the script once.
- Confirm data download completes.
- Confirm simulation output is produced.
- Confirm plots render without errors.

## Commit And PR Guidance
- Use short, imperative commit messages.
- Keep each commit scoped to one logical change.
- Recent history includes messages like `Hot fixes`, but prefer slightly more specific variants when possible.

Pull requests should include:
- A brief summary of the modeling or code change.
- Any dependency or runtime impact.
- Sample output or screenshots when plots change.
- Linked issue or rationale when fixing a bug.

## Configuration Notes
- Do not hardcode credentials or private data.
- External market data is fetched live, so document ticker, horizon, or simulation-count changes when reproducibility matters.
- Treat `yfinance` output shape as variable across versions; write code that handles both flat and MultiIndex column layouts when possible.
