import numpy as np
import pandas as pd

from portfolio_analysis import build_portfolio_analytics


def build_synthetic_price_frame():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=260)

    sector_tickers = [
        "AAPL",
        "MSFT",
        "AVGO",
        "UNH",
        "JNJ",
        "AMGN",
        "WMT",
        "COST",
        "PG",
    ]

    market_returns = rng.normal(0.00035, 0.01, len(dates))
    data = {}
    data["SPY"] = 100 * np.cumprod(1 + market_returns)

    for index, ticker in enumerate(sector_tickers):
        beta = 0.65 + 0.05 * (index % 3)
        idiosyncratic = rng.normal(0.00015, 0.008, len(dates))
        returns = 0.00025 + beta * market_returns + idiosyncratic
        data[ticker] = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(data, index=dates)


def test_build_portfolio_analytics_produces_core_outputs():
    frame = build_synthetic_price_frame()
    result = build_portfolio_analytics(frame)

    assert not result.sector_summary.empty
    assert result.correlation_matrix.shape[0] == result.correlation_matrix.shape[1]
    assert np.isclose(result.pca_explained_variance_ratio.sum(), 1.0)
    assert result.pca_eigenvalues[0] >= result.pca_eigenvalues[1]
    assert "Composite Portfolio" in result.capm_summary["Sector"].values
    assert result.portfolio_rolling_correlation.dropna().shape[0] > 0
