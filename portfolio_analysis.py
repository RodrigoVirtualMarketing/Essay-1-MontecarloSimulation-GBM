"""Portfolio analytics inspired by the Quant Guild portfolio-management lecture."""

from dataclasses import dataclass
import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


DEFAULT_PORTFOLIO_SECTOR_MAP = {
    "Tech": ["AAPL", "MSFT", "AVGO"],
    "Healthcare": ["UNH", "JNJ", "AMGN"],
    "Consumer Staples": ["WMT", "COST", "PG"],
}
DEFAULT_PORTFOLIO_BENCHMARK = "SPY"
DEFAULT_PORTFOLIO_ROLLING_WINDOW = 63


@dataclass
class PortfolioAnalyticsResult:
    sector_map: Dict[str, list[str]]
    benchmark_ticker: str
    close_frame: pd.DataFrame
    return_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    sector_frame: pd.DataFrame
    portfolio_wealth: pd.Series
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    rolling_correlation: pd.DataFrame
    portfolio_rolling_correlation: pd.Series
    correlation_matrix: pd.DataFrame
    pca_eigenvalues: np.ndarray
    pca_explained_variance_ratio: np.ndarray
    pca_cumulative_variance_ratio: np.ndarray
    pca_loadings: pd.DataFrame
    capm_summary: pd.DataFrame
    sector_summary: pd.DataFrame


def get_close_series(data, ticker):
    """Extract a close series for `ticker` from flat or MultiIndex yfinance output."""
    if isinstance(data.columns, pd.MultiIndex):
        for column in ((ticker, "Close"), ("Close", ticker)):
            if column in data.columns:
                return data[column].dropna()
        raise KeyError(f"Close price column not found for {ticker}.")

    if "Close" in data.columns:
        return data["Close"].dropna()

    if ticker in data.columns:
        return data[ticker].dropna()

    raise KeyError(f"Close price column not found for {ticker}.")


def build_close_frame(data, tickers):
    series_map = {}
    missing = []

    for ticker in tickers:
        try:
            series_map[ticker] = get_close_series(data, ticker)
        except KeyError:
            missing.append(ticker)

    if missing:
        logger.warning("Missing close series for tickers: %s", ", ".join(missing))

    if not series_map:
        raise ValueError("No close series available for portfolio analysis.")

    close_frame = pd.concat(series_map, axis=1).sort_index()
    close_frame = close_frame.ffill().dropna()

    if close_frame.empty:
        raise ValueError("Close frame is empty after cleaning.")

    return close_frame


def normalize_frame(close_frame):
    return (close_frame / close_frame.iloc[0]) * 100


def max_drawdown(wealth_series):
    running_max = wealth_series.cummax()
    drawdown = wealth_series / running_max - 1
    return float(drawdown.min())


def annualized_sharpe(returns, risk_free_rate=0.0):
    if returns.empty:
        return np.nan

    excess = returns - (risk_free_rate / 252.0)
    volatility = excess.std(ddof=1)
    if volatility <= 1e-12:
        return np.nan
    return float(excess.mean() / volatility * np.sqrt(252))


def annualized_alpha(daily_alpha):
    if pd.isna(daily_alpha):
        return np.nan
    return float((1 + daily_alpha) ** 252 - 1)


def compute_pca_summary(return_frame):
    corr_matrix = return_frame.corr().fillna(0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = float(np.sum(eigenvalues))
    explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    loadings = pd.DataFrame(
        eigenvectors,
        index=corr_matrix.index,
        columns=[f"PC{i+1}" for i in range(len(eigenvalues))],
    )

    return corr_matrix, eigenvalues, explained_variance_ratio, cumulative_variance_ratio, loadings


def compute_capm_summary(portfolio_returns, benchmark_returns):
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "benchmark"]

    if len(aligned) < 3:
        return {
            "beta": np.nan,
            "alpha_daily": np.nan,
            "alpha_annualized": np.nan,
            "r_value": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(aligned["benchmark"], aligned["portfolio"])
    return {
        "beta": float(slope),
        "alpha_daily": float(intercept),
        "alpha_annualized": annualized_alpha(float(intercept)),
        "r_value": float(r_value),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


def compute_sector_summary(sector_frame, benchmark_returns):
    summary_rows = []
    sector_returns = sector_frame.pct_change().dropna()
    aligned_benchmark = benchmark_returns.reindex(sector_returns.index).dropna()

    for sector in sector_frame.columns:
        wealth = sector_frame[sector].dropna()
        returns = sector_returns[sector].reindex(aligned_benchmark.index).dropna()
        benchmark = aligned_benchmark.reindex(returns.index).dropna()
        common_index = returns.index.intersection(benchmark.index)
        returns = returns.reindex(common_index)
        benchmark = benchmark.reindex(common_index)

        capm = compute_capm_summary(returns, benchmark)
        summary_rows.append(
            {
                "Sector": sector,
                "Final value": float(wealth.iloc[-1]),
                "Total return %": float((wealth.iloc[-1] / wealth.iloc[0] - 1) * 100),
                "Annualized vol %": float(returns.std(ddof=1) * np.sqrt(252) * 100) if len(returns) > 1 else np.nan,
                "Sharpe": annualized_sharpe(returns),
                "Max drawdown %": max_drawdown(wealth) * 100,
                "Beta vs benchmark": capm["beta"],
                "Alpha annualized %": capm["alpha_annualized"] * 100 if not pd.isna(capm["alpha_annualized"]) else np.nan,
                "R value": capm["r_value"],
            }
        )

    return pd.DataFrame(summary_rows)


def build_portfolio_analytics(data, sector_map=None, benchmark_ticker=DEFAULT_PORTFOLIO_BENCHMARK, rolling_window=DEFAULT_PORTFOLIO_ROLLING_WINDOW):
    sector_map = sector_map or DEFAULT_PORTFOLIO_SECTOR_MAP
    all_tickers = list(dict.fromkeys([ticker for tickers in sector_map.values() for ticker in tickers] + [benchmark_ticker]))

    close_frame = build_close_frame(data, all_tickers)
    normalized_frame = normalize_frame(close_frame)
    return_frame = close_frame.pct_change().dropna()

    available_sector_map = {}
    for sector, tickers in sector_map.items():
        available = [ticker for ticker in tickers if ticker in normalized_frame.columns]
        if available:
            available_sector_map[sector] = available
        else:
            logger.warning("Skipping sector without data: %s", sector)

    if benchmark_ticker not in normalized_frame.columns:
        raise KeyError(f"Benchmark ticker {benchmark_ticker} is missing from the price frame.")

    sector_frame = pd.DataFrame(index=normalized_frame.index)
    for sector, tickers in available_sector_map.items():
        sector_frame[sector] = normalized_frame[tickers].mean(axis=1)

    if sector_frame.empty:
        raise ValueError("No sector composites could be built.")

    portfolio_wealth = sector_frame.mean(axis=1)
    portfolio_returns = portfolio_wealth.pct_change().dropna()
    benchmark_returns = return_frame[benchmark_ticker].dropna()
    aligned_portfolio_benchmark = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned_portfolio_benchmark.columns = ["portfolio", "benchmark"]

    correlation_matrix, eigenvalues, explained_variance_ratio, cumulative_variance_ratio, loadings = compute_pca_summary(
        return_frame[[ticker for ticker in return_frame.columns if ticker != benchmark_ticker]]
    )

    rolling_correlation = pd.DataFrame(index=sector_frame.index)
    for sector in sector_frame.columns:
        rolling_correlation[sector] = sector_frame[sector].pct_change().rolling(window=rolling_window).corr(benchmark_returns)

    portfolio_rolling_correlation = aligned_portfolio_benchmark["portfolio"].rolling(window=rolling_window).corr(
        aligned_portfolio_benchmark["benchmark"]
    )

    capm_rows = []
    for sector in sector_frame.columns:
        sector_returns = sector_frame[sector].pct_change().dropna()
        capm = compute_capm_summary(sector_returns, benchmark_returns)
        capm_rows.append(
            {
                "Sector": sector,
                "Beta": capm["beta"],
                "Alpha daily %": capm["alpha_daily"] * 100 if not pd.isna(capm["alpha_daily"]) else np.nan,
                "Alpha annualized %": capm["alpha_annualized"] * 100 if not pd.isna(capm["alpha_annualized"]) else np.nan,
                "R value": capm["r_value"],
                "P value": capm["p_value"],
                "Std err": capm["std_err"],
            }
        )

    portfolio_capm = compute_capm_summary(portfolio_returns, benchmark_returns.reindex(portfolio_returns.index))
    capm_rows.append(
        {
            "Sector": "Composite Portfolio",
            "Beta": portfolio_capm["beta"],
            "Alpha daily %": portfolio_capm["alpha_daily"] * 100 if not pd.isna(portfolio_capm["alpha_daily"]) else np.nan,
            "Alpha annualized %": portfolio_capm["alpha_annualized"] * 100 if not pd.isna(portfolio_capm["alpha_annualized"]) else np.nan,
            "R value": portfolio_capm["r_value"],
            "P value": portfolio_capm["p_value"],
            "Std err": portfolio_capm["std_err"],
        }
    )

    capm_summary = pd.DataFrame(capm_rows)
    sector_summary = compute_sector_summary(sector_frame, benchmark_returns)

    return PortfolioAnalyticsResult(
        sector_map=available_sector_map,
        benchmark_ticker=benchmark_ticker,
        close_frame=close_frame,
        return_frame=return_frame,
        normalized_frame=normalized_frame,
        sector_frame=sector_frame,
        portfolio_wealth=portfolio_wealth,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        rolling_correlation=rolling_correlation,
        portfolio_rolling_correlation=portfolio_rolling_correlation,
        correlation_matrix=correlation_matrix,
        pca_eigenvalues=eigenvalues,
        pca_explained_variance_ratio=explained_variance_ratio,
        pca_cumulative_variance_ratio=cumulative_variance_ratio,
        pca_loadings=loadings,
        capm_summary=capm_summary,
        sector_summary=sector_summary,
    )
