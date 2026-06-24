import os
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t


warnings.filterwarnings("ignore", category=FutureWarning)


logger = logging.getLogger(__name__)


START_DATE = "1950-01-01"
SINGLE_TICKER = "AAPL"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "PLTR"]
NUM_SIMULATIONS = 10_000
NUM_DAYS_SINGLE_TICKER = 252
NUM_DAYS_MULTIPLE_TICKERS = 22
ROLLING_WINDOW = 252
EWMA_LAMBDA = 0.94
STUDENT_T_DF = 5
VAR_LEVEL = 0.05
REPORT_PATH = Path("outputs/monte_carlo_quant_report.pdf")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


@dataclass
class SimulationResult:
    ticker: str
    current_price: float
    dynamic_mu: float
    dynamic_sigma: float
    implied_sigma: Optional[float]
    sigma_used: float
    calibration_error: float
    simulation_df: np.ndarray
    final_prices: np.ndarray
    median_final_price: float
    representative_path: np.ndarray
    var_5_return: float
    cvar_5_return: float
    lower_bound: float
    upper_bound: float


def warning_cleaner(message, category, filename, lineno, file=None, line=None):
    logger.warning("%s: %s", category.__name__, message)


warnings.showwarning = warning_cleaner


def get_stock_data(tickers, start_date):
    logger.info("Downloading price data | tickers=%s | start=%s", tickers, start_date)
    data = yf.download(tickers, start=start_date, group_by="ticker", auto_adjust=True, progress=False)
    logger.info("Downloaded price data | rows=%s | columns=%s", len(data), len(data.columns))
    return data


def get_close_series(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        for column in ((ticker, "Close"), ("Close", ticker)):
            if column in data.columns:
                return data[column].dropna()
        raise KeyError(f"Close price column not found for {ticker}.")

    if "Close" in data.columns:
        return data["Close"].dropna()

    raise KeyError(f"Close price column not found for {ticker}.")


def get_log_returns(close_prices):
    return np.log(close_prices / close_prices.shift(1)).dropna()


def kalman_filter_mean(returns, process_variance=1e-6, observation_variance=None):
    """Local-level Kalman filter for a time-varying expected return."""
    if returns.empty:
        raise ValueError("Returns series is empty.")

    observation_variance = observation_variance or returns.var()
    mean_estimate = returns.iloc[0]
    estimate_variance = observation_variance
    filtered_means = []

    for observed_return in returns:
        predicted_mean = mean_estimate
        predicted_variance = estimate_variance + process_variance
        kalman_gain = predicted_variance / (predicted_variance + observation_variance)
        mean_estimate = predicted_mean + kalman_gain * (observed_return - predicted_mean)
        estimate_variance = (1 - kalman_gain) * predicted_variance
        filtered_means.append(mean_estimate)

    return pd.Series(filtered_means, index=returns.index)


def ewma_volatility(returns, lambda_=EWMA_LAMBDA):
    variance = returns.pow(2).ewm(alpha=1 - lambda_, adjust=False).mean()
    return np.sqrt(variance)


def get_atm_implied_volatility(ticker, current_price, target_days):
    """Optional forward-looking sigma from the nearest ATM option expiry."""
    try:
        logger.info("%s | Fetching option chain for implied volatility", ticker)
        yf_ticker = yf.Ticker(ticker)
        expirations = yf_ticker.options
        if not expirations:
            logger.warning("%s | No option expirations available", ticker)
            return None

        today = pd.Timestamp.today(tz=None).normalize()
        expiration_dates = pd.to_datetime(expirations)
        target_date = today + pd.Timedelta(days=target_days)
        expiration = expirations[np.argmin(np.abs((expiration_dates - target_date).days))]
        logger.info("%s | Selected option expiration=%s", ticker, expiration)
        chain = yf_ticker.option_chain(expiration)
        option_rows = pd.concat([chain.calls, chain.puts], ignore_index=True)
        option_rows = option_rows.dropna(subset=["strike", "impliedVolatility"])

        if option_rows.empty:
            logger.warning("%s | Option chain has no usable implied volatility rows", ticker)
            return None

        option_rows["distance_to_spot"] = (option_rows["strike"] - current_price).abs()
        atm_rows = option_rows.nsmallest(4, "distance_to_spot")
        implied_volatility = atm_rows["impliedVolatility"].median()
        daily_implied_volatility = float(implied_volatility / np.sqrt(252))
        logger.info("%s | Daily implied volatility=%.6f", ticker, daily_implied_volatility)
        return daily_implied_volatility
    except Exception as exc:
        logger.warning("%s | Implied volatility unavailable: %s", ticker, exc)
        return None


def calibration_error(mu_series, sigma_series, lookback=20):
    recent_mu = mu_series.dropna().tail(lookback)
    recent_sigma = sigma_series.dropna().tail(lookback)

    if len(recent_mu) < 2 or len(recent_sigma) < 2:
        return np.nan

    mu_change = recent_mu.diff().abs().mean()
    sigma_change = recent_sigma.diff().abs().mean()
    current_sigma = max(recent_sigma.iloc[-1], 1e-12)
    return float((mu_change + sigma_change) / current_sigma)


def unit_variance_student_t(size, df=STUDENT_T_DF):
    shocks = t.rvs(df=df, size=size)
    return shocks / np.sqrt(df / (df - 2))


def simulate_dynamic_gbm(
    close_prices,
    ticker,
    num_days,
    num_simulations,
    use_implied_volatility=True,
    alpha=0.08,
    beta=0.90,
):
    logger.info(
        "%s | Starting simulation | days=%s | simulations=%s",
        ticker,
        num_days,
        num_simulations,
    )
    log_returns = get_log_returns(close_prices)
    logger.info("%s | Log returns ready | observations=%s", ticker, len(log_returns))

    mu_series = kalman_filter_mean(log_returns)
    sigma_series = ewma_volatility(log_returns)

    dynamic_mu = float(mu_series.iloc[-1])
    dynamic_sigma = float(sigma_series.iloc[-1])
    current_price = float(close_prices.iloc[-1])
    implied_sigma = None

    if use_implied_volatility:
        implied_sigma = get_atm_implied_volatility(ticker, current_price, num_days)

    sigma_used = float(np.nanmean([dynamic_sigma, implied_sigma])) if implied_sigma else dynamic_sigma
    logger.info(
        "%s | Parameters | current=%.4f | mu=%.8f | sigma_ewma=%.6f | sigma_used=%.6f",
        ticker,
        current_price,
        dynamic_mu,
        dynamic_sigma,
        sigma_used,
    )

    long_run_variance = float(sigma_series.tail(ROLLING_WINDOW).pow(2).mean())
    long_run_variance = max(long_run_variance, sigma_used**2, 1e-12)
    omega = long_run_variance * max(1 - alpha - beta, 1e-6)

    simulation_df = np.zeros((num_days, num_simulations))

    for simulation_index in range(num_simulations):
        if simulation_index and simulation_index % max(num_simulations // 4, 1) == 0:
            logger.info("%s | Simulation progress %.0f%%", ticker, simulation_index / num_simulations * 100)

        price = current_price
        variance = sigma_used**2

        for day_index in range(num_days):
            shock = unit_variance_student_t(1)[0]
            sigma_t = np.sqrt(max(variance, 1e-12))
            log_return = (dynamic_mu - 0.5 * variance) + sigma_t * shock
            price *= np.exp(log_return)
            simulation_df[day_index, simulation_index] = price

            # Volatility clustering: large simulated shocks lift future variance.
            variance = omega + alpha * (sigma_t * shock) ** 2 + beta * variance

    final_prices = simulation_df[-1, :]
    median_final_price = float(np.median(final_prices))
    representative_index = int(np.argmin(np.abs(final_prices - median_final_price)))
    representative_path = simulation_df[:, representative_index]
    final_returns = (final_prices / current_price) - 1
    var_5_return = float(np.percentile(final_returns, VAR_LEVEL * 100))
    cvar_5_return = float(final_returns[final_returns <= var_5_return].mean())
    logger.info(
        "%s | Simulation complete | median=%.4f | VaR5=%.2f%% | CVaR5=%.2f%%",
        ticker,
        median_final_price,
        var_5_return * 100,
        cvar_5_return * 100,
    )

    return SimulationResult(
        ticker=ticker,
        current_price=current_price,
        dynamic_mu=dynamic_mu,
        dynamic_sigma=dynamic_sigma,
        implied_sigma=implied_sigma,
        sigma_used=sigma_used,
        calibration_error=calibration_error(mu_series, sigma_series),
        simulation_df=simulation_df,
        final_prices=final_prices,
        median_final_price=median_final_price,
        representative_path=representative_path,
        var_5_return=var_5_return,
        cvar_5_return=cvar_5_return,
        lower_bound=float(np.percentile(final_prices, 2.5)),
        upper_bound=float(np.percentile(final_prices, 97.5)),
    )


def summarize_results(results):
    logger.info("Building consolidated summary | tickers=%s", len(results))
    rows = []
    for result in results:
        rows.append(
            {
                "Ticker": result.ticker,
                "Precio actual": result.current_price,
                "Mediana simulada": result.median_final_price,
                "Piso 95%": result.lower_bound,
                "Techo 95%": result.upper_bound,
                "Cambio % vs actual": ((result.median_final_price / result.current_price) - 1) * 100,
                "VaR 5%": result.var_5_return * 100,
                "CVaR 5%": result.cvar_5_return * 100,
                "Sigma usada diaria": result.sigma_used,
                "Error calibracion": result.calibration_error,
            }
        )

    return pd.DataFrame(rows).sort_values("Cambio % vs actual", ascending=False).reset_index(drop=True)


def add_paths_plot(ax, result):
    ax.plot(result.simulation_df, alpha=0.04, color="#4E79A7", linewidth=0.6)
    ax.plot(
        result.representative_path,
        color="#C0392B",
        linestyle="--",
        linewidth=2,
        label="Trayectoria cercana a la mediana",
    )
    ax.set_title(f"Escenarios de precio: {result.ticker}", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Precio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def add_final_distribution_plot(ax, result):
    sns.histplot(result.final_prices, bins=50, kde=True, color="#A8DADC", ax=ax)
    ax.axvline(result.current_price, color="#2E7D32", linestyle=":", label=f"Actual: {result.current_price:.2f}")
    ax.axvline(
        result.median_final_price,
        color="#C0392B",
        linestyle="--",
        label=f"Mediana: {result.median_final_price:.2f}",
    )
    ax.axvline(result.lower_bound, color="#6C757D", linestyle="-.", label=f"Piso 95%: {result.lower_bound:.2f}")
    ax.axvline(result.upper_bound, color="#6C757D", linestyle="-.", label=f"Techo 95%: {result.upper_bound:.2f}")
    ax.set_title(f"Distribucion final: {result.ticker}", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Precio final")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def add_cover_page(pdf, summary_df):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.82, "Monte Carlo GBM", fontsize=24, fontweight="bold", color="#111111")
    fig.text(0.08, 0.76, "Informe quant de escenarios y riesgo", fontsize=14, color="#333333")
    fig.text(
        0.08,
        0.68,
        (
            f"Fuente: yfinance | Inicio data: {START_DATE} | "
            f"Simulaciones: {NUM_SIMULATIONS:,} | Horizonte comparativo: {NUM_DAYS_MULTIPLE_TICKERS} dias"
        ),
        fontsize=10,
        color="#555555",
    )

    best_row = summary_df.iloc[0]
    worst_tail_row = summary_df.sort_values("CVaR 5%").iloc[0]
    most_unstable_row = summary_df.sort_values("Error calibracion", ascending=False).iloc[0]

    cards = [
        ("Mejor mediana vs actual", best_row["Ticker"], f"{best_row['Cambio % vs actual']:.2f}%"),
        ("Peor cola CVaR 5%", worst_tail_row["Ticker"], f"{worst_tail_row['CVaR 5%']:.2f}%"),
        ("Mayor error calibracion", most_unstable_row["Ticker"], f"{most_unstable_row['Error calibracion']:.4f}"),
    ]

    for index, (title, ticker, value) in enumerate(cards):
        x_position = 0.08 + index * 0.29
        fig.text(x_position, 0.55, title, fontsize=9, color="#555555")
        fig.text(x_position, 0.49, ticker, fontsize=20, fontweight="bold", color="#111111")
        fig.text(x_position, 0.44, value, fontsize=16, color="#C0392B")

    fig.text(
        0.08,
        0.30,
        "Lectura: el reporte no busca acertar un precio. Ordena escenarios, riesgo de cola y estabilidad del modelo.",
        fontsize=11,
        color="#333333",
    )
    fig.text(
        0.08,
        0.25,
        "Metricas clave: mediana simulada, rango 95%, VaR 5%, CVaR 5%, sigma usada y error de calibracion.",
        fontsize=11,
        color="#333333",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_summary_table_page(pdf, summary_df):
    display_df = summary_df.copy()
    numeric_columns = display_df.select_dtypes(include=[np.number]).columns
    display_df[numeric_columns] = display_df[numeric_columns].round(4)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Resumen consolidado", loc="left", fontsize=16, fontweight="bold", pad=18)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#111111")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F7F7F7" if row % 2 == 0 else "white")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_risk_return_page(pdf, summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    sorted_return = summary_df.sort_values("Cambio % vs actual")
    sorted_tail = summary_df.sort_values("CVaR 5%")

    axes[0].barh(sorted_return["Ticker"], sorted_return["Cambio % vs actual"], color="#4E79A7")
    axes[0].axvline(0, color="#111111", linewidth=0.8)
    axes[0].set_title("Cambio mediano vs precio actual", loc="left", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("%")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(sorted_tail["Ticker"], sorted_tail["CVaR 5%"], color="#C0392B")
    axes[1].axvline(0, color="#111111", linewidth=0.8)
    axes[1].set_title("Perdida media en el peor 5% (CVaR)", loc="left", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("%")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Decision bajo incertidumbre", x=0.08, y=0.96, ha="left", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_ticker_page(pdf, result):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [1, 1]})
    add_paths_plot(axes[0], result)
    add_final_distribution_plot(axes[1], result)

    metrics = (
        f"Actual: {result.current_price:.2f} | Mediana: {result.median_final_price:.2f} | "
        f"VaR 5%: {result.var_5_return * 100:.2f}% | CVaR 5%: {result.cvar_5_return * 100:.2f}% | "
        f"Error calibracion: {result.calibration_error:.4f}"
    )
    fig.suptitle(result.ticker, x=0.08, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.08, 0.94, metrics, fontsize=9, color="#333333")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf_report(results, summary_df, output_path=REPORT_PATH):
    logger.info("Creating PDF report | path=%s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        logger.info("PDF page | cover")
        add_cover_page(pdf, summary_df)
        logger.info("PDF page | consolidated table")
        add_summary_table_page(pdf, summary_df)
        logger.info("PDF page | risk-return")
        add_risk_return_page(pdf, summary_df)
        for result in results:
            logger.info("PDF page | ticker=%s", result.ticker)
            add_ticker_page(pdf, result)

    logger.info("PDF report complete | path=%s", output_path)
    return output_path


def run_single_ticker(ticker=SINGLE_TICKER):
    logger.info("Running single ticker workflow | ticker=%s", ticker)
    data = get_stock_data(ticker, START_DATE)
    close_prices = get_close_series(data, ticker)
    result = simulate_dynamic_gbm(close_prices, ticker, NUM_DAYS_SINGLE_TICKER, NUM_SIMULATIONS)

    print(f"--- Resultado {ticker} ---")
    print(f"Precio actual: {result.current_price:.2f}")
    print(f"Mediana simulada: {result.median_final_price:.2f}")
    print(f"VaR 5%: {result.var_5_return * 100:.2f}%")
    print(f"CVaR 5%: {result.cvar_5_return * 100:.2f}%")
    print(f"Error calibracion: {result.calibration_error:.4f}")


def run_multiple_tickers(tickers=None, output_path=REPORT_PATH):
    tickers = tickers or TICKERS
    logger.info("Running multi ticker workflow | tickers=%s", tickers)
    data = get_stock_data(tickers, START_DATE)
    results = []

    for ticker in tickers:
        logger.info("Processing ticker | %s", ticker)
        close_prices = get_close_series(data, ticker)
        result = simulate_dynamic_gbm(close_prices, ticker, NUM_DAYS_MULTIPLE_TICKERS, NUM_SIMULATIONS)
        results.append(result)

    summary_df = summarize_results(results)
    print(f"\n--- Resumen consolidado ({NUM_DAYS_MULTIPLE_TICKERS} dias, {NUM_SIMULATIONS} simulaciones) ---")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    report_path = create_pdf_report(results, summary_df, output_path=output_path)
    print(f"\nPDF generado: {report_path}")

    return summary_df


if __name__ == "__main__":
    setup_logging()
    sns.set_theme(style="darkgrid")
    run_multiple_tickers()
