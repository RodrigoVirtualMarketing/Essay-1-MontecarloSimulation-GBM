import os
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t

from option_analysis import (
    DEFAULT_OPTION_RISK_FREE_RATE,
    OptionOverlayResult,
    annualize_daily_volatility,
    build_option_overlay,
)
from portfolio_analysis import (
    DEFAULT_PORTFOLIO_BENCHMARK,
    DEFAULT_PORTFOLIO_ROLLING_WINDOW,
    DEFAULT_PORTFOLIO_SECTOR_MAP,
    PortfolioAnalyticsResult,
    build_portfolio_analytics,
)


warnings.filterwarnings("ignore", category=FutureWarning)


logger = logging.getLogger(__name__)


DATA_START_DATE = "2010-01-01"
CALIBRATION_START_DATE = "2020-01-01"
START_DATE = DATA_START_DATE
SINGLE_TICKER = "AAPL"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "SNDK","T.TO", "NVDA","SLBT"]
NUM_SIMULATIONS = 10_000
NUM_DAYS_SINGLE_TICKER = 252
NUM_DAYS_MULTIPLE_TICKERS = 252
LOOKBACK_DAYS = 756
ROLLING_WINDOW = 252
EWMA_LAMBDA = 0.94
STUDENT_T_DF = 5
VAR_LEVEL = 0.05
CALIBRATION_ERROR_THRESHOLD = 0.08
REPORT_PATH = Path("outputs/monte_carlo_quant_report.pdf")
REPORT_FIGSIZE = (11, 8.5)
REPORT_DPI = 140
REPORT_LEFT = 0.07
REPORT_RIGHT = 0.95
REPORT_TOP = 0.90
REPORT_BOTTOM = 0.08
REPORT_NAVY = "#17212B"
REPORT_BLUE = "#2F5D8C"
REPORT_RED = "#B23A3A"
REPORT_GREEN = "#2F6F4E"
REPORT_GRAY = "#6B7280"
REPORT_LIGHT_GRAY = "#F2F4F7"
REPORT_BORDER = "#D8DEE6"
REPORT_TEXT = "#20242A"
MAX_PATHS_TO_PLOT = 450
PORTFOLIO_SECTOR_MAP = DEFAULT_PORTFOLIO_SECTOR_MAP
PORTFOLIO_BENCHMARK_TICKER = DEFAULT_PORTFOLIO_BENCHMARK
PORTFOLIO_ROLLING_WINDOW = DEFAULT_PORTFOLIO_ROLLING_WINDOW
OPTION_RISK_FREE_RATE = DEFAULT_OPTION_RISK_FREE_RATE


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
    calibration_start_date: str
    calibration_observations: int
    dynamic_mu: float
    dynamic_sigma: float
    implied_sigma: Optional[float]
    sigma_used: float
    calibration_error: float
    circuit_breaker_status: str
    rankable: bool
    simulation_df: np.ndarray
    final_prices: np.ndarray
    median_final_price: float
    median_return: float
    asymmetry_score: float
    representative_path: np.ndarray
    var_5_return: float
    cvar_5_return: float
    lower_bound: float
    upper_bound: float
    option_overlay: Optional[OptionOverlayResult]


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


def get_calibration_prices(close_prices, calibration_start_date=CALIBRATION_START_DATE, lookback_days=LOOKBACK_DAYS):
    calibration_prices = close_prices.loc[close_prices.index >= pd.Timestamp(calibration_start_date)].dropna()
    if len(calibration_prices) > lookback_days + 1:
        calibration_prices = calibration_prices.tail(lookback_days + 1)

    if len(calibration_prices) < 30:
        raise ValueError("Not enough calibration observations after applying the recent-regime window.")

    return calibration_prices


def get_circuit_breaker_status(calibration_error_value):
    if np.isnan(calibration_error_value):
        return "SIN_LECTURA"
    if calibration_error_value >= CALIBRATION_ERROR_THRESHOLD:
        return "MODELO_DESCALIBRADO"
    return "CALIBRADO"


def is_rankable_status(status):
    return status == "CALIBRADO"


def calculate_asymmetry_score(median_return, cvar_return):
    denominator = abs(cvar_return)
    if denominator <= 1e-12:
        return np.nan
    return float(median_return / denominator)


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
    calibration_prices = get_calibration_prices(close_prices)
    log_returns = get_log_returns(calibration_prices)
    logger.info(
        "%s | Calibration window ready | start=%s | prices=%s | returns=%s",
        ticker,
        calibration_prices.index.min().date(),
        len(calibration_prices),
        len(log_returns),
    )

    mu_series = kalman_filter_mean(log_returns)
    sigma_series = ewma_volatility(log_returns)

    dynamic_mu = float(mu_series.iloc[-1])
    dynamic_sigma = float(sigma_series.iloc[-1])
    current_price = float(close_prices.iloc[-1])
    implied_sigma = None
    option_overlay = None

    if use_implied_volatility:
        try:
            option_overlay = build_option_overlay(
                ticker=ticker,
                current_price=current_price,
                target_days=num_days,
                model_sigma_annualized=annualize_daily_volatility(dynamic_sigma),
                risk_free_rate=OPTION_RISK_FREE_RATE,
            )
            implied_sigma = option_overlay.snapshot.daily_implied_volatility
            logger.info(
                (
                    "%s | Option overlay | expiry=%s | strike=%.2f | market=%.4f | "
                    "model=%.4f | ratio=%.4f | signal=%s"
                ),
                ticker,
                option_overlay.snapshot.expiration,
                option_overlay.snapshot.strike,
                option_overlay.snapshot.market_price,
                option_overlay.model_quote.price,
                option_overlay.market_to_model_ratio,
                option_overlay.signal.action,
            )
        except Exception as exc:
            logger.warning("%s | Option overlay unavailable: %s", ticker, exc)

    sigma_used = float(np.nanmean([dynamic_sigma, implied_sigma])) if implied_sigma else dynamic_sigma
    calibration_error_value = calibration_error(mu_series, sigma_series)
    circuit_breaker_status = get_circuit_breaker_status(calibration_error_value)
    rankable = is_rankable_status(circuit_breaker_status)
    logger.info(
        (
            "%s | Parameters | current=%.4f | mu=%.8f | sigma_ewma=%.6f | "
            "sigma_used=%.6f | calibration_error=%.4f | status=%s"
        ),
        ticker,
        current_price,
        dynamic_mu,
        dynamic_sigma,
        sigma_used,
        calibration_error_value,
        circuit_breaker_status,
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
    median_return = float((median_final_price / current_price) - 1)
    representative_index = int(np.argmin(np.abs(final_prices - median_final_price)))
    representative_path = simulation_df[:, representative_index]
    final_returns = (final_prices / current_price) - 1
    var_5_return = float(np.percentile(final_returns, VAR_LEVEL * 100))
    cvar_5_return = float(final_returns[final_returns <= var_5_return].mean())
    asymmetry_score = calculate_asymmetry_score(median_return, cvar_5_return)
    logger.info("%s | Simulation complete | median=%.4f | median_return=%.2f%% | CVaR5=%.2f%% | asymmetry=%.4f", ticker, median_final_price, median_return * 100, cvar_5_return * 100, asymmetry_score)

    return SimulationResult(
        ticker=ticker,
        current_price=current_price,
        calibration_start_date=str(calibration_prices.index.min().date()),
        calibration_observations=len(log_returns),
        dynamic_mu=dynamic_mu,
        dynamic_sigma=dynamic_sigma,
        implied_sigma=implied_sigma,
        sigma_used=sigma_used,
        calibration_error=calibration_error_value,
        circuit_breaker_status=circuit_breaker_status,
        rankable=rankable,
        simulation_df=simulation_df,
        final_prices=final_prices,
        median_final_price=median_final_price,
        median_return=median_return,
        asymmetry_score=asymmetry_score,
        representative_path=representative_path,
        var_5_return=var_5_return,
        cvar_5_return=cvar_5_return,
        lower_bound=float(np.percentile(final_prices, 2.5)),
        upper_bound=float(np.percentile(final_prices, 97.5)),
        option_overlay=option_overlay,
    )


def summarize_results(results):
    logger.info("Building consolidated summary | tickers=%s", len(results))
    rows = []
    for result in results:
        option_overlay = result.option_overlay
        rows.append(
            {
                "Ticker": result.ticker,
                "Precio actual": result.current_price,
                "Mediana simulada": result.median_final_price,
                "Piso 95%": result.lower_bound,
                "Techo 95%": result.upper_bound,
                "Cambio % vs actual": result.median_return * 100,
                "VaR 5%": result.var_5_return * 100,
                "CVaR 5%": result.cvar_5_return * 100,
                "Score asimetria": result.asymmetry_score,
                "Sigma usada diaria": result.sigma_used,
                "Error calibracion": result.calibration_error,
                "Estado modelo": result.circuit_breaker_status,
                "Rankeable": result.rankable,
                "Inicio calibracion": result.calibration_start_date,
                "Obs calibracion": result.calibration_observations,
                "Exp ATM": option_overlay.snapshot.expiration if option_overlay else "",
                "Strike ATM": option_overlay.snapshot.strike if option_overlay else np.nan,
                "Opcion mercado": option_overlay.snapshot.market_price if option_overlay else np.nan,
                "Opcion modelo": option_overlay.model_quote.price if option_overlay else np.nan,
                "Ratio opcion M/T": option_overlay.market_to_model_ratio if option_overlay else np.nan,
                "Senal opcion ATM": option_overlay.signal.action if option_overlay else "SIN_OPCION",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["Rankeable", "Score asimetria"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


def format_percent(value):
    return f"{value:.2f}%"


def format_price(value):
    return f"{value:,.2f}"


def format_model_status(status):
    status_map = {
        "CALIBRADO": "calibrado",
        "MODELO_DESCALIBRADO": "recalibrar",
        "SIN_LECTURA": "sin lectura",
    }
    return status_map.get(status, str(status).lower())


def classify_calibration_error(value):
    if np.isnan(value):
        return "sin lectura"
    if value < 0.03:
        return "estable"
    if value < CALIBRATION_ERROR_THRESHOLD:
        return "vigilar"
    return "descalibrado"


def classify_tail_risk(cvar_return):
    if cvar_return <= -0.12:
        return "riesgo alto"
    if cvar_return <= -0.06:
        return "riesgo medio"
    return "riesgo controlado"


def investor_diagnosis(row):
    if row.get("Estado modelo") == "MODELO_DESCALIBRADO":
        return "Modelo descalibrado: requiere ajuste de parametros."

    change = row["Cambio % vs actual"]
    cvar = row["CVaR 5%"]
    score = row["Score asimetria"]

    if change > 0 and cvar > -8 and score > 0:
        return "Centro sobre spot y cola manejable."
    if change > 0 and cvar <= -8:
        return "Centro positivo, pero la perdida mala pesa."
    if change <= 0 and cvar <= -8:
        return "Centro bajo spot y cola exigente."
    if change <= 0:
        return "Centro bajo spot; no es precio objetivo."
    return "Escenario mixto: revisar asimetria."


def investor_diagnosis_short(row):
    if row.get("Estado modelo") == "MODELO_DESCALIBRADO":
        return "Recalibrar"

    change = row["Cambio % vs actual"]
    cvar = row["CVaR 5%"]

    if change > 0 and cvar > -8:
        return "Centro compensa"
    if change > 0 and cvar <= -8:
        return "Cola pesa"
    if change <= 0 and cvar <= -8:
        return "Centro bajo + cola"
    if change <= 0:
        return "Centro bajo"
    return "Mixto"


def add_wrapped_text(fig, x, y, text, chars=58, fontsize=9, color=REPORT_TEXT, line_height=0.023, weight=None):
    for index, line in enumerate(textwrap.wrap(text, width=chars)):
        fig.text(x, y - index * line_height, line, fontsize=fontsize, color=color, weight=weight)


def make_report_figure():
    fig = plt.figure(figsize=REPORT_FIGSIZE, dpi=REPORT_DPI)
    fig.patch.set_facecolor("white")
    return fig


def add_report_header(fig, title, subtitle=None, section="Monte Carlo GBM"):
    fig.text(REPORT_LEFT, 0.965, section.upper(), fontsize=7.5, color=REPORT_GRAY, weight="bold")
    fig.text(REPORT_LEFT, 0.925, title, fontsize=17, color=REPORT_NAVY, weight="bold")
    if subtitle:
        fig.text(REPORT_LEFT, 0.895, subtitle, fontsize=9.5, color=REPORT_GRAY)
    fig.add_artist(
        plt.Line2D(
            [REPORT_LEFT, REPORT_RIGHT],
            [0.875, 0.875],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=0.8,
        )
    )


def add_report_footer(fig, page_label):
    footer = (
        f"{page_label} | Fuente: yfinance | Data: {DATA_START_DATE} | Calibracion: {CALIBRATION_START_DATE} | "
        f"{NUM_SIMULATIONS:,} simulaciones"
    )
    fig.add_artist(
        plt.Line2D(
            [REPORT_LEFT, REPORT_RIGHT],
            [0.045, 0.045],
            transform=fig.transFigure,
            color=REPORT_BORDER,
            linewidth=0.8,
        )
    )
    fig.text(REPORT_LEFT, 0.025, footer, fontsize=7.5, color=REPORT_GRAY)


def add_metric_card(fig, x, y, width, title, value, note="", accent=REPORT_BLUE):
    card = plt.Rectangle(
        (x, y),
        width,
        0.135,
        transform=fig.transFigure,
        facecolor=REPORT_LIGHT_GRAY,
        edgecolor=REPORT_BORDER,
        linewidth=0.8,
    )
    fig.add_artist(card)
    fig.add_artist(
        plt.Line2D(
            [x, x],
            [y, y + 0.135],
            transform=fig.transFigure,
            color=accent,
            linewidth=3,
        )
    )
    fig.text(x + 0.018, y + 0.096, title.upper(), fontsize=7.5, color=REPORT_GRAY, weight="bold")
    fig.text(x + 0.018, y + 0.052, value, fontsize=17, color=REPORT_NAVY, weight="bold")
    if note:
        fig.text(x + 0.018, y + 0.022, note, fontsize=7.8, color=REPORT_GRAY)


def add_text_block(fig, x, y, title, body, chars=52, fontsize=8.8, color=REPORT_TEXT, line_height=0.023):
    fig.text(x, y, title, fontsize=10.5, color=REPORT_NAVY, weight="bold")
    add_wrapped_text(fig, x, y - 0.035, body, chars=chars, fontsize=fontsize, color=color, line_height=line_height)


def add_paths_plot(ax, result):
    path_count = result.simulation_df.shape[1]
    if path_count > MAX_PATHS_TO_PLOT:
        sample_indexes = np.linspace(0, path_count - 1, MAX_PATHS_TO_PLOT, dtype=int)
        paths_to_plot = result.simulation_df[:, sample_indexes]
    else:
        paths_to_plot = result.simulation_df

    ax.plot(paths_to_plot, alpha=0.08, color=REPORT_BLUE, linewidth=0.45)
    ax.plot(
        result.representative_path,
        color=REPORT_RED,
        linestyle="--",
        linewidth=2,
        label="Camino central",
    )
    ax.set_title(f"Escenarios simulados: {result.ticker}", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax.set_xlabel("Dias")
    ax.set_ylabel("Precio")
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def add_final_distribution_plot(ax, result):
    sns.histplot(result.final_prices, bins=42, kde=True, color="#D9E6F2", edgecolor="white", ax=ax)
    ax.axvline(result.current_price, color=REPORT_GREEN, linestyle=":", label=f"Actual: {result.current_price:.2f}")
    ax.axvline(
        result.median_final_price,
        color=REPORT_RED,
        linestyle="--",
        label=f"Mediana: {result.median_final_price:.2f}",
    )
    ax.axvline(result.lower_bound, color=REPORT_GRAY, linestyle="-.", label=f"Piso 95%: {result.lower_bound:.2f}")
    ax.axvline(result.upper_bound, color=REPORT_GRAY, linestyle="-.", label=f"Techo 95%: {result.upper_bound:.2f}")
    ax.set_title(f"Distribucion final: {result.ticker}", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax.set_xlabel("Precio final")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def add_cover_page(pdf, summary_df):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Informe quant de decision bajo incertidumbre",
        "Lectura ejecutiva: EV, asimetria, perdida mala y estado de calibracion.",
        "Reporte ejecutivo",
    )

    best_row = summary_df[summary_df["Rankeable"]].iloc[0] if summary_df["Rankeable"].any() else summary_df.iloc[0]
    worst_tail_row = summary_df.sort_values("CVaR 5%").iloc[0]
    most_unstable_row = summary_df.sort_values("Error calibracion", ascending=False).iloc[0]
    add_metric_card(
        fig,
        0.07,
        0.67,
        0.27,
        "Mejor asimetria",
        str(best_row["Ticker"]),
        f"Score: {best_row['Score asimetria']:.3f}",
        REPORT_GREEN if best_row["Score asimetria"] >= 0 else REPORT_RED,
    )
    add_metric_card(
        fig,
        0.37,
        0.67,
        0.27,
        "Peor perdida mala",
        str(worst_tail_row["Ticker"]),
        f"CVaR 5%: {format_percent(worst_tail_row['CVaR 5%'])}",
        REPORT_RED,
    )
    add_metric_card(
        fig,
        0.67,
        0.67,
        0.27,
        "Revisar calibracion",
        str(most_unstable_row["Ticker"]),
        f"{format_model_status(most_unstable_row['Estado modelo'])} | Error: {most_unstable_row['Error calibracion']:.4f}",
        REPORT_RED if most_unstable_row["Estado modelo"] == "MODELO_DESCALIBRADO" else REPORT_BLUE,
    )
    add_metric_card(
        fig,
        0.67,
        0.50,
        0.27,
        "Horizonte",
        f"{NUM_DAYS_MULTIPLE_TICKERS} dias",
        f"Calibracion: {LOOKBACK_DAYS} obs max",
        REPORT_BLUE,
    )

    add_text_block(
        fig,
        0.07,
        0.32,
        "Diagnostico del sistema",
        (
            "El motor no entrega una prediccion. Entrega una politica de lectura: usar solo "
            "activos calibrados, exigir EV positivo y verificar que la cola no destruya la asimetria."
        ),
        chars=62,
    )
    add_text_block(
        fig,
        0.55,
        0.32,
        "Regla de lectura",
        (
            "Una accion descalibrada no entra al ranking principal. EV positivo no equivale a compra: "
            "solo habilita revision con sizing, liquidez, tesis y validacion fuera de muestra."
        ),
        chars=52,
    )
    fig.text(0.07, 0.15, "Universo analizado", fontsize=10.5, color=REPORT_NAVY, weight="bold")
    fig.text(0.07, 0.11, ", ".join(summary_df["Ticker"].astype(str)), fontsize=18, color=REPORT_TEXT, weight="bold")
    add_report_footer(fig, "Portada")
    pdf.savefig(fig)
    plt.close(fig)


def add_policy_page(pdf):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Politica de decision del reporte",
        "El output no es una prediccion; es un filtro de decision bajo incertidumbre.",
        "Gobierno del modelo",
    )

    steps = [
        ("01", "Sistema incierto", "Mercado y UFC no tienen probabilidades fijas. El modelo estima con informacion actual."),
        ("02", "EV primero", "La decision solo puede avanzar si el valor esperado es positivo."),
        ("03", "Riesgo despues", "CVaR y drawdown mandan sobre una mediana atractiva."),
        ("04", "Disyuntor", "Si el modelo esta descalibrado, se muestra pero no se rankea."),
        ("05", "Sizing", "La siguiente capa debe traducir EV y riesgo a exposicion, no a certeza."),
    ]

    y_position = 0.74
    for number, title, body in steps:
        fig.text(0.08, y_position, number, fontsize=13, color=REPORT_BLUE, weight="bold")
        fig.text(0.15, y_position, title, fontsize=11, color=REPORT_NAVY, weight="bold")
        add_wrapped_text(fig, 0.15, y_position - 0.035, body, chars=72, fontsize=9.1, color=REPORT_TEXT)
        y_position -= 0.12

    add_text_block(
        fig,
        0.58,
        0.74,
        "Formula base",
        "EV = (P_model * Pago) - ((1 - P_model) * Perdida). En acciones, P_model sale de Monte Carlo. En UFC, saldra del modelo IA.",
        chars=48,
    )
    add_text_block(
        fig,
        0.58,
        0.51,
        "Lectura final",
        "RANKEABLE no significa comprar. Significa que el activo paso el filtro cuantitativo inicial y merece revision de posicion.",
        chars=48,
    )

    add_report_footer(fig, "Politica")
    pdf.savefig(fig)
    plt.close(fig)


def add_methodology_page(pdf):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Como leer el modelo",
        "El reporte separa oportunidad, dispersion, perdida mala y estabilidad de parametros.",
        "Metodologia",
    )

    principles = [
        ("Centro", "La mediana muestra el punto medio de las simulaciones. No es precio objetivo."),
        ("Rango", "Piso 95% y techo 95% muestran donde cae la mayor parte de escenarios."),
        ("Perdida mala", "VaR y CVaR resumen que pasa en el peor 5% de escenarios."),
        ("Asimetria", "El score compara centro simulado contra perdida mala. Es mejor que mirar solo direccion."),
        ("Disyuntor", "Si el modelo esta descalibrado, no se rankea hasta ajustar parametros."),
    ]

    y_position = 0.74
    for index, (title, body) in enumerate(principles, start=1):
        fig.text(0.08, y_position, f"{index:02d}", fontsize=13, color=REPORT_BLUE, weight="bold")
        fig.text(0.14, y_position, title, fontsize=11, color=REPORT_NAVY, weight="bold")
        add_wrapped_text(fig, 0.14, y_position - 0.035, body, chars=58, fontsize=9.1, color=REPORT_TEXT)
        y_position -= 0.12

    add_text_block(
        fig,
        0.58,
        0.74,
        "Decision practica",
        (
            "Comprar, mantener o descartar no sale de una sola metrica. La decision razonable "
            "cruza centro, perdida mala, asimetria y estado del modelo."
        ),
        chars=47,
    )
    add_text_block(
        fig,
        0.58,
        0.55,
        "Uso correcto",
        (
            "El informe prioriza revision. No reemplaza tesis fundamental, sizing, liquidez, "
            "costos ni validacion fuera de muestra."
        ),
        chars=47,
    )
    add_report_footer(fig, "Metodologia")
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_table_page(pdf, summary_df):
    display_df = summary_df.copy()
    display_df["Cola"] = display_df["CVaR 5%"].apply(lambda value: classify_tail_risk(value / 100))
    display_df["Estado"] = display_df["Estado modelo"].apply(format_model_status)
    display_df["Lectura"] = display_df.apply(investor_diagnosis_short, axis=1)
    display_df = display_df[
        [
            "Ticker",
            "Precio actual",
            "Mediana simulada",
            "Cambio % vs actual",
            "CVaR 5%",
            "Score asimetria",
            "Estado",
            "Cola",
            "Lectura",
        ]
    ]
    display_df["Precio actual"] = display_df["Precio actual"].map(format_price)
    display_df["Mediana simulada"] = display_df["Mediana simulada"].map(format_price)
    display_df["Cambio % vs actual"] = display_df["Cambio % vs actual"].map(format_percent)
    display_df["CVaR 5%"] = display_df["CVaR 5%"].map(format_percent)
    display_df["Score asimetria"] = display_df["Score asimetria"].map(lambda value: f"{value:.3f}")

    fig = make_report_figure()
    add_report_header(
        fig,
        "Resumen consolidado",
        "Ranking comparable bajo la misma regla de simulacion.",
        "Resultados",
    )
    ax = fig.add_axes([0.055, 0.18, 0.89, 0.62])
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.26, 1, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    column_widths = [0.07, 0.10, 0.12, 0.12, 0.10, 0.11, 0.11, 0.13, 0.24]

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(REPORT_BORDER)
        cell.set_linewidth(0.55)
        if column < len(column_widths):
            cell.set_width(column_widths[column])
        if row == 0:
            cell.set_facecolor(REPORT_NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(REPORT_LIGHT_GRAY if row % 2 == 0 else "white")
            if column in (3, 4, 5):
                column_name = {3: "Cambio % vs actual", 4: "CVaR 5%", 5: "Score asimetria"}[column]
                raw_value = summary_df.iloc[row - 1, summary_df.columns.get_loc(column_name)]
            if column == 6 and summary_df.iloc[row - 1]["Estado modelo"] == "MODELO_DESCALIBRADO":
                cell.set_text_props(color=REPORT_RED, weight="bold")

    add_report_footer(fig, "Resumen")
    pdf.savefig(fig)
    plt.close(fig)


def add_risk_return_page(pdf, summary_df):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Asimetria del escenario",
        "El score compara centro simulado contra perdida mala. Modelos descalibrados no rankean.",
        "Diagnostico",
    )
    ax = fig.add_axes([0.10, 0.20, 0.78, 0.58])
    sorted_score = summary_df.sort_values("Score asimetria")
    colors = [
        REPORT_GRAY
        if not rankable
        else REPORT_GREEN
        if score >= 0
        else REPORT_RED
        for score, rankable in zip(sorted_score["Score asimetria"], sorted_score["Rankeable"])
    ]
    ax.barh(sorted_score["Ticker"], sorted_score["Score asimetria"], color=colors)
    ax.axvline(0, color=REPORT_NAVY, linewidth=0.8)
    ax.set_title("Score de asimetria = retorno mediano / abs(CVaR 5%)", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    add_text_block(
        fig,
        0.10,
        0.115,
        "Lectura",
        (
            "Score positivo: el centro simulado queda sobre el spot. Score negativo: el centro queda debajo. "
            "Barra gris: el modelo esta descalibrado y no debe usarse para ranking."
        ),
        chars=125,
    )

    add_report_footer(fig, "Diagnostico")
    pdf.savefig(fig)
    plt.close(fig)


def add_option_overlay_page(pdf, summary_df):
    option_df = summary_df[summary_df["Senal opcion ATM"] != "SIN_OPCION"].copy()
    if option_df.empty:
        return

    display_df = option_df[
        [
            "Ticker",
            "Exp ATM",
            "Strike ATM",
            "Opcion mercado",
            "Opcion modelo",
            "Ratio opcion M/T",
            "Senal opcion ATM",
        ]
    ]
    display_df["Strike ATM"] = display_df["Strike ATM"].map(format_price)
    display_df["Opcion mercado"] = display_df["Opcion mercado"].map(format_price)
    display_df["Opcion modelo"] = display_df["Opcion modelo"].map(format_price)
    display_df["Ratio opcion M/T"] = display_df["Ratio opcion M/T"].map(lambda value: f"{value:.2f}x")

    fig = make_report_figure()
    add_report_header(
        fig,
        "Overlay ATM de opciones",
        "Integracion de tech-engine: precio de mercado vs Black-Scholes usando sigma del modelo.",
        "Derivados",
    )
    ax = fig.add_axes([0.07, 0.24, 0.86, 0.52])
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="upper center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.05, 1, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(REPORT_BORDER)
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor(REPORT_NAVY)
            cell.set_text_props(color="white", weight="bold")
            continue

        cell.set_facecolor(REPORT_LIGHT_GRAY if row % 2 == 0 else "white")
        signal_value = option_df.iloc[row - 1]["Senal opcion ATM"]
        if column == 6 and signal_value == "BUY":
            cell.set_text_props(color=REPORT_GREEN, weight="bold")
        elif column == 6 and signal_value == "SELL":
            cell.set_text_props(color=REPORT_RED, weight="bold")

    add_text_block(
        fig,
        0.07,
        0.15,
        "Lectura",
        (
            "La senal ATM no reemplaza el motor GBM. Solo compara cuanto paga hoy el mercado "
            "de opciones frente a una prima teorica construida con la volatilidad del modelo."
        ),
        chars=120,
    )
    add_report_footer(fig, "Overlay opciones")
    pdf.savefig(fig)
    plt.close(fig)


def add_portfolio_overview_page(pdf, portfolio_result: PortfolioAnalyticsResult):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Cartera cuantitativa: diversificacion y exposicion al mercado",
        "La cartera sectorial resume el equilibrio entre riesgo idiosincratico, riesgo de industria y riesgo sistematico.",
        "Portfolio management",
    )

    sector_summary = portfolio_result.sector_summary.copy()
    if sector_summary.empty:
        raise ValueError("Portfolio sector summary is empty.")

    best_sharpe_row = sector_summary.loc[sector_summary["Sharpe"].idxmax()]
    lowest_beta_row = sector_summary.loc[sector_summary["Beta vs benchmark"].idxmin()]
    worst_drawdown_row = sector_summary.loc[sector_summary["Max drawdown %"].idxmin()]
    portfolio_total_return = float((portfolio_result.portfolio_wealth.iloc[-1] / portfolio_result.portfolio_wealth.iloc[0] - 1) * 100)
    benchmark_total_return = float((portfolio_result.normalized_frame[portfolio_result.benchmark_ticker].iloc[-1] / portfolio_result.normalized_frame[portfolio_result.benchmark_ticker].iloc[0] - 1) * 100)

    add_metric_card(
        fig,
        0.07,
        0.73,
        0.20,
        "Cartera igual peso",
        format_percent(portfolio_total_return),
        "Promedio de sectores",
        REPORT_GREEN if portfolio_total_return >= 0 else REPORT_RED,
    )
    add_metric_card(
        fig,
        0.29,
        0.73,
        0.20,
        portfolio_result.benchmark_ticker,
        format_percent(benchmark_total_return),
        "Benchmark de mercado",
        REPORT_GRAY,
    )
    add_metric_card(
        fig,
        0.51,
        0.73,
        0.20,
        "Mejor Sharpe",
        best_sharpe_row["Sector"],
        f"Sharpe: {best_sharpe_row['Sharpe']:.2f}",
        REPORT_BLUE,
    )
    add_metric_card(
        fig,
        0.73,
        0.73,
        0.20,
        "Beta mas baja",
        lowest_beta_row["Sector"],
        f"Beta: {lowest_beta_row['Beta vs benchmark']:.2f}",
        REPORT_GREEN,
    )

    ax_top = fig.add_axes([0.08, 0.34, 0.84, 0.35])
    sector_colors = {
        "Tech": REPORT_BLUE,
        "Healthcare": REPORT_GREEN,
        "Consumer Staples": REPORT_RED,
        "Composite Portfolio": "#FFFFFF",
        portfolio_result.benchmark_ticker: REPORT_GRAY,
    }
    sector_styles = {
        "Tech": "solid",
        "Healthcare": "solid",
        "Consumer Staples": "solid",
        "Composite Portfolio": "solid",
        portfolio_result.benchmark_ticker: "dashed",
    }

    for sector in portfolio_result.sector_frame.columns:
        ax_top.plot(
            portfolio_result.sector_frame.index,
            portfolio_result.sector_frame[sector],
            color=sector_colors.get(sector, REPORT_BLUE),
            linewidth=1.8,
            linestyle=sector_styles.get(sector, "solid"),
            alpha=0.85,
            label=sector,
        )

    ax_top.plot(
        portfolio_result.portfolio_wealth.index,
        portfolio_result.portfolio_wealth,
        color=REPORT_NAVY,
        linewidth=2.8,
        label="Equal-weight portfolio",
    )
    benchmark_wealth = portfolio_result.normalized_frame[portfolio_result.benchmark_ticker]
    ax_top.plot(
        benchmark_wealth.index,
        benchmark_wealth,
        color=REPORT_GRAY,
        linewidth=2.0,
        linestyle="--",
        label=portfolio_result.benchmark_ticker,
    )
    ax_top.set_title("Crecimiento normalizado de sectores y benchmark", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax_top.set_ylabel("Indice base 100")
    ax_top.grid(True, alpha=0.18)
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    ax_bottom = fig.add_axes([0.08, 0.12, 0.84, 0.16])
    portfolio_corr = portfolio_result.portfolio_rolling_correlation.dropna()
    ax_bottom.plot(portfolio_corr.index, portfolio_corr, color=REPORT_NAVY, linewidth=2.2, label="Equal-weight portfolio")
    for sector in portfolio_result.rolling_correlation.columns:
        corr_series = portfolio_result.rolling_correlation[sector].dropna()
        ax_bottom.plot(corr_series.index, corr_series, linewidth=1.5, label=sector)
    ax_bottom.axhline(0, color=REPORT_NAVY, linewidth=0.8)
    ax_bottom.set_title("Correlacion movil contra el mercado", loc="left", fontsize=10.5, fontweight="bold", color=REPORT_NAVY)
    ax_bottom.set_ylabel("Corr.")
    ax_bottom.grid(True, alpha=0.18)
    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.legend(frameon=False, ncol=4, fontsize=7.8, loc="upper left")

    add_text_block(
        fig,
        0.08,
        0.03,
        "Lectura",
        (
            "La diversificacion elimina riesgo idiosincratico e industrial, pero el riesgo sistematico permanece. "
            "Cuando el mercado entra en tension, correlaciones y betas suelen subir; esa es la parte que el portfolio manager no puede ignorar."
        ),
        chars=138,
        fontsize=8.2,
        line_height=0.020,
    )

    add_report_footer(fig, "Portfolio overview")
    pdf.savefig(fig)
    plt.close(fig)


def add_portfolio_correlation_page(pdf, portfolio_result: PortfolioAnalyticsResult):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Estructura de riesgo: correlacion y clustering",
        "La matriz de correlacion muestra si la cartera esta diversificada de verdad o solo lo parece por agregacion.",
        "Risk decomposition",
    )

    ax_heatmap = fig.add_axes([0.07, 0.26, 0.54, 0.56])
    sns.heatmap(
        portfolio_result.correlation_matrix,
        ax=ax_heatmap,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.82, "label": "Correlation"},
        linewidths=0.25,
        linecolor="white",
    )
    ax_heatmap.set_title("Correlacion entre activos del universo", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax_heatmap.tick_params(axis="x", rotation=45, labelsize=7)
    ax_heatmap.tick_params(axis="y", rotation=0, labelsize=7)

    summary_df = portfolio_result.sector_summary.copy()
    display_df = summary_df[
        [
            "Sector",
            "Total return %",
            "Annualized vol %",
            "Sharpe",
            "Max drawdown %",
            "Beta vs benchmark",
        ]
    ].copy()
    display_df["Total return %"] = display_df["Total return %"].map(format_percent)
    display_df["Annualized vol %"] = display_df["Annualized vol %"].map(format_percent)
    display_df["Sharpe"] = display_df["Sharpe"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "na")
    display_df["Max drawdown %"] = display_df["Max drawdown %"].map(format_percent)
    display_df["Beta vs benchmark"] = display_df["Beta vs benchmark"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "na")

    ax_table = fig.add_axes([0.64, 0.28, 0.29, 0.50])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.35)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(REPORT_BORDER)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(REPORT_NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(REPORT_LIGHT_GRAY if row % 2 == 0 else "white")
            if column in (3, 5):
                raw_value = summary_df.iloc[row - 1, summary_df.columns.get_loc("Sharpe" if column == 3 else "Beta vs benchmark")]
                cell.set_text_props(color=REPORT_GREEN if raw_value >= 0 else REPORT_RED)

    add_text_block(
        fig,
        0.64,
        0.17,
        "Lectura",
        (
            "Diversificar no elimina por completo el riesgo de mercado. La correlacion tiende a subir cuando el entorno se estresa, "
            "por eso la matriz de correlacion debe revisarse junto con la beta y el drawdown."
        ),
        chars=88,
        fontsize=8.3,
        line_height=0.020,
    )

    add_report_footer(fig, "Portfolio risk")
    pdf.savefig(fig)
    plt.close(fig)


def add_portfolio_factor_page(pdf, portfolio_result: PortfolioAnalyticsResult):
    fig = make_report_figure()
    add_report_header(
        fig,
        "Factores de cartera: PCA y CAPM",
        "La descomposicion espectral comprime la covarianza; CAPM traduce la exposicion sistematica a beta y alpha.",
        "Factor view",
    )

    ax_scree = fig.add_axes([0.07, 0.52, 0.38, 0.30])
    pcs = [f"PC{i + 1}" for i in range(len(portfolio_result.pca_eigenvalues))]
    ax_scree.bar(pcs, portfolio_result.pca_eigenvalues, color=REPORT_BLUE, alpha=0.85)
    ax_scree.set_title("Scree plot", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax_scree.set_ylabel("Eigenvalue")
    ax_scree.grid(axis="y", alpha=0.18)
    ax_scree.spines[["top", "right"]].set_visible(False)

    ax_cum = ax_scree.twinx()
    ax_cum.plot(pcs, portfolio_result.pca_cumulative_variance_ratio, color=REPORT_GREEN, marker="o", linewidth=2.0)
    ax_cum.set_ylim(0, 1.05)
    ax_cum.set_ylabel("Cumulative explained variance")

    ax_loadings = fig.add_axes([0.52, 0.46, 0.40, 0.36])
    loading_subset = portfolio_result.pca_loadings.iloc[:, :3]
    sns.heatmap(
        loading_subset,
        ax=ax_loadings,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.78, "label": "Loading"},
        linewidths=0.25,
        linecolor="white",
    )
    ax_loadings.set_title("Loadings sobre las 3 primeras componentes", loc="left", fontsize=11, fontweight="bold", color=REPORT_NAVY)
    ax_loadings.tick_params(axis="x", rotation=0, labelsize=7)
    ax_loadings.tick_params(axis="y", rotation=0, labelsize=7)

    capm_df = portfolio_result.capm_summary.copy()
    display_df = capm_df[["Sector", "Beta", "Alpha annualized %", "R value"]].copy()
    display_df["Beta"] = display_df["Beta"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "na")
    display_df["Alpha annualized %"] = display_df["Alpha annualized %"].map(lambda value: format_percent(value) if pd.notna(value) else "na")
    display_df["R value"] = display_df["R value"].map(lambda value: f"{value:.2f}" if pd.notna(value) else "na")

    ax_table = fig.add_axes([0.07, 0.12, 0.85, 0.20])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.3)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(REPORT_BORDER)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(REPORT_NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(REPORT_LIGHT_GRAY if row % 2 == 0 else "white")
            if column == 1 and capm_df.iloc[row - 1]["Beta"] >= 0:
                cell.set_text_props(color=REPORT_GREEN)
            if column == 2 and capm_df.iloc[row - 1]["Alpha annualized %"] >= 0:
                cell.set_text_props(color=REPORT_GREEN)

    add_text_block(
        fig,
        0.07,
        0.04,
        "Lectura",
        (
            "La primera componente suele capturar la mayor parte de la varianza común. Beta mide exposicion al mercado; "
            "alpha es el residuo. Si una estrategia depende de que el mercado suba para ganar, eso es beta, no alpha."
        ),
        chars=130,
        fontsize=8.2,
        line_height=0.020,
    )

    add_report_footer(fig, "Portfolio factors")
    pdf.savefig(fig)
    plt.close(fig)


def add_ticker_page(pdf, result):
    fig = make_report_figure()
    status_accent = REPORT_GRAY if not result.rankable else REPORT_BLUE
    add_report_header(
        fig,
        f"Ficha de accion: {result.ticker}",
        "Escenarios simulados, distribucion final y lectura de riesgo.",
        "Detalle por empresa",
    )

    add_metric_card(fig, 0.07, 0.73, 0.20, "Precio actual", format_price(result.current_price), "", REPORT_BLUE)
    add_metric_card(
        fig,
        0.29,
        0.73,
        0.20,
        "Mediana",
        format_price(result.median_final_price),
        f"{format_percent(result.median_return * 100)} vs actual",
        REPORT_GREEN if result.median_return >= 0 else REPORT_RED,
    )
    add_metric_card(
        fig,
        0.51,
        0.73,
        0.20,
        "Asimetria",
        f"{result.asymmetry_score:.3f}",
        "mediana / abs(CVaR)",
        REPORT_GREEN if result.asymmetry_score >= 0 and result.rankable else REPORT_RED if result.rankable else REPORT_GRAY,
    )
    add_metric_card(
        fig,
        0.73,
        0.73,
        0.20,
        "Estado modelo",
        format_model_status(result.circuit_breaker_status),
        f"Error: {result.calibration_error:.4f}",
        status_accent,
    )

    if not result.rankable:
        fig.text(
            0.07,
            0.685,
            "Circuit breaker activo: modelo descalibrado. No usar esta ficha como ranking hasta recalibrar parametros.",
            fontsize=8.5,
            color=REPORT_RED,
            weight="bold",
        )

    axes = [
        fig.add_axes([0.07, 0.40, 0.86, 0.25]),
        fig.add_axes([0.07, 0.10, 0.86, 0.22]),
    ]
    add_paths_plot(axes[0], result)
    add_final_distribution_plot(axes[1], result)

    add_report_footer(fig, f"Detalle {result.ticker}")
    pdf.savefig(fig)
    plt.close(fig)


def create_pdf_report(results, summary_df, output_path=REPORT_PATH, portfolio_result=None):
    logger.info("Creating PDF report | path=%s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        logger.info("PDF page | cover")
        add_cover_page(pdf, summary_df)
        logger.info("PDF page | methodology")
        add_methodology_page(pdf)
        logger.info("PDF page | consolidated table")
        add_summary_table_page(pdf, summary_df)
        logger.info("PDF page | risk-return")
        add_risk_return_page(pdf, summary_df)
        if summary_df["Senal opcion ATM"].ne("SIN_OPCION").any():
            logger.info("PDF page | options overlay")
            add_option_overlay_page(pdf, summary_df)
        if portfolio_result is not None:
            logger.info("PDF page | portfolio overview")
            add_portfolio_overview_page(pdf, portfolio_result)
            logger.info("PDF page | portfolio correlation")
            add_portfolio_correlation_page(pdf, portfolio_result)
            logger.info("PDF page | portfolio factors")
            add_portfolio_factor_page(pdf, portfolio_result)
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
    print(f"Inicio calibracion: {result.calibration_start_date}")
    print(f"Observaciones calibracion: {result.calibration_observations}")
    print(f"Precio actual: {result.current_price:.2f}")
    print(f"Mediana simulada: {result.median_final_price:.2f}")
    print(f"Cambio mediano vs actual: {result.median_return * 100:.2f}%")
    print(f"VaR 5%: {result.var_5_return * 100:.2f}%")
    print(f"CVaR 5%: {result.cvar_5_return * 100:.2f}%")
    print(f"Score asimetria: {result.asymmetry_score:.4f}")
    print(f"Error calibracion: {result.calibration_error:.4f}")
    print(f"Estado modelo: {result.circuit_breaker_status}")


def run_multiple_tickers(tickers=None, output_path=REPORT_PATH):
    tickers = tickers or TICKERS
    logger.info("Running multi ticker workflow | tickers=%s", tickers)
    portfolio_universe = list(dict.fromkeys([ticker for values in PORTFOLIO_SECTOR_MAP.values() for ticker in values] + [PORTFOLIO_BENCHMARK_TICKER]))
    download_universe = list(dict.fromkeys(list(tickers) + portfolio_universe))
    data = get_stock_data(download_universe, START_DATE)
    results = []

    for ticker in tickers:
        logger.info("Processing ticker | %s", ticker)
        close_prices = get_close_series(data, ticker)
        result = simulate_dynamic_gbm(close_prices, ticker, NUM_DAYS_MULTIPLE_TICKERS, NUM_SIMULATIONS)
        results.append(result)

    portfolio_result = None
    try:
        portfolio_result = build_portfolio_analytics(
            data,
            sector_map=PORTFOLIO_SECTOR_MAP,
            benchmark_ticker=PORTFOLIO_BENCHMARK_TICKER,
            rolling_window=PORTFOLIO_ROLLING_WINDOW,
        )
    except Exception as exc:
        logger.warning("Portfolio analysis unavailable: %s", exc)

    summary_df = summarize_results(results)
    print(f"\n--- Resumen consolidado ({NUM_DAYS_MULTIPLE_TICKERS} dias, {NUM_SIMULATIONS} simulaciones) ---")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))

    report_path = create_pdf_report(results, summary_df, output_path=output_path, portfolio_result=portfolio_result)
    print(f"\nPDF generado: {report_path}")

    return summary_df


if __name__ == "__main__":
    setup_logging()
    sns.set_theme(style="darkgrid")
    run_multiple_tickers()
