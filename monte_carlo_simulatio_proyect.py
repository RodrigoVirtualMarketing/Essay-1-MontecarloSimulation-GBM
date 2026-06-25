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


warnings.filterwarnings("ignore", category=FutureWarning)


logger = logging.getLogger(__name__)


DATA_START_DATE = "2010-01-01"
CALIBRATION_START_DATE = "2020-01-01"
START_DATE = DATA_START_DATE
SINGLE_TICKER = "AAPL"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "PLTR"]
NUM_SIMULATIONS = 10_000
NUM_DAYS_SINGLE_TICKER = 252
NUM_DAYS_MULTIPLE_TICKERS = 22
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

    if use_implied_volatility:
        implied_sigma = get_atm_implied_volatility(ticker, current_price, num_days)

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
    logger.info(
        "%s | Simulation complete | median=%.4f | median_return=%.2f%% | CVaR5=%.2f%% | asymmetry=%.4f",
        ticker,
        median_final_price,
        median_return * 100,
        cvar_5_return * 100,
        asymmetry_score,
    )

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


def add_text_block(fig, x, y, title, body, chars=52):
    fig.text(x, y, title, fontsize=10.5, color=REPORT_NAVY, weight="bold")
    add_wrapped_text(fig, x, y - 0.035, body, chars=chars, fontsize=8.8, color=REPORT_TEXT, line_height=0.023)


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
        "Informe quant de escenarios y riesgo",
        "Lectura para inversionista: centro probable, rango, perdida mala y estabilidad del modelo.",
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

    add_text_block(
        fig,
        0.07,
        0.49,
        "Diagnostico del sistema",
        (
            "El motor no entrega una prediccion. Entrega una distribucion de escenarios. "
            "Para invertir, la lectura practica es evaluar si el centro compensa la perdida "
            "mala y si el modelo esta calibrado."
        ),
        chars=62,
    )
    add_text_block(
        fig,
        0.55,
        0.49,
        "Regla de lectura",
        (
            "Una accion descalibrada no entra al ranking principal. Una mediana debajo del "
            "spot no significa empresa debil; significa centro simulado bajo el precio actual."
        ),
        chars=52,
    )
    fig.text(0.07, 0.26, "Universo analizado", fontsize=10.5, color=REPORT_NAVY, weight="bold")
    fig.text(0.07, 0.22, ", ".join(summary_df["Ticker"].astype(str)), fontsize=18, color=REPORT_TEXT, weight="bold")
    add_report_footer(fig, "Portada")
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
                cell.set_text_props(color=REPORT_GREEN if raw_value >= 0 else REPORT_RED)
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


def add_ticker_page(pdf, result):
    fig = make_report_figure()
    change = result.median_return * 100
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
        f"{format_percent(change)} vs actual",
        REPORT_GREEN if change >= 0 else REPORT_RED,
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


def create_pdf_report(results, summary_df, output_path=REPORT_PATH):
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
