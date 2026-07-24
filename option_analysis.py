"""ATM option overlay extracted from the external tech-engine repo."""

from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


logger = logging.getLogger(__name__)


ANNUAL_TRADING_DAYS = 252
DEFAULT_OPTION_RISK_FREE_RATE = 0.053
DEFAULT_OPTION_BUY_THRESHOLD = 0.80
DEFAULT_OPTION_SELL_THRESHOLD = 1.50
MIN_TIME_TO_EXPIRY = 1e-5
MIN_VOLATILITY = 1e-6


@dataclass
class OptionMarketSnapshot:
    ticker: str
    option_type: str
    expiration: str
    days_to_expiry: int
    strike: float
    market_price: float
    implied_volatility: float

    @property
    def daily_implied_volatility(self) -> float:
        return float(self.implied_volatility / math.sqrt(ANNUAL_TRADING_DAYS))


@dataclass
class BlackScholesQuote:
    price: float
    d1: float
    d2: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class OptionSignal:
    action: str
    confidence: float
    price_ratio: float
    reason: str


@dataclass
class OptionOverlayResult:
    snapshot: OptionMarketSnapshot
    model_quote: BlackScholesQuote
    model_sigma_annualized: float
    signal: OptionSignal

    @property
    def market_to_model_ratio(self) -> float:
        return self.signal.price_ratio


def annualize_daily_volatility(daily_sigma: float) -> float:
    return float(max(daily_sigma, 0.0) * math.sqrt(ANNUAL_TRADING_DAYS))


def calculate_time_to_expiry(expiration: str, as_of: Optional[datetime] = None) -> float:
    expiry = datetime.strptime(expiration, "%Y-%m-%d")
    reference = as_of or datetime.utcnow()
    seconds_to_expiry = max((expiry - reference).total_seconds(), 0.0)
    return max(seconds_to_expiry / (365.0 * 24 * 60 * 60), MIN_TIME_TO_EXPIRY)


def resolve_option_market_price(option_row: pd.Series) -> float:
    bid = option_row.get("bid")
    ask = option_row.get("ask")
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)

    last_price = option_row.get("lastPrice")
    if pd.notna(last_price) and last_price > 0:
        return float(last_price)

    return float("nan")


def select_target_expiration(expirations: list[str], target_days: int, as_of: Optional[datetime] = None) -> str:
    if not expirations:
        raise ValueError("No option expirations available.")

    reference = pd.Timestamp(as_of or datetime.utcnow()).normalize()
    target_date = reference + pd.Timedelta(days=target_days)
    expiration_index = pd.to_datetime(expirations)
    closest_index = int(np.argmin(np.abs((expiration_index - target_date).days)))
    return expirations[closest_index]


def select_atm_option_row(option_frame: pd.DataFrame, spot_price: float) -> pd.Series:
    if option_frame.empty:
        raise ValueError("Option chain is empty.")

    rows = option_frame.dropna(subset=["strike", "impliedVolatility"]).copy()
    if rows.empty:
        raise ValueError("Option chain has no valid strike and implied volatility rows.")

    rows["market_price"] = rows.apply(resolve_option_market_price, axis=1)
    rows = rows.dropna(subset=["market_price"])
    if rows.empty:
        raise ValueError("Option chain has no usable market prices.")

    rows["distance_to_spot"] = (rows["strike"] - spot_price).abs()
    return rows.nsmallest(1, ["distance_to_spot", "market_price"]).iloc[0]


def calculate_black_scholes_quote(
    spot_price: float,
    strike: float,
    time_to_expiry: float,
    sigma_annualized: float,
    risk_free_rate: float,
    option_type: str = "call",
) -> BlackScholesQuote:
    if spot_price <= 0:
        raise ValueError("Spot price must be positive.")
    if strike <= 0:
        raise ValueError("Strike must be positive.")

    time_to_expiry = max(time_to_expiry, MIN_TIME_TO_EXPIRY)
    sigma_annualized = max(sigma_annualized, MIN_VOLATILITY)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot_price / strike)
        + (risk_free_rate + 0.5 * sigma_annualized**2) * time_to_expiry
    ) / (sigma_annualized * sqrt_t)
    d2 = d1 - sigma_annualized * sqrt_t

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    nmd1 = norm.cdf(-d1)
    nmd2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        price = spot_price * nd1 - discounted_strike * nd2
        delta = nd1
        theta_numerator = (
            -spot_price * pdf_d1 * sigma_annualized / (2 * sqrt_t)
            - risk_free_rate * discounted_strike * nd2
        )
    elif option_type == "put":
        price = discounted_strike * nmd2 - spot_price * nmd1
        delta = nd1 - 1
        theta_numerator = (
            -spot_price * pdf_d1 * sigma_annualized / (2 * sqrt_t)
            + risk_free_rate * discounted_strike * nmd2
        )
    else:
        raise ValueError(f"Unsupported option type: {option_type}")

    gamma = pdf_d1 / (spot_price * sigma_annualized * sqrt_t)
    theta = theta_numerator / 365.0
    vega = spot_price * pdf_d1 * sqrt_t / 100.0

    return BlackScholesQuote(
        price=float(price),
        d1=float(d1),
        d2=float(d2),
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta),
        vega=float(vega),
    )


def generate_option_signal(
    market_price: float,
    theoretical_price: float,
    buy_threshold: float = DEFAULT_OPTION_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_OPTION_SELL_THRESHOLD,
) -> OptionSignal:
    if theoretical_price <= 0:
        raise ValueError("Theoretical option price must be positive.")

    ratio = market_price / theoretical_price

    if ratio > sell_threshold:
        action = "SELL"
        confidence = min((ratio - 1.0) / max(sell_threshold - 1.0, 1e-9), 1.0)
        reason = "Market option price is above the model price."
    elif ratio < buy_threshold:
        action = "BUY"
        confidence = min((1.0 - ratio) / max(1.0 - buy_threshold, 1e-9), 1.0)
        reason = "Market option price is below the model price."
    else:
        action = "HOLD"
        confidence = 0.5
        reason = "Market option price is close to the model price."

    return OptionSignal(
        action=action,
        confidence=float(confidence),
        price_ratio=float(ratio),
        reason=reason,
    )


def analyze_option_market(
    snapshot: OptionMarketSnapshot,
    current_price: float,
    model_sigma_annualized: float,
    risk_free_rate: float = DEFAULT_OPTION_RISK_FREE_RATE,
) -> OptionOverlayResult:
    quote = calculate_black_scholes_quote(
        spot_price=current_price,
        strike=snapshot.strike,
        time_to_expiry=calculate_time_to_expiry(snapshot.expiration),
        sigma_annualized=model_sigma_annualized,
        risk_free_rate=risk_free_rate,
        option_type=snapshot.option_type,
    )
    signal = generate_option_signal(snapshot.market_price, quote.price)
    return OptionOverlayResult(
        snapshot=snapshot,
        model_quote=quote,
        model_sigma_annualized=float(model_sigma_annualized),
        signal=signal,
    )


def build_option_overlay(
    ticker: str,
    current_price: float,
    target_days: int,
    model_sigma_annualized: float,
    risk_free_rate: float = DEFAULT_OPTION_RISK_FREE_RATE,
    option_type: str = "call",
) -> OptionOverlayResult:
    logger.info("%s | Fetching ATM %s option overlay", ticker, option_type)
    yf_ticker = yf.Ticker(ticker)
    expiration = select_target_expiration(list(yf_ticker.options), target_days)
    chain = yf_ticker.option_chain(expiration)
    option_frame = chain.calls if option_type == "call" else chain.puts
    atm_row = select_atm_option_row(option_frame, current_price)

    snapshot = OptionMarketSnapshot(
        ticker=ticker,
        option_type=option_type,
        expiration=expiration,
        days_to_expiry=max((datetime.strptime(expiration, "%Y-%m-%d") - datetime.utcnow()).days, 0),
        strike=float(atm_row["strike"]),
        market_price=float(atm_row["market_price"]),
        implied_volatility=float(atm_row["impliedVolatility"]),
    )
    return analyze_option_market(
        snapshot=snapshot,
        current_price=current_price,
        model_sigma_annualized=model_sigma_annualized,
        risk_free_rate=risk_free_rate,
    )
