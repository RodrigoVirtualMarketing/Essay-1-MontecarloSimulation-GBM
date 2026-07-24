import math
from datetime import datetime

import pandas as pd

from option_analysis import (
    OptionMarketSnapshot,
    analyze_option_market,
    calculate_black_scholes_quote,
    generate_option_signal,
    select_atm_option_row,
    select_target_expiration,
)


def test_calculate_black_scholes_quote_matches_reference_call_price():
    quote = calculate_black_scholes_quote(
        spot_price=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        sigma_annualized=0.2,
        risk_free_rate=0.05,
    )

    assert math.isclose(quote.price, 10.4506, rel_tol=1e-4)
    assert math.isclose(quote.delta, 0.6368, rel_tol=1e-4)


def test_generate_option_signal_classifies_market_gap():
    assert generate_option_signal(20.0, 10.0).action == "SELL"
    assert generate_option_signal(6.0, 10.0).action == "BUY"
    assert generate_option_signal(9.0, 10.0).action == "HOLD"


def test_select_target_expiration_chooses_nearest_target_horizon():
    expirations = ["2026-08-21", "2026-10-16", "2027-01-15"]

    selected = select_target_expiration(
        expirations,
        target_days=95,
        as_of=datetime(2026, 7, 7),
    )

    assert selected == "2026-10-16"


def test_select_atm_option_row_prefers_nearest_strike_with_usable_price():
    option_frame = pd.DataFrame(
        [
            {"strike": 95.0, "bid": 7.0, "ask": 7.4, "lastPrice": 7.2, "impliedVolatility": 0.31},
            {"strike": 100.0, "bid": 4.9, "ask": 5.1, "lastPrice": 5.0, "impliedVolatility": 0.28},
            {"strike": 105.0, "bid": 2.9, "ask": 3.1, "lastPrice": 3.0, "impliedVolatility": 0.27},
        ]
    )

    row = select_atm_option_row(option_frame, spot_price=101.0)

    assert row["strike"] == 100.0
    assert math.isclose(row["market_price"], 5.0, rel_tol=1e-9)


def test_analyze_option_market_builds_overlay_result():
    snapshot = OptionMarketSnapshot(
        ticker="AAPL",
        option_type="call",
        expiration="2026-10-16",
        days_to_expiry=101,
        strike=100.0,
        market_price=9.5,
        implied_volatility=0.26,
    )

    result = analyze_option_market(
        snapshot=snapshot,
        current_price=102.0,
        model_sigma_annualized=0.22,
        risk_free_rate=0.03,
    )

    assert result.model_quote.price > 0
    assert result.market_to_model_ratio > 0
    assert result.signal.action in {"BUY", "HOLD", "SELL"}
