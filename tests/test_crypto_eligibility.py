"""Point-in-time universe eligibility (milestone 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.data.eligibility import (
    EligibilityRule,
    build_eligibility_mask,
    eligibility_summary,
    entry_dates,
    thin_universe_fraction,
)
from tf.data.synthetic import generate_synthetic_prices
from tf.engine.backtester import Backtester


def _staggered(n: int = 900, sol_offset: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "BTC": np.arange(n, dtype=float) + 100.0,
            "SOL": [np.nan] * sol_offset + list(np.arange(n - sol_offset, dtype=float) + 100.0),
        },
        index=idx,
    )


def test_rule_rejects_nonsense_configuration() -> None:
    with pytest.raises(ValueError):
        EligibilityRule(min_history=-1)
    with pytest.raises(ValueError):
        EligibilityRule(entry_lag=-5)
    with pytest.raises(ValueError):
        EligibilityRule(evaluation_frequency="fortnightly")


def test_rule_from_config_accepts_duration_strings() -> None:
    rule = EligibilityRule.from_config(
        {"min_history": "365D", "entry_lag": "30D", "evaluation_frequency": "monthly"}
    )
    assert rule.min_history == 365
    assert rule.entry_lag == 30


def test_instrument_waits_for_history_plus_entry_lag() -> None:
    """The core acceptance criterion: no trading before the rule has held."""

    prices = _staggered(sol_offset=400)
    rule = EligibilityRule(
        min_history=100, entry_lag=30, evaluation_frequency="daily"
    )
    mask = build_eligibility_mask(prices, rule)

    first_sol_observation = prices["SOL"].dropna().index[0]
    entry = entry_dates(mask)["SOL"]
    earliest_allowed = first_sol_observation + pd.Timedelta(days=100 + 30)
    assert entry is not None
    assert entry >= earliest_allowed


def test_monthly_evaluation_holds_membership_between_dates() -> None:
    prices = _staggered(sol_offset=400)
    rule = EligibilityRule(min_history=100, entry_lag=30, evaluation_frequency="monthly")
    mask = build_eligibility_mask(prices, rule)
    entry = entry_dates(mask)["SOL"]
    assert entry is not None
    assert entry.day == 1  # membership changes only on evaluation dates


def test_listing_date_blocks_pre_listing_trading() -> None:
    prices = _staggered(sol_offset=0)  # data exists from the start
    rule = EligibilityRule(min_history=10, entry_lag=0, evaluation_frequency="daily")
    listing = prices.index[500]
    mask = build_eligibility_mask(
        prices, rule, listing_dates={"SOL": listing}
    )
    assert not mask["SOL"].loc[: listing - pd.Timedelta(days=1)].any()
    assert mask["SOL"].loc[listing:].any()


def test_stale_data_makes_an_instrument_ineligible() -> None:
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    values = np.arange(400, dtype=float) + 100.0
    frame = pd.DataFrame({"BTC": values}, index=idx)
    frame.iloc[200:210, 0] = np.nan  # a ten-day outage

    rule = EligibilityRule(
        min_history=50, max_stale_days=1, entry_lag=0, evaluation_frequency="daily"
    )
    mask = build_eligibility_mask(frame, rule)
    assert not mask["BTC"].iloc[205]
    assert mask["BTC"].iloc[199]


def test_summary_and_thin_universe_reporting() -> None:
    prices = _staggered(sol_offset=400)
    rule = EligibilityRule(min_history=100, entry_lag=30, evaluation_frequency="daily")
    mask = build_eligibility_mask(prices, rule)

    summary = eligibility_summary(mask)
    assert set(summary.index) == {"BTC", "SOL"}
    assert summary.loc["BTC", "entry_date"] < summary.loc["SOL", "entry_date"]

    # For most of this window only BTC is eligible, and the report must say so.
    assert 0.0 < thin_universe_fraction(mask, minimum=2) < 1.0


def test_partial_history_keeps_the_longer_series(tmp_path) -> None:
    """A late listing must not truncate the whole backtest to its own window."""

    prices = generate_synthetic_prices(
        ["BTC", "SOL"], "2018-01-01", "2023-12-31", freq="D", seed=2,
        start_offsets={"SOL": 365 * 3},
    )
    universe = [
        {"symbol": "BTC", "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4},
        {"symbol": "SOL", "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4},
    ]
    cfg = {
        "data": {"calendar": "CRYPTO_DAILY", "allow_partial_history": True},
        "universe": {
            "eligibility": {
                "min_history": "365D",
                "entry_lag": "30D",
                "evaluation_frequency": "monthly",
            }
        },
        "backtest": {
            "start": "2018-01-01",
            "end": "2023-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12"},
        "risk": {"periods_per_year": 365, "target_portfolio_vol": 0.10},
        "execution": {"adv_limit_pct": 0.0},
    }
    result = Backtester(prices, universe, cfg).run()

    # The full BTC window survives rather than collapsing to SOL's.
    assert result.nav.index[0].year == 2018
    assert len(result.nav) > 2000
    assert np.isfinite(result.nav).all()

    sol_positions = result.positions["SOL"]
    first_sol_trade = sol_positions.index[sol_positions.abs() > 0]
    first_sol_data = prices["SOL"].dropna().index[0]
    assert len(first_sol_trade) > 0
    assert first_sol_trade[0] >= first_sol_data + pd.Timedelta(days=395)

    # Leading NaN prices must not poison the accounting identity.
    ledger = result.ledger
    assert (ledger["nav"] - ledger["cash"] - ledger["asset_value"]).abs().max() < 1e-6


def test_default_behaviour_is_unchanged_without_the_flag(tmp_path) -> None:
    """Without allow_partial_history the historical truncation still applies."""

    prices = generate_synthetic_prices(
        ["BTC", "SOL"], "2018-01-01", "2021-12-31", freq="D", seed=2,
        start_offsets={"SOL": 365 * 2},
    )
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "SOL")
    ]
    cfg = {
        "data": {"calendar": "CRYPTO_DAILY"},
        "backtest": {
            "start": "2018-01-01",
            "end": "2021-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12"},
        "risk": {"periods_per_year": 365},
        "execution": {"adv_limit_pct": 0.0},
    }
    result = Backtester(prices, universe, cfg).run()
    assert result.nav.index[0].year == 2020  # truncated to SOL's history


def test_max_asset_weight_caps_concentration() -> None:
    from tf.portfolio.sizing import _apply_max_asset_weight

    weights = pd.DataFrame(
        {"BTC": [0.9, 0.5], "ETH": [0.1, 0.5]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    capped = _apply_max_asset_weight(weights, 0.6)
    gross = weights.abs().sum(axis=1)
    assert capped.iloc[0]["BTC"] <= 0.6 * gross.iloc[0] + 1e-12
    # An already-balanced row is untouched.
    assert capped.iloc[1]["BTC"] == pytest.approx(0.5)

    assert _apply_max_asset_weight(weights, None).equals(weights)
