"""Regression tests for the sizing findings (R12-R14, R16, R19, R25)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.data.synthetic import generate_synthetic_prices
from tf.engine.backtester import Backtester
from tf.portfolio.sizing import (
    _apply_max_asset_weight,
    volatility_target_positions,
)


def _uptrend(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"BTC": 100 * np.exp(np.linspace(0, 1.5, n))}, index=idx)


def test_max_asset_weight_holds_at_the_fixed_point() -> None:
    """R14: post-cap shares must respect the cap, with gross preserved."""

    weights = pd.DataFrame({"A": [0.8], "B": [0.1], "C": [0.1]})
    capped = _apply_max_asset_weight(weights, 0.5)
    shares = capped.abs().div(capped.abs().sum(axis=1), axis=0)
    assert shares.iloc[0].max() <= 0.5 + 1e-9
    assert capped.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_max_asset_weight_leaves_infeasible_rows_alone() -> None:
    """R14: one live asset against a 50% cap must not be liquidated to zero."""

    weights = pd.DataFrame({"A": [1.0], "B": [0.0]})
    capped = _apply_max_asset_weight(weights, 0.5)
    assert capped.iloc[0, 0] == pytest.approx(1.0)


def test_risk_share_cap_no_longer_strangles_the_vol_target() -> None:
    """R12: the cap now binds the risk budget, not post-vol notional.

    With 4 equal instruments and a cap of 0.25 (exactly equal weight) the cap
    must not change sizing at all; the old notional-clip reading of the same
    number cut every instrument's weight by whatever vol happened to be.
    """

    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))) for s in "ABCD"},
        index=idx,
    )
    signals = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    kwargs = dict(
        point_values={s: 1.0 for s in "ABCD"},
        capital=1e6,
        target_portfolio_vol=0.10,
        periods_per_year=365,
        contract_rounding={s: 1e-6 for s in "ABCD"},
    )
    uncapped = volatility_target_positions(prices, signals, **kwargs)
    capped = volatility_target_positions(
        prices, signals, max_position_weight=0.25, **kwargs
    )
    pd.testing.assert_frame_equal(uncapped, capped)

    # A binding cap (0.10 against equal weight 0.25) must reduce exposure.
    tight = volatility_target_positions(
        prices, signals, max_position_weight=0.10, **kwargs
    )
    assert tight.abs().sum().sum() < uncapped.abs().sum().sum()


def test_prelisting_nans_do_not_fabricate_warm_volatility() -> None:
    """R16: a late-listing instrument must wait out the real warm-up.

    The old fillna(0.0) manufactured zero returns for the pre-listing period,
    so the EWMA was 'warm' (and near zero) on the first genuine trading day,
    producing a hugely oversized position.
    """

    idx = pd.date_range("2022-01-01", periods=300, freq="D")
    rng = np.random.default_rng(1)
    late = np.full(300, np.nan)
    late[200:] = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 100)))
    prices = pd.DataFrame({"SOL": late}, index=idx).ffill()
    signals = pd.DataFrame(1.0, index=idx, columns=["SOL"])

    positions = volatility_target_positions(
        prices,
        signals,
        {"SOL": 1.0},
        capital=1e6,
        target_portfolio_vol=0.10,
        min_vol_periods=30,
        periods_per_year=365,
        contract_rounding={"SOL": 1e-6},
    )
    first_listing = 200
    # No position for at least min_vol_periods after listing: the fabricated
    # zeros would have allowed one on day one.
    assert (positions.iloc[first_listing : first_listing + 29].abs() < 1e-12).all().all()
    assert (positions.iloc[first_listing + 40 :].abs() > 0).any().any()


def test_fraction_threshold_mode_scales_with_position_size() -> None:
    """R19: a 5% fraction threshold means 5% for BTC and for XRP alike."""

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"X": [100.0, 100.0, 100.0]}, index=idx)
    signals = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=idx)
    vol = pd.DataFrame(0.10, index=idx, columns=["X"])

    base = dict(
        point_values={"X": 1.0},
        capital=1e6,
        target_portfolio_vol=0.10,
        volatility=vol,
        contract_rounding={"X": 1e-6},
        rebalance_threshold=0.05,
    )
    contracts_mode = volatility_target_positions(
        prices, signals, rebalance_threshold_mode="contracts", **base
    )
    fraction_mode = volatility_target_positions(
        prices, signals, rebalance_threshold_mode="fraction", **base
    )
    # Positions are ~10,000 units; 0.05 contracts is noise while 5% of the
    # position is a real band. Both should hold a steady position here.
    assert contracts_mode.iloc[-1, 0] == pytest.approx(10_000.0, rel=1e-3)
    assert fraction_mode.iloc[-1, 0] == pytest.approx(10_000.0, rel=1e-3)


def test_compounding_keeps_exposure_proportional_to_equity(tmp_path) -> None:
    """R13: with compounding on, notional/NAV stays level as NAV grows."""

    prices = _uptrend()
    universe = [{"symbol": "BTC", "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-6}]

    def cfg(compound: bool) -> dict:
        return {
            "data": {"calendar": "CRYPTO_DAILY"},
            "backtest": {
                "start": "2022-01-01",
                "end": "2023-05-15",
                "starting_nav": 1_000_000.0,
                "results_dir": str(tmp_path),
                "compound_capital": compound,
            },
            "signals": {"preset": "mop2012_tsmom", "horizons": [30]},
            "risk": {
                "periods_per_year": 365,
                "target_portfolio_vol": 0.20,
                "min_vol_periods": 20,
                "rebalance_threshold": 0.0,
            },
            "execution": {"adv_limit_pct": 0.0},
        }

    on = Backtester(prices, universe, cfg(True)).run()
    off = Backtester(prices, universe, cfg(False)).run()

    def exposure_ratio(result):
        notional = (result.positions["BTC"] * prices["BTC"]).reindex(result.nav.index)
        frac = (notional / result.nav).dropna()
        return float(frac.iloc[60]), float(frac.iloc[-2])

    early_on, late_on = exposure_ratio(on)
    early_off, late_off = exposure_ratio(off)

    assert late_on == pytest.approx(early_on, rel=0.05)   # holds its leverage
    assert late_off < 0.5 * early_off                     # fixed capital decays
    assert on.nav.iloc[-1] > off.nav.iloc[-1]             # compounding compounds
