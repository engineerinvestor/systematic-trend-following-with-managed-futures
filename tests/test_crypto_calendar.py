"""Calendar frequency and 365-day annualization (CRYPTO_SPEC.md milestone 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.data.calendar import TradingCalendar
from tf.data.synthetic import generate_synthetic_prices
from tf.eval.metrics import CALENDAR_DAYS, performance_summary
from tf.portfolio.sizing import _compute_volatility
from tf.risk.vol import com_to_lambda, ewma_vol, rolling_volatility


def test_daily_calendar_keeps_weekends() -> None:
    cal = TradingCalendar(freq="daily")
    sessions = cal.sessions("2024-01-01", "2024-01-14")
    assert len(sessions) == 14
    assert sessions.dayofweek.max() == 6


def test_weekday_calendar_unchanged() -> None:
    cal = TradingCalendar()
    sessions = cal.sessions("2024-01-01", "2024-01-14")
    assert len(sessions) == 10
    assert sessions.dayofweek.max() == 4


def test_from_name_maps_crypto_and_defaults_safely() -> None:
    assert TradingCalendar.from_name("CRYPTO_DAILY").freq == "daily"
    assert TradingCalendar.from_name("GENERIC").freq == "weekday"
    assert TradingCalendar.from_name(None).freq == "weekday"
    # Unknown names must degrade to the historical default, not explode.
    assert TradingCalendar.from_name("no-such-calendar").freq == "weekday"


def test_invalid_frequency_rejected() -> None:
    with pytest.raises(ValueError):
        TradingCalendar(freq="hourly")


def test_daily_align_preserves_weekend_observations() -> None:
    cal = TradingCalendar(freq="daily")
    idx = pd.date_range("2024-01-01", "2024-01-14", freq="D")
    frame = pd.DataFrame({"BTC": np.arange(len(idx), dtype=float) + 100.0}, index=idx)
    aligned = cal.align(frame, "2024-01-01", "2024-01-14")
    pd.testing.assert_frame_equal(aligned, frame)


def test_com_to_lambda_roundtrip() -> None:
    lam = com_to_lambda(60)
    assert lam == pytest.approx(60 / 61)
    # Center of mass is lam / (1 - lam).
    assert lam / (1 - lam) == pytest.approx(60.0)


def test_annualisation_factor_is_exactly_sqrt_365() -> None:
    """The annualisation factor itself must be exact, free of estimator noise."""

    idx = pd.date_range("2015-01-01", periods=1500, freq="D")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame({"BTC": rng.normal(0.0, 0.03, size=len(idx))}, index=idx)

    raw = ewma_vol(returns, lam=0.97, min_periods=20, annualize=False)
    annualised = ewma_vol(
        returns, lam=0.97, min_periods=20, periods_per_year=CALENDAR_DAYS
    )
    ratio = (annualised["BTC"] / raw["BTC"]).dropna()
    ratio = ratio[np.isfinite(ratio)]
    assert (ratio - np.sqrt(CALENDAR_DAYS)).abs().max() < 1e-9


def test_annualised_vol_recovers_known_daily_sigma() -> None:
    """A known daily vol on a 7-day series must annualize to sigma * sqrt(365).

    Tolerance is 3 percent: the full-sample standard deviation of 4,000 draws
    carries a relative standard error near 1.1 percent, so a tighter bound would
    be testing the random seed rather than the annualisation.
    """

    sigma = 0.03
    idx = pd.date_range("2015-01-01", periods=4000, freq="D")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame({"BTC": rng.normal(0.0, sigma, size=len(idx))}, index=idx)

    vol = rolling_volatility(
        returns, window=len(idx), min_periods=len(idx), periods_per_year=CALENDAR_DAYS
    )
    realised = float(vol["BTC"].iloc[-1])
    expected = sigma * np.sqrt(CALENDAR_DAYS)
    assert realised == pytest.approx(expected, rel=0.03)

    # Using the weekday factor on the same series understates vol materially.
    weekday = rolling_volatility(
        returns, window=len(idx), min_periods=len(idx), periods_per_year=252
    )
    assert float(weekday["BTC"].iloc[-1]) < 0.85 * expected


def test_performance_summary_annualisation_is_parameterised() -> None:
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    rng = np.random.default_rng(3)
    nav = pd.Series(1_000_000 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx)))), index=idx)

    weekday = performance_summary(nav)
    calendar = performance_summary(nav, periods_per_year=CALENDAR_DAYS)

    ratio = calendar["Volatility"] / weekday["Volatility"]
    assert ratio == pytest.approx(np.sqrt(365 / 252), rel=1e-6)
    # The default must stay 252 so existing futures results are untouched.
    assert weekday["Volatility"] == pytest.approx(
        performance_summary(nav, periods_per_year=252)["Volatility"]
    )


def test_vol_warmup_mask_removes_backfill_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(11)
    prices = pd.DataFrame(
        {"BTC": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, len(idx))))}, index=idx
    )

    masked = _compute_volatility(
        prices,
        vol_model="ewma",
        vol_lookback=20,
        ewma_lambda=0.94,
        min_vol_periods=20,
        volatility=None,
        vol_warmup="mask",
    )
    backfilled = _compute_volatility(
        prices,
        vol_model="ewma",
        vol_lookback=20,
        ewma_lambda=0.94,
        min_vol_periods=20,
        volatility=None,
        vol_warmup="bfill",
    )

    # Under the default mask the warm-up carries no volatility estimate, so no
    # position is taken; the old bfill filled it from data not yet observed.
    assert masked["BTC"].iloc[0] == 0.0
    assert backfilled["BTC"].iloc[0] > 0.0
    assert masked["BTC"].iloc[-1] == pytest.approx(backfilled["BTC"].iloc[-1])


def test_synthetic_generator_supports_daily_and_correlation() -> None:
    daily = generate_synthetic_prices(["BTC", "ETH"], "2024-01-01", "2024-03-31", freq="D")
    assert daily.index.dayofweek.max() == 6

    correlated = generate_synthetic_prices(
        ["BTC", "ETH"], "2020-01-01", "2024-01-01", freq="D", common_factor=0.8, seed=5
    )
    corr = correlated.pct_change().corr().loc["BTC", "ETH"]
    assert corr > 0.5

    independent = generate_synthetic_prices(
        ["BTC", "ETH"], "2020-01-01", "2024-01-01", freq="D", common_factor=0.0, seed=5
    )
    assert abs(independent.pct_change().corr().loc["BTC", "ETH"]) < 0.2


def test_synthetic_start_offsets_produce_staggered_history() -> None:
    frame = generate_synthetic_prices(
        ["BTC", "SOL"], "2024-01-01", "2024-06-30", freq="D", start_offsets={"SOL": 100}
    )
    assert frame["BTC"].notna().all()
    assert frame["SOL"].isna().sum() == 100
