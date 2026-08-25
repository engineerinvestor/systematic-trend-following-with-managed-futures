"""Crypto data layer: metadata, sessions, rolls, costs (milestone 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.costs.crypto import CryptoCostModel, scale_execution_costs
from tf.data.calendar import TradingCalendar
from tf.data.cme_crypto import (
    CME_CRYPTO_247_START,
    build_universe,
    fill_lag_days,
    last_friday,
    next_session,
    quarterly_roll_schedule,
)
from tf.data.continuous import build_continuous_series
from tf.data.crypto import (
    data_span,
    local_csv_universe,
    spot_universe,
    validate_crypto_prices,
)
from tf.data.ingest import load_prices_or_generate
from tf.data.metadata import ContractMetadata, UniverseDefinition


def test_metadata_accepts_crypto_fields() -> None:
    meta = ContractMetadata.from_dict(
        {
            "symbol": "BTC",
            "sector": "Crypto",
            "point_value": 5.0,
            "tick_size": 5.0,
            "listing_date": "2017-12-18",
            "calendar": "CRYPTO_DAILY",
            "venue": "CME",
            "contract_size": 5.0,
            "session_regime_changes": {"2026-05-29": "daily"},
        }
    )
    assert meta.venue == "CME"
    assert meta.listing_timestamp == pd.Timestamp("2017-12-18")


def test_metadata_still_rejects_typos() -> None:
    with pytest.raises(KeyError):
        ContractMetadata.from_dict(
            {"symbol": "BTC", "sector": "Crypto", "point_value": 5.0, "tickk_size": 5.0}
        )


def test_reference_universe_round_trips() -> None:
    universe = build_universe()
    definition = UniverseDefinition.from_payload(universe)
    assert [c.symbol for c in definition.contracts] == ["BTC", "ETH", "SOL", "XRP"]
    frame = definition.as_dataframe()
    assert frame.loc["XRP", "contract_size"] == 50_000.0
    assert set(frame["venue"]) == {"CME"}


def test_spot_universe_uses_fractional_sizing() -> None:
    universe = spot_universe(["BTC"])
    assert universe[0]["point_value"] == 1.0
    assert universe[0]["contract_step"] < 1.0
    assert universe[0]["data_symbol"] == "BTC-USD"


def test_saturday_signal_fills_at_next_session_before_the_regime_change() -> None:
    saturday = pd.Timestamp("2024-06-01")
    assert saturday.dayofweek == 5
    assert fill_lag_days(saturday) == 2
    assert next_session(saturday) == pd.Timestamp("2024-06-03")


def test_no_fill_lag_once_cme_crypto_trades_around_the_clock() -> None:
    weekend = pd.Timestamp("2026-07-04")  # a Saturday after the change
    assert weekend.dayofweek == 5
    assert weekend > CME_CRYPTO_247_START
    assert fill_lag_days(weekend) == 0
    assert next_session(weekend) == weekend


def test_quarterly_roll_moves_into_the_next_contract() -> None:
    schedule = quarterly_roll_schedule("BTC", "2024-01-01", "2024-12-31")
    dates = [ts for ts, _ in schedule]
    codes = [code for _, code in schedule]

    assert codes[0] == "BTCH24"  # front contract in January
    march_expiry = last_friday(2024, 3)
    raw_roll = march_expiry - pd.Timedelta(days=5)
    # Five calendar days before a Friday expiry is a Sunday; the schedule snaps
    # back to the prior business day so RollEngine's exact-date matching can
    # actually fire on a weekday session index.
    assert dates[1] <= raw_roll
    assert dates[1].dayofweek < 5
    assert (raw_roll - dates[1]).days <= 2
    # Five days before the March expiry the position is in June, not March.
    assert codes[1] == "BTCM24"
    assert dates == sorted(dates)


def test_roll_schedule_feeds_continuous_series() -> None:
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    rng = np.random.default_rng(0)
    contracts = {}
    for code in ("BTCH24", "BTCM24", "BTCU24"):
        contracts[code] = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    prices = pd.DataFrame(contracts, index=idx)

    schedule = quarterly_roll_schedule("BTC", "2024-01-01", "2024-06-30")
    schedule = [(ts, code) for ts, code in schedule if code in prices.columns]
    series = build_continuous_series(prices, schedule, method="stitched")

    assert isinstance(series, pd.Series)
    assert series.notna().sum() > 100
    assert (series > 0).all()


def test_validate_crypto_prices_rejects_gaps_and_bad_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    good = pd.DataFrame({"BTC": np.linspace(100, 110, 10)}, index=idx)
    validate_crypto_prices(good)

    gapped = good.drop(index=idx[4])
    with pytest.raises(ValueError, match="gap"):
        validate_crypto_prices(gapped)

    negative = good.copy()
    negative.iloc[3, 0] = -1.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_crypto_prices(negative)


def test_data_span_reports_per_instrument_history() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {"BTC": np.arange(20.0) + 1, "SOL": [np.nan] * 12 + list(np.arange(8.0) + 1)},
        index=idx,
    )
    span = data_span(frame)
    assert span.loc["BTC", "first_observation"] == idx[0]
    assert span.loc["SOL", "first_observation"] == idx[12]
    assert span.loc["SOL", "observations"] == 8


def test_corrupt_csv_raises_instead_of_becoming_synthetic(tmp_path) -> None:
    """A data-quality failure must never be replaced by generated prices."""

    path = tmp_path / "btc.csv"
    # A negative close: validate_price_data(min_price=0.0) must reject it.
    rows = ["date,close"]
    for i, day in enumerate(pd.date_range("2024-01-01", periods=40, freq="D")):
        close = -5.0 if i == 10 else 100.0 + i
        rows.append(f"{day.date()},{close}")
    path.write_text("\n".join(rows))

    universe = local_csv_universe({"BTC": str(path)})
    with pytest.raises(ValueError):
        load_prices_or_generate(
            universe,
            "2024-01-01",
            "2024-02-09",
            calendar=TradingCalendar.from_name("CRYPTO_DAILY"),
        )


def test_clean_csv_loads_on_a_seven_day_calendar(tmp_path) -> None:
    path = tmp_path / "btc.csv"
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = ["date,close"] + [
        f"{day.date()},{100.0 + i}" for i, day in enumerate(idx)
    ]
    path.write_text("\n".join(rows))

    prices = load_prices_or_generate(
        local_csv_universe({"BTC": str(path)}),
        "2024-01-01",
        "2024-02-09",
        calendar=TradingCalendar.from_name("CRYPTO_DAILY"),
        strict=True,
    )
    assert len(prices) == 40
    assert prices.index.dayofweek.max() == 6  # weekends survived


def test_strict_mode_forbids_the_synthetic_fallback(tmp_path) -> None:
    """A missing file must fail loudly rather than yield generated prices."""

    universe = local_csv_universe({"BTC": str(tmp_path / "absent.csv")})
    with pytest.raises((FileNotFoundError, ValueError)):
        load_prices_or_generate(universe, "2024-01-01", "2024-02-01", strict=True)

    # Without strict the historical fallback still applies, but it is now the
    # only path that can produce synthetic data from a real-data request.
    prices = load_prices_or_generate(universe, "2024-01-01", "2024-02-01")
    assert not prices.empty


def test_cost_model_widens_spreads_in_the_early_era() -> None:
    model = CryptoCostModel()
    assert model.half_spread_for("BTC", "2019-06-01") > model.half_spread_for("BTC", "2025-06-01")
    assert model.half_spread_for("SOL") > model.half_spread_for("BTC")
    # An unknown coin falls back to the widest assumption, never to zero.
    assert model.half_spread_for("DOGE") >= model.half_spread_for("SUI")


def test_cost_multiplier_scales_every_component() -> None:
    base = {"commission_per_contract": 2.5, "impact": {"k": 0.05, "alpha": 0.5}, "min_slippage_ticks": 0.5}
    scaled = scale_execution_costs(base, 4.0)
    assert scaled["commission_per_contract"] == pytest.approx(10.0)
    assert scaled["impact"]["k"] == pytest.approx(0.2)
    assert scaled["min_slippage_ticks"] == pytest.approx(2.0)
    assert scaled["impact"]["alpha"] == 0.5  # exponent is not a cost level


def test_total_cost_is_monotonic_in_the_multiplier() -> None:
    kwargs = dict(
        symbol="BTC", quantity=3.0, price=60_000.0, point_value=5.0, adv=500.0, tick_value=5.0
    )
    cheap = CryptoCostModel(cost_multiplier=1.0).total_cost(**kwargs)
    dear = CryptoCostModel(cost_multiplier=4.0).total_cost(**kwargs)
    assert dear > cheap


def test_leading_gaps_are_allowed_but_interior_gaps_are_not() -> None:
    """A late listing is not missing data; a trading suspension is."""

    from tf.data.validators import validate_price_data

    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    late = pd.DataFrame(
        {"SOL": [np.nan] * 30 + list(np.arange(30, dtype=float) + 100.0)}, index=idx
    )
    validate_price_data(late, min_price=0.0, max_consecutive_missing=5)

    with pytest.raises(ValueError, match="consecutive missing"):
        validate_price_data(
            late, min_price=0.0, max_consecutive_missing=5, allow_leading_gaps=False
        )

    # A hole inside the observed history is a real data-quality failure, of the
    # kind a delisting produces.
    suspended = pd.DataFrame({"XRP": np.arange(60, dtype=float) + 100.0}, index=idx)
    suspended.iloc[20:40, 0] = np.nan
    with pytest.raises(ValueError, match="consecutive missing"):
        validate_price_data(suspended, min_price=0.0, max_consecutive_missing=5)


def test_column_with_no_observations_is_rejected() -> None:
    from tf.data.validators import validate_price_data

    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"BTC": np.arange(10.0) + 1, "GHOST": [np.nan] * 10}, index=idx)
    with pytest.raises(ValueError, match="no observations"):
        validate_price_data(frame, min_price=0.0, max_consecutive_missing=5)
