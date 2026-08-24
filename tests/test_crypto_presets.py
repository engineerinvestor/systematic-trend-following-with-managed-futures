"""Signal fixes, horizon resolution, and reference presets (milestone 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.crypto.horizons import resolve_horizon, resolve_horizons
from tf.crypto.presets import PRESETS, apply_direction, build_signals, get_preset
from tf.data.synthetic import generate_synthetic_prices
from tf.engine.backtester import Backtester
from tf.signals.breakout import channel_breakout, donchian_breakout
from tf.signals.momentum import timeseries_momentum
from tf.signals.moving_average import price_minus_ma


def _trending(direction: int = 1, n: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    start, stop = (100.0, 500.0) if direction > 0 else (500.0, 100.0)
    return pd.DataFrame({"BTC": np.linspace(start, stop, n)}, index=idx)


def _cyclical(period: int, n: int = 1800, phase: float = 0.0) -> pd.DataFrame:
    """A series that reverses every ``period`` bars."""

    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n)
    return pd.DataFrame(
        {"BTC": 200 + 40 * np.sin(2 * np.pi * t / period + phase)}, index=idx
    )


def test_horizons_resolve_against_the_calendar() -> None:
    assert resolve_horizon("365D", "daily") == 365
    assert resolve_horizon("365D", "weekday") == 252
    assert resolve_horizon(30, "daily") == 30
    assert resolve_horizons(("30D", "90D", "365D"), "daily") == (30, 90, 365)


def test_invalid_horizons_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_horizon("not-a-duration", "daily")
    with pytest.raises(ValueError):
        resolve_horizon(0, "daily")
    with pytest.raises(ValueError):
        resolve_horizons((), "daily")


def test_donchian_breakout_actually_breaks_out() -> None:
    """The prior implementation could never reach +/-1; this one must."""

    up = _trending(1)
    signal = donchian_breakout(up, window=50)
    assert signal["BTC"].max() == pytest.approx(1.0)
    assert donchian_breakout(_trending(-1), window=50)["BTC"].min() == pytest.approx(-1.0)


def test_channel_breakout_is_bounded_well_below_one() -> None:
    """Documents the existing oscillator's range so the difference is explicit."""

    signal = channel_breakout(_trending(1), window=50)
    assert signal["BTC"].max() < 0.2


def test_price_minus_ma_uses_the_price_level() -> None:
    up = _trending(1)
    assert price_minus_ma(up, window=100, transform="sign")["BTC"].iloc[-1] == 1.0
    assert price_minus_ma(_trending(-1), window=100, transform="sign")["BTC"].iloc[-1] == -1.0
    with pytest.raises(ValueError):
        price_minus_ma(up, window=100, transform="softmax")


def test_momentum_sign_transform_is_binary() -> None:
    up = _trending(1)
    signal = timeseries_momentum(
        up, lookbacks=(30,), skip_last_n=0, transform="sign", weighting="equal"
    )
    values = set(np.unique(signal["BTC"].to_numpy()))
    assert values <= {-1.0, 0.0, 1.0}


def test_momentum_rejects_unknown_options() -> None:
    up = _trending(1)
    with pytest.raises(ValueError):
        timeseries_momentum(up, transform="quadratic")
    with pytest.raises(ValueError):
        timeseries_momentum(up, weighting="golden")


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_follows_the_trend(name: str) -> None:
    direction = "long_flat" if name == "btc_long_flat" else "long_short"
    assert build_signals(_trending(1), name, direction=direction)["BTC"].iloc[-1] > 0


@pytest.mark.parametrize("name", ["mop2012_tsmom", "tsmom_1_3_12", "bottom_up_multisystem"])
def test_long_short_presets_go_short_in_a_downtrend(name: str) -> None:
    assert build_signals(_trending(-1), name)["BTC"].iloc[-1] < 0


def test_long_flat_never_shorts() -> None:
    signal = build_signals(_trending(-1), "btc_long_flat", direction="long_flat")
    assert signal["BTC"].min() >= 0.0
    assert signal["BTC"].iloc[-1] == 0.0


def test_long_flat_direction_clips_a_long_short_preset() -> None:
    down = _trending(-1)
    assert build_signals(down, "tsmom_1_3_12", direction="long_flat")["BTC"].iloc[-1] == 0.0
    assert build_signals(down, "tsmom_1_3_12", direction="long_short")["BTC"].iloc[-1] < 0.0


def test_apply_direction_rejects_unknown_modes() -> None:
    frame = pd.DataFrame({"BTC": [1.0, -1.0]})
    with pytest.raises(ValueError):
        apply_direction(frame, "long_only")


def test_mop_preset_refuses_a_skip_period() -> None:
    """Time-series momentum has no skip month; a config asking for one must fail."""

    with pytest.raises(ValueError, match="skip"):
        build_signals(_trending(1), "mop2012_tsmom", skip_last_n=20)


def test_btc_long_flat_refuses_long_short() -> None:
    with pytest.raises(ValueError):
        build_signals(_trending(1), "btc_long_flat", direction="long_short")


def test_unknown_preset_lists_the_available_ones() -> None:
    with pytest.raises(KeyError, match="tsmom_1_3_12"):
        get_preset("aqr_2012")


@pytest.mark.parametrize("phase", [0.0, 0.5, 1.0, 1.5])
def test_trend_loses_money_when_the_series_reverses_at_its_own_horizon(
    phase: float,
) -> None:
    """A trend system must be punished by reversals at the horizon it measures.

    The reversal has to happen at the signal's own horizon. A 30-bar signal on a
    60-bar cycle is maximally out of phase, so it buys every top and sells every
    bottom. Short-horizon mean reversion does not do this: a 12-month signal is
    simply blind to day-to-day reversals, which is why a negatively
    autocorrelated daily series leaves long-horizon momentum profitable.
    """

    prices = _cyclical(period=60, phase=phase)
    signal = build_signals(prices, "mop2012_tsmom", horizons=("30D",))
    forward_returns = prices["BTC"].pct_change().shift(-1)
    pnl = (signal["BTC"] * forward_returns).dropna()
    assert pnl.sum() < 0


def test_long_horizon_trend_survives_short_horizon_mean_reversion() -> None:
    """Documents the converse, which is easy to assume away."""

    n = 2000
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(4)
    shocks = rng.normal(0.0, 0.01, n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = -0.5 * returns[i - 1] + shocks[i]  # strong daily reversal
    prices = pd.DataFrame({"BTC": 100 * np.exp(np.cumsum(returns))}, index=idx)

    signal = build_signals(prices, "tsmom_1_3_12")
    forward_returns = prices["BTC"].pct_change().shift(-1)
    pnl = (signal["BTC"] * forward_returns).dropna()
    # The 1, 3, and 12-month horizons never sample the reverting component.
    assert pnl.sum() > 0


def test_multisystem_ensemble_averages_its_components() -> None:
    prices = _trending(1)
    single = build_signals(prices, "bottom_up_multisystem", systems=("pmac",), horizons=(50,))
    combined = build_signals(
        prices, "bottom_up_multisystem", systems=("pmac", "breakout"), horizons=(50,)
    )
    assert single["BTC"].iloc[-1] == pytest.approx(1.0)
    assert -1.0 <= combined["BTC"].iloc[-1] <= 1.0


def test_multisystem_rejects_unknown_systems() -> None:
    with pytest.raises(ValueError, match="Unknown trend systems"):
        build_signals(_trending(1), "bottom_up_multisystem", systems=("rsi",))


def test_preset_runs_end_to_end_on_a_seven_day_calendar(tmp_path) -> None:
    prices = generate_synthetic_prices(
        ["BTC", "ETH"], "2021-01-01", "2023-12-31", freq="D", common_factor=0.6, seed=3
    )
    universe = [
        {"symbol": "BTC", "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4},
        {"symbol": "ETH", "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4},
    ]
    cfg = {
        "data": {"calendar": "CRYPTO_DAILY"},
        "backtest": {
            "start": "2021-01-01",
            "end": "2023-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12", "direction": "long_short"},
        "risk": {
            "periods_per_year": 365,
            "ewma_center_of_mass": 60,
            "target_portfolio_vol": 0.10,
            "max_asset_weight": 0.5,
        },
        "execution": {"adv_limit_pct": 0.0},
    }
    result = Backtester(prices, universe, cfg).run()

    assert not result.nav.empty
    assert (result.nav.index.dayofweek >= 5).sum() > 0  # weekends are traded
    # Accounting identity: NAV equals cash plus marked asset value throughout.
    ledger = result.ledger
    assert (ledger["nav"] - ledger["cash"] - ledger["asset_value"]).abs().max() < 1e-6


def test_legacy_momentum_config_still_dispatches(tmp_path) -> None:
    """A config without a preset must behave exactly as it did before."""

    prices = generate_synthetic_prices(["ES"], "2021-01-01", "2022-12-31", seed=1)
    universe = [{"symbol": "ES", "sector": "Equities", "point_value": 50}]
    cfg = {
        "backtest": {
            "start": "2021-01-01",
            "end": "2022-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"momentum": {"lookbacks": [20, 60], "skip_last_n": 5}},
        "risk": {"target_portfolio_vol": 0.12},
        "execution": {"adv_limit_pct": 1.0, "adv_contracts": {"ES": 1000}},
    }
    result = Backtester(prices, universe, cfg).run()
    assert not result.nav.empty


def test_disable_trend_and_vol_scaling_toggles(tmp_path) -> None:
    """The 2x2 decomposition needs both knobs to actually change the result."""

    prices = generate_synthetic_prices(
        ["BTC", "ETH"], "2021-01-01", "2023-12-31", freq="D", seed=9
    )
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "ETH")
    ]
    base = {
        "data": {"calendar": "CRYPTO_DAILY"},
        "backtest": {
            "start": "2021-01-01",
            "end": "2023-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12"},
        "risk": {"periods_per_year": 365, "target_portfolio_vol": 0.10},
        "execution": {"adv_limit_pct": 0.0},
    }
    trend = Backtester(prices, universe, base).run()

    no_trend = Backtester(prices, universe, base).run(
        parameter_overrides={"signals": {"disable_trend": True}}
    )
    no_vol = Backtester(prices, universe, base).run(
        parameter_overrides={"risk": {"disable_vol_scaling": True}}
    )

    assert not trend.nav.equals(no_trend.nav)
    assert not trend.nav.equals(no_vol.nav)
