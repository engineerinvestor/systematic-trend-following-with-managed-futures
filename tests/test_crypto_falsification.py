"""Falsification suite (milestone 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tf.cli import main as cli_main
from tf.data.synthetic import generate_synthetic_prices
from tf.engine.backtester import Backtester
from tf.eval.falsification import (
    attribution_by_asset,
    attribution_by_side,
    buy_and_hold_nav,
    fast_nav_from_signals,
    run_falsification,
    run_placebo,
    run_vol_scaling_2x2,
    vol_managed_nav,
)
from tf.eval.metrics import performance_summary


def _prices(symbols=("BTC", "ETH", "SOL"), seed: int = 11) -> pd.DataFrame:
    return generate_synthetic_prices(
        list(symbols), "2020-01-01", "2021-12-31", freq="D", common_factor=0.6, seed=seed
    )


def _config(tmp_path, **overrides) -> dict:
    cfg = {
        "data": {"calendar": "CRYPTO_DAILY"},
        "backtest": {
            "start": "2020-01-01",
            "end": "2021-12-31",
            "starting_nav": 1_000_000.0,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12", "direction": "long_short"},
        "risk": {
            "periods_per_year": 365,
            "ewma_center_of_mass": 60,
            "target_portfolio_vol": 0.10,
            "max_asset_weight": 0.6,
        },
        "execution": {
            "adv_limit_pct": 0.5,
            "adv_contracts": {"BTC": 20000, "ETH": 20000, "SOL": 20000},
            "commission_per_contract": 0.01,
            "impact": {"k": 0.02, "alpha": 0.5},
            "min_slippage_ticks": 0.5,
            "tick_value": 0.01,
        },
    }
    cfg.update(overrides)
    return cfg


def _backtester(tmp_path, symbols=("BTC", "ETH", "SOL")) -> Backtester:
    prices = _prices(symbols)
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in symbols
    ]
    return Backtester(prices, universe, _config(tmp_path))


def test_buy_and_hold_matches_a_hand_computed_return() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"BTC": [100.0, 110.0, 121.0]}, index=idx)
    nav = buy_and_hold_nav(prices, starting_nav=1_000.0)
    assert nav.iloc[-1] == pytest.approx(1_210.0)


def test_vol_managed_buy_and_hold_reduces_realised_volatility() -> None:
    prices = _prices(("BTC",))
    plain = buy_and_hold_nav(prices)
    managed = vol_managed_nav(prices, target_vol=0.10, periods_per_year=365)

    plain_vol = performance_summary(plain, periods_per_year=365)["Volatility"]
    managed_vol = performance_summary(managed, periods_per_year=365)["Volatility"]
    assert managed_vol < plain_vol


def test_vol_scaling_2x2_covers_every_cell(tmp_path) -> None:
    frame = run_vol_scaling_2x2(_backtester(tmp_path), periods_per_year=365)
    assert set(frame.index) == {("no", "no"), ("no", "yes"), ("yes", "no"), ("yes", "yes")}
    assert {"CAGR", "Volatility", "Sharpe", "Max Drawdown"} <= set(frame.columns)

    # Volatility scaling must actually change realised volatility.
    assert frame.loc[("yes", "yes"), "Volatility"] != frame.loc[("yes", "no"), "Volatility"]
    # Disabling the trend signal must change the result.
    assert frame.loc[("yes", "yes"), "CAGR"] != frame.loc[("no", "yes"), "CAGR"]


def test_placebo_mean_sharpe_is_near_zero() -> None:
    """A signal stripped of direction must have no expected edge."""

    prices = _prices(("BTC", "ETH", "SOL", "XRP"), seed=5)
    rng = np.random.default_rng(0)
    signals = pd.DataFrame(
        rng.choice((-1.0, 1.0), size=prices.shape), index=prices.index, columns=prices.columns
    )
    frame = run_placebo(
        prices,
        signals,
        periods_per_year=365,
        target_vol=0.10,
        starting_nav=1_000_000.0,
        draws=64,
        seed=3,
    )
    row = frame.loc["Sharpe"]
    # The placebo distribution is symmetric about zero by construction, so its
    # mean must sit within a fraction of its own spread of zero.
    assert abs(row["placebo_mean"]) < 0.5 * max(row["placebo_std"], 1e-9)


def test_placebo_reports_its_own_coarseness() -> None:
    prices = _prices(("BTC", "ETH"))
    signals = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    frame = run_placebo(
        prices, signals, periods_per_year=365, target_vol=0.1,
        starting_nav=1_000_000.0, draws=50, seed=1,
    )
    # Two instruments allow only four distinct sign combinations.
    assert frame.loc["Sharpe", "distinct_sign_combinations"] == 4
    assert frame.attrs["is_coarse"] is True


def test_fast_nav_path_tracks_the_direction_of_a_trend() -> None:
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    prices = pd.DataFrame({"BTC": np.linspace(100.0, 300.0, 600)}, index=idx)
    long_signal = pd.DataFrame(1.0, index=idx, columns=["BTC"])
    short_signal = pd.DataFrame(-1.0, index=idx, columns=["BTC"])

    up = fast_nav_from_signals(prices, long_signal, periods_per_year=365)
    down = fast_nav_from_signals(prices, short_signal, periods_per_year=365)
    assert up.iloc[-1] > up.iloc[0]
    assert down.iloc[-1] < down.iloc[0]


def test_attribution_by_side_reconciles_to_net(tmp_path) -> None:
    result = _backtester(tmp_path).run()
    frame = attribution_by_side(result)
    assert set(frame.index) == {"long", "short", "total gross", "costs", "net"}

    gross = frame.loc["long", "gross_pnl"] + frame.loc["short", "gross_pnl"]
    assert frame.loc["total gross", "gross_pnl"] == pytest.approx(gross)
    assert frame.loc["net", "gross_pnl"] == pytest.approx(
        gross + frame.loc["costs", "gross_pnl"]
    )


def test_attribution_by_asset_shares_sum_to_one(tmp_path) -> None:
    result = _backtester(tmp_path).run()
    frame = attribution_by_asset(result)
    assert not frame.empty
    assert frame["share"].sum() == pytest.approx(1.0)


def test_cost_stress_bites_when_costs_are_configured(tmp_path) -> None:
    report = run_falsification(_backtester(tmp_path), placebo_draws=8, seed=0)
    stress = report.cost_stress
    assert list(stress.index) == ["1x costs", "2x costs", "4x costs"]
    # With real costs configured the rows must differ.
    assert stress["Sharpe"].nunique() > 1
    assert not stress.attrs.get("is_frictionless")


def test_frictionless_config_is_flagged(tmp_path) -> None:
    prices = _prices(("BTC", "ETH"))
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "ETH")
    ]
    cfg = _config(tmp_path)
    cfg["execution"] = {"adv_limit_pct": 0.0}  # no costs at all
    report = run_falsification(Backtester(prices, universe, cfg), placebo_draws=4, seed=0)
    assert report.cost_stress.attrs.get("is_frictionless")
    assert any("costs are configured at zero" in note for note in report.notes)


def test_full_report_produces_every_table(tmp_path) -> None:
    report = run_falsification(_backtester(tmp_path), placebo_draws=8, seed=0)
    titles = [title for title, frame in report.tables() if not frame.empty]
    for expected in (
        "Strategy vs. Benchmarks",
        "Trend Signal vs. Volatility Scaling",
        "Randomised-Signal Placebo",
        "Signal Delay Sensitivity",
        "Transaction Cost Stress",
        "Volatility Lookback Sensitivity",
        "Leave-One-Asset-Out",
        "Attribution by Asset",
        "Attribution by Side",
        "Data Span",
    ):
        assert expected in titles

    assert "strategy" in report.variants.index
    assert len(report.leave_one_out) == 4  # all assets plus one row per drop
    assert list(report.signal_delay.index)[0] == "delay +0"


def test_report_warns_when_annualisation_is_left_at_252(tmp_path) -> None:
    prices = _prices(("BTC", "ETH"))
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "ETH")
    ]
    cfg = _config(tmp_path)
    cfg["risk"].pop("periods_per_year")
    report = run_falsification(Backtester(prices, universe, cfg), placebo_draws=4, seed=0)
    assert any("periods_per_year" in note for note in report.notes)


def test_cli_falsify_writes_a_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    import shutil
    from pathlib import Path

    import tf

    source = Path(tf.__file__).resolve().parents[2] / "configs"
    shutil.copytree(source, tmp_path / "configs")

    # Shorten the window so the test exercises the wiring, not the arithmetic.
    config_path = tmp_path / "configs" / "crypto" / "tsmom_1_3_12.yaml"
    config_path.write_text(
        config_path.read_text()
        .replace('start: "2018-01-01"', 'start: "2020-01-01"')
        .replace('end: "2024-12-31"', 'end: "2021-12-31"')
    )

    cli_main(
        [
            "crypto",
            "falsify",
            "--config",
            "configs/crypto/tsmom_1_3_12.yaml",
            "--run-id",
            "cli-test",
            "--placebo-draws",
            "4",
        ]
    )

    outdir = tmp_path / "results" / "cli-test"
    assert (outdir / "report.html").exists()
    assert (outdir / "trend_signal_vs_volatility_scaling.csv").exists()
    html = (outdir / "report.html").read_text()
    assert "Trend Signal vs. Volatility Scaling" in html
    assert "Randomised-Signal Placebo" in html


def test_cost_bite_detects_a_negligible_cost_model() -> None:
    """Costs too small to constrain a strategy are as misleading as no costs.

    The threshold is annualised turnover: a full-period figure over a long
    window would trip the detector on strategies that barely trade.
    """

    from tf.eval.falsification import _cost_bite

    barely = pd.DataFrame(
        {"CAGR": [0.0966, 0.0964, 0.0959], "Turnover (ann.)": [8.6, 8.6, 8.6]},
        index=["1x costs", "2x costs", "4x costs"],
    )
    attrs = _cost_bite(barely)
    assert attrs["costs_barely_bite"] is True
    assert attrs["is_frictionless"] is False

    biting = pd.DataFrame(
        {"CAGR": [0.09, 0.05, -0.02], "Turnover (ann.)": [8.6, 8.4, 8.2]},
        index=["1x costs", "2x costs", "4x costs"],
    )
    assert _cost_bite(biting)["costs_barely_bite"] is False

    # Low turnover is a reason for costs not to matter, not a warning sign.
    quiet = pd.DataFrame(
        {"CAGR": [0.09, 0.09, 0.09], "Turnover (ann.)": [0.03, 0.03, 0.03]},
        index=["1x costs", "2x costs", "4x costs"],
    )
    assert _cost_bite(quiet)["costs_barely_bite"] is False
