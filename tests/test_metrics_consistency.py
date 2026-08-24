"""Regression tests for the adversarial-review metrics findings (R1, R7, R11)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tf import api
from tf.cli import _periods_per_year_from, _write_backtest_outputs
from tf.data.synthetic import generate_synthetic_prices
from tf.engine.backtester import Backtester
from tf.eval.metrics import CALENDAR_DAYS, performance_summary


def _crypto_cfg(tmp_path) -> dict:
    return {
        "data": {"calendar": "CRYPTO_DAILY", "prefer": "synthetic"},
        "backtest": {
            "start": "2021-01-01",
            "end": "2022-12-31",
            "starting_nav": 1_000_000.0,
            "seed": 7,
            "results_dir": str(tmp_path),
        },
        "signals": {"preset": "tsmom_1_3_12"},
        "risk": {"periods_per_year": 365, "target_portfolio_vol": 0.10},
        "execution": {"adv_limit_pct": 0.0},
        "universe": {"assets_file": "configs/crypto/universe.yaml"},
    }


def test_tf_run_summary_uses_the_config_annualisation(tmp_path) -> None:
    """R1: sizing at 365 while reporting at 252 mis-states Sharpe/vol by ~17%."""

    cfg = _crypto_cfg(tmp_path)
    result, context = api.run_backtest(cfg)
    outdir = tmp_path / "run-x"
    _write_backtest_outputs(result, config=context.config, outdir=outdir, metadata={})
    written = json.loads((outdir / "summary.json").read_text())

    expected = performance_summary(
        result.nav, trades=result.trades, periods_per_year=CALENDAR_DAYS
    )
    assert written["Volatility"] == pytest.approx(expected["Volatility"])
    assert written["Sharpe"] == pytest.approx(expected["Sharpe"], nan_ok=True)
    assert written["CAGR"] == pytest.approx(expected["CAGR"])

    wrong = performance_summary(result.nav, trades=result.trades)  # 252 default
    if expected["Volatility"] > 0:
        assert written["Volatility"] != pytest.approx(wrong["Volatility"])


def test_periods_per_year_helper_defaults_and_reads_config() -> None:
    assert _periods_per_year_from({}) == 252
    assert _periods_per_year_from({"risk": {"periods_per_year": 365}}) == 365
    assert _periods_per_year_from({"risk": "not-a-mapping"}) == 252


def test_walkforward_summaries_use_the_config_annualisation(tmp_path) -> None:
    """R1: per-fold IS/OOS metrics must not silently revert to 252."""

    cfg = _crypto_cfg(tmp_path)
    prices = generate_synthetic_prices(
        ["BTC", "ETH"], "2021-01-01", "2022-12-31", freq="D", seed=3
    )
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "ETH")
    ]
    bt = Backtester(prices, universe, cfg)
    folds = bt.run_walk_forward(insample=200, oos=60)
    assert folds
    fold = folds[0]
    assert fold.periods_per_year == 365
    expected = performance_summary(fold.oos_nav, periods_per_year=365)
    actual = fold.oos_summary
    actual = actual() if callable(actual) else actual
    assert actual["Volatility"] == pytest.approx(expected["Volatility"], nan_ok=True)
    wrong = performance_summary(fold.oos_nav)  # 252 default
    if expected["Volatility"] > 0:
        assert actual["Volatility"] != pytest.approx(wrong["Volatility"])


def test_turnover_is_full_period_and_annualised_variant_matches() -> None:
    """R7: the two turnover figures must relate by the window length in years."""

    idx = pd.date_range("2020-01-01", periods=731, freq="D")  # two years, 365 basis
    nav = pd.Series(1_000_000.0, index=idx)
    trades = pd.DataFrame({"notional": [500_000.0, -500_000.0, 1_000_000.0]})

    summary = performance_summary(nav, trades=trades, periods_per_year=365)
    assert summary["Turnover"] == pytest.approx(2.0)
    assert summary["Turnover (ann.)"] == pytest.approx(1.0, rel=1e-2)


def test_sortino_and_calmar_are_nan_when_undefined() -> None:
    """R11: a window with no losing day must not report the worst-looking 0.0."""

    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    monotone = pd.Series(np.linspace(1_000_000, 1_100_000, 100), index=idx)
    summary = performance_summary(monotone, periods_per_year=365)
    assert np.isnan(summary["Sortino"])
    assert np.isnan(summary["Calmar"])
    assert summary["Sharpe"] > 0


def test_bankrupt_and_recover_reports_total_loss() -> None:
    """R11: intermediate non-positive NAV is a wipeout, not a recovery story."""

    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    values = np.concatenate([
        np.linspace(1_000_000, -100_000, 100),
        np.linspace(-100_000, 1_500_000, 100),
    ])
    nav = pd.Series(values, index=idx)
    summary = performance_summary(nav, periods_per_year=365)
    assert summary["CAGR"] == -1.0
    assert summary["Max Drawdown"] >= -1.0


def test_sensitivity_turnover_sweep_is_not_structurally_zero(tmp_path) -> None:
    """R1: sensitivity must pass trades through, or Turnover sweeps all-zero."""

    from tf.research.sensitivity import compute_metric_sensitivity

    prices = generate_synthetic_prices(
        ["BTC", "ETH"], "2021-01-01", "2022-06-30", freq="D", seed=5
    )
    universe = [
        {"symbol": s, "sector": "Crypto", "point_value": 1.0, "contract_step": 1e-4}
        for s in ("BTC", "ETH")
    ]
    cfg = _crypto_cfg(tmp_path)
    cfg["backtest"]["end"] = "2022-06-30"
    bt = Backtester(prices, universe, cfg)
    frame = compute_metric_sensitivity(
        bt, "risk.target_portfolio_vol", [0.05, 0.10], metric="Turnover"
    )
    assert (frame["Turnover"] > 0).all()
