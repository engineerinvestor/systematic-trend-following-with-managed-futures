"""Falsification suite: evidence that a strategy is not what it appears to be.

Most backtest reports are built to show a strategy working. This one is built
to find the ways it does not. Every result is placed beside naive benchmarks,
re-run at higher costs and later signals, decomposed into the parts contributed
by the trend signal and by volatility scaling, and compared against a placebo
whose signals carry no information.

Nothing here proves a strategy works. Surviving it means the obvious
explanations for a result have been ruled out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..costs.crypto import scale_execution_costs
from ..engine.backtester import Backtester, BacktestResults
from ..eval.analytics import compute_pnl_contributions
from ..eval.metrics import TRADING_DAYS, performance_summary
from ..research.monte_carlo import bootstrap_confidence_intervals

logger = logging.getLogger(__name__)

#: Signal delays applied on top of the strategy's own lag. If a result survives
#: only at zero extra delay it depends on trading instantly on the signal.
DEFAULT_SIGNAL_DELAYS = (1, 2, 5)

#: Multiples of the configured cost model.
DEFAULT_COST_MULTIPLES = (1.0, 2.0, 4.0)

#: Alternative volatility lookbacks, in bars.
DEFAULT_VOL_LOOKBACKS = (20, 40, 60, 90)

#: Draws for the randomised-sign placebo.
DEFAULT_PLACEBO_DRAWS = 1_000


@dataclass
class FalsificationReport:
    """Everything the suite produced, as tables ready for rendering."""

    variants: pd.DataFrame
    vol_scaling_2x2: pd.DataFrame
    signal_delay: pd.DataFrame
    cost_stress: pd.DataFrame
    vol_lookback: pd.DataFrame
    leave_one_out: pd.DataFrame
    placebo: pd.DataFrame
    attribution: pd.DataFrame
    side_attribution: pd.DataFrame
    data_span: pd.DataFrame
    confidence_intervals: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def tables(self) -> list[tuple[str, pd.DataFrame]]:
        """Return ``(title, frame)`` pairs in the order they should be read."""

        return [
            ("Strategy vs. Benchmarks", self.variants),
            ("Trend Signal vs. Volatility Scaling", self.vol_scaling_2x2),
            ("Randomised-Signal Placebo", self.placebo),
            ("Signal Delay Sensitivity", self.signal_delay),
            ("Transaction Cost Stress", self.cost_stress),
            ("Volatility Lookback Sensitivity", self.vol_lookback),
            ("Leave-One-Asset-Out", self.leave_one_out),
            ("Attribution by Asset", self.attribution),
            ("Attribution by Side", self.side_attribution),
            ("Data Span", self.data_span),
            ("Bootstrap Confidence Intervals", self.confidence_intervals),
        ]


def _summary_row(
    result: BacktestResults, label: str, periods_per_year: int
) -> dict[str, object]:
    summary = performance_summary(
        result.nav, trades=result.trades, periods_per_year=periods_per_year
    )
    row: dict[str, object] = {"variant": label}
    row.update(summary)
    return row


def _run(backtester: Backtester, overrides: Mapping[str, object] | None = None):
    return backtester.run(parameter_overrides=dict(overrides or {}))


def _periods_per_year(cfg: Mapping[str, object]) -> int:
    risk_cfg = cfg.get("risk", {}) or {}
    return int(risk_cfg.get("periods_per_year", TRADING_DAYS))


def buy_and_hold_nav(
    prices: pd.DataFrame,
    *,
    weights: Mapping[str, float] | None = None,
    starting_nav: float = 1_000_000.0,
) -> pd.Series:
    """Equal-weighted buy and hold, rebalanced never."""

    returns = prices.pct_change().fillna(0.0)
    if weights is None:
        active = prices.notna()
        weight_frame = active.div(active.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        weight_frame = pd.DataFrame(
            {col: float(weights.get(col, 0.0)) for col in prices.columns},
            index=prices.index,
        )
    portfolio_returns = (returns * weight_frame).sum(axis=1)
    return starting_nav * (1.0 + portfolio_returns).cumprod()


def vol_managed_nav(
    prices: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    lookback: int = 60,
    periods_per_year: int = 365,
    starting_nav: float = 1_000_000.0,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Buy and hold scaled to a volatility target, with no trend signal.

    This is the control for the volatility-scaling critique: if it captures most
    of a strategy's result, the trend signal is not what produced it.
    """

    returns = prices.pct_change().fillna(0.0)
    active = prices.notna()
    weight_frame = active.div(active.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    portfolio_returns = (returns * weight_frame).sum(axis=1)

    realised = portfolio_returns.ewm(span=lookback, min_periods=lookback).std()
    realised = realised * np.sqrt(periods_per_year)
    # Lagged so the scaling uses only information available beforehand.
    leverage = (target_vol / realised.replace(0.0, np.nan)).shift(1)
    leverage = leverage.clip(upper=max_leverage).fillna(0.0)

    scaled = portfolio_returns * leverage
    return starting_nav * (1.0 + scaled).cumprod()


def fast_nav_from_signals(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    target_vol: float = 0.10,
    lookback: int = 60,
    periods_per_year: int = 365,
    starting_nav: float = 1_000_000.0,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Vectorised signal-to-NAV path, gross of the order queue.

    The placebo needs hundreds of runs, and a full order-book simulation for
    each is not viable. A placebo tests whether a signal carries information,
    not whether it can be executed, so this path skips queueing, partial fills,
    and commissions. Do not use it for a headline result.
    """

    returns = prices.pct_change().fillna(0.0)
    aligned = signals.reindex_like(prices).fillna(0.0)

    instrument_vol = returns.ewm(span=lookback, min_periods=lookback).std()
    instrument_vol = (instrument_vol * np.sqrt(periods_per_year)).shift(1)
    instrument_vol = instrument_vol.replace(0.0, np.nan)

    gross = aligned.abs().sum(axis=1).replace(0.0, np.nan)
    budget = aligned.abs().div(gross, axis=0).fillna(0.0)
    weights = (budget * target_vol).div(instrument_vol).replace([np.inf, -np.inf], 0.0)
    weights = weights.fillna(0.0) * np.sign(aligned)

    leverage = weights.abs().sum(axis=1)
    scale = (max_leverage / leverage.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(scale, axis=0)

    portfolio_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    return starting_nav * (1.0 + portfolio_returns).cumprod()


def _nav_summary(nav: pd.Series, label: str, periods_per_year: int) -> dict[str, object]:
    row: dict[str, object] = {"variant": label}
    row.update(performance_summary(nav, periods_per_year=periods_per_year))
    return row


def run_benchmark_battery(
    backtester: Backtester,
    *,
    prices: pd.DataFrame,
    base_result: BacktestResults,
    periods_per_year: int,
    starting_nav: float,
    target_vol: float,
) -> pd.DataFrame:
    """Compare the strategy against benchmarks that require no skill."""

    rows = [_summary_row(base_result, "strategy", periods_per_year)]

    first_column = prices.columns[0]
    single = prices[[first_column]]
    rows.append(
        _nav_summary(
            buy_and_hold_nav(single, starting_nav=starting_nav),
            f"{first_column} buy and hold",
            periods_per_year,
        )
    )
    rows.append(
        _nav_summary(
            buy_and_hold_nav(prices, starting_nav=starting_nav),
            "equal-weight buy and hold",
            periods_per_year,
        )
    )
    rows.append(
        _nav_summary(
            vol_managed_nav(
                prices,
                target_vol=target_vol,
                periods_per_year=periods_per_year,
                starting_nav=starting_nav,
            ),
            "vol-managed buy and hold",
            periods_per_year,
        )
    )

    for label, overrides in (
        (
            "200-day long/flat",
            {"signals": {"preset": "btc_long_flat", "direction": "long_flat"}},
        ),
        (
            "single 12-month horizon",
            {"signals": {"preset": "mop2012_tsmom", "horizons": ["365D"]}},
        ),
    ):
        try:
            rows.append(_summary_row(_run(backtester, overrides), label, periods_per_year))
        except Exception as exc:  # pragma: no cover - benchmark is optional
            logger.warning("Benchmark %s failed: %s", label, exc)

    return pd.DataFrame(rows).set_index("variant")


def run_vol_scaling_2x2(
    backtester: Backtester,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    """Decompose a result into trend-signal and volatility-scaling parts.

    Published time-series momentum performance is substantially attributable to
    volatility scaling rather than the directional signal, so a report that does
    not separate the two cannot say which one produced its numbers.
    """

    cells = {
        ("no", "no"): {"signals": {"disable_trend": True}, "risk": {"disable_vol_scaling": True}},
        ("no", "yes"): {"signals": {"disable_trend": True}},
        ("yes", "no"): {"risk": {"disable_vol_scaling": True}},
        ("yes", "yes"): {},
    }

    rows = []
    for (trend, vol), overrides in cells.items():
        result = _run(backtester, overrides)
        summary = performance_summary(
            result.nav, trades=result.trades, periods_per_year=periods_per_year
        )
        rows.append(
            {
                "trend_signal": trend,
                "vol_scaling": vol,
                **{k: summary[k] for k in ("CAGR", "Volatility", "Sharpe", "Max Drawdown")},
            }
        )
    frame = pd.DataFrame(rows)
    return frame.set_index(["trend_signal", "vol_scaling"]).sort_index()


def run_signal_delay(
    backtester: Backtester,
    *,
    periods_per_year: int,
    delays: Sequence[int] = DEFAULT_SIGNAL_DELAYS,
) -> pd.DataFrame:
    """Re-run with the signal pushed further into the future."""

    rows = [_summary_row(_run(backtester), "delay +0", periods_per_year)]
    for delay in delays:
        overrides = {"signals": {"lag": 1 + int(delay)}}
        try:
            rows.append(
                _summary_row(_run(backtester, overrides), f"delay +{delay}", periods_per_year)
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Signal delay %s failed: %s", delay, exc)
    return pd.DataFrame(rows).set_index("variant")


def run_cost_stress(
    backtester: Backtester,
    base_config: Mapping[str, object],
    *,
    periods_per_year: int,
    multiples: Sequence[float] = DEFAULT_COST_MULTIPLES,
) -> pd.DataFrame:
    """Re-run at multiples of the configured cost model."""

    execution_cfg = base_config.get("execution", {}) or {}
    rows = []
    for multiple in multiples:
        overrides = {"execution": scale_execution_costs(execution_cfg, float(multiple))}
        rows.append(
            _summary_row(_run(backtester, overrides), f"{multiple:g}x costs", periods_per_year)
        )
    frame = pd.DataFrame(rows).set_index("variant")
    frame.attrs.update(_cost_bite(frame))
    return frame


#: Annual return difference below which quadrupling costs counts as no effect.
NEGLIGIBLE_COST_EFFECT = 0.001

#: Turnover above which costs are expected to matter at all.
MATERIAL_TURNOVER = 1.0


def _cost_bite(frame: pd.DataFrame) -> dict[str, object]:
    """Judge whether the cost stress actually changed anything.

    Identical rows mean costs are configured at zero. Rows that barely move
    despite heavy trading mean the cost model is calibrated too small to
    constrain the strategy, which is the same problem wearing a disguise: the
    net figures are effectively gross either way.
    """

    if len(frame) < 2:
        return {"is_frictionless": False, "cost_effect": 0.0}

    spread = float(frame["CAGR"].max() - frame["CAGR"].min())
    turnover = float(frame["Turnover"].max()) if "Turnover" in frame else 0.0
    return {
        "is_frictionless": bool(frame["CAGR"].nunique() == 1),
        "cost_effect": spread,
        "costs_barely_bite": bool(
            spread < NEGLIGIBLE_COST_EFFECT and turnover > MATERIAL_TURNOVER
        ),
        "turnover": turnover,
    }


def run_vol_lookback(
    backtester: Backtester,
    *,
    periods_per_year: int,
    lookbacks: Sequence[int] = DEFAULT_VOL_LOOKBACKS,
) -> pd.DataFrame:
    """Re-run across alternative volatility estimation windows."""

    rows = []
    for lookback in lookbacks:
        overrides = {
            "risk": {"ewma_center_of_mass": float(lookback), "vol_lookback": int(lookback)}
        }
        rows.append(
            _summary_row(_run(backtester, overrides), f"{lookback}-bar vol", periods_per_year)
        )
    return pd.DataFrame(rows).set_index("variant")


def run_leave_one_out(
    backtester: Backtester,
    prices: pd.DataFrame,
    *,
    periods_per_year: int,
) -> pd.DataFrame:
    """Drop each instrument in turn to see whether one name carries the result."""

    rows = [_summary_row(_run(backtester), "all assets", periods_per_year)]
    if len(prices.columns) < 2:
        return pd.DataFrame(rows).set_index("variant")

    for column in prices.columns:
        remaining = [c for c in prices.columns if c != column]
        subset = prices[remaining]
        universe = [u for u in backtester.universe_meta if u["symbol"] in remaining]
        try:
            trimmed = Backtester(subset, universe, backtester._base_cfg)
            rows.append(_summary_row(_run(trimmed), f"without {column}", periods_per_year))
        except Exception as exc:  # pragma: no cover
            logger.warning("Leave-one-out for %s failed: %s", column, exc)
    return pd.DataFrame(rows).set_index("variant")


def run_placebo(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    periods_per_year: int,
    target_vol: float,
    starting_nav: float,
    draws: int = DEFAULT_PLACEBO_DRAWS,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Compare the strategy to signals with the same shape but random signs.

    The placebo keeps the strategy's position sizes and timing and randomises
    only the direction, so it isolates whether the signal's direction carried
    information. A strategy whose Sharpe sits inside the placebo distribution
    has not demonstrated anything.
    """

    rng = np.random.default_rng(seed)
    actual_nav = fast_nav_from_signals(
        prices,
        signals,
        target_vol=target_vol,
        periods_per_year=periods_per_year,
        starting_nav=starting_nav,
    )
    actual = performance_summary(actual_nav, periods_per_year=periods_per_year)["Sharpe"]

    sharpes = np.empty(draws, dtype=float)
    for i in range(draws):
        # One sign per instrument per draw, so a placebo run is as persistent as
        # the real signal rather than being averaged away by daily coin flips.
        flips = rng.choice((-1.0, 1.0), size=(1, signals.shape[1]))
        placebo_signals = signals * flips
        nav = fast_nav_from_signals(
            prices,
            placebo_signals,
            target_vol=target_vol,
            periods_per_year=periods_per_year,
            starting_nav=starting_nav,
        )
        sharpes[i] = performance_summary(nav, periods_per_year=periods_per_year)["Sharpe"]

    percentile = float((sharpes < actual).mean() * 100.0)
    distinct = 2 ** signals.shape[1]
    frame_attrs = {
        "distinct_sign_combinations": distinct,
        "is_coarse": bool(distinct < draws),
    }
    result = pd.DataFrame(
        [
            {
                "metric": "Sharpe",
                # Named for the path that produced it: this is not the headline
                # Sharpe, which includes queueing and costs.
                "strategy_fast_path": actual,
                "placebo_mean": float(np.mean(sharpes)),
                "placebo_std": float(np.std(sharpes, ddof=1)) if draws > 1 else 0.0,
                "placebo_p05": float(np.percentile(sharpes, 5)),
                "placebo_p95": float(np.percentile(sharpes, 95)),
                "strategy_percentile": percentile,
                "draws": int(draws),
                "distinct_sign_combinations": distinct,
            }
        ]
    ).set_index("metric")
    result.attrs.update(frame_attrs)
    return result


def attribution_by_side(result: BacktestResults) -> pd.DataFrame:
    """Split PnL into what the long and short books contributed."""

    contributions = compute_pnl_contributions(
        result.prices, result.positions, result.point_values
    )
    if contributions.empty or result.positions is None:
        return pd.DataFrame()

    held = result.positions.reindex_like(contributions).shift(1).fillna(0.0)
    long_pnl = contributions.where(held > 0, 0.0).sum().sum()
    short_pnl = contributions.where(held < 0, 0.0).sum().sum()

    costs = result.costs
    total_costs = float(costs["total"].sum()) if costs is not None and "total" in costs else 0.0
    gross = float(long_pnl + short_pnl)

    return pd.DataFrame(
        [
            {"side": "long", "gross_pnl": float(long_pnl)},
            {"side": "short", "gross_pnl": float(short_pnl)},
            {"side": "total gross", "gross_pnl": gross},
            {"side": "costs", "gross_pnl": -total_costs},
            {"side": "net", "gross_pnl": gross - total_costs},
        ]
    ).set_index("side")


def attribution_by_asset(result: BacktestResults) -> pd.DataFrame:
    """Total PnL contributed by each instrument."""

    contributions = compute_pnl_contributions(
        result.prices, result.positions, result.point_values
    )
    if contributions.empty:
        return pd.DataFrame()
    totals = contributions.sum().sort_values(ascending=False)
    frame = totals.to_frame("gross_pnl")
    total = float(totals.sum())
    frame["share"] = totals / total if total else np.nan
    frame.index.name = "symbol"
    return frame


def data_span_table(prices: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-instrument history, and how often the universe was too thin."""

    from ..data.crypto import data_span
    from ..data.eligibility import thin_universe_fraction

    span = data_span(prices)
    if mask is not None and not mask.empty:
        span["eligible_bars"] = mask.sum()
        span["eligible_fraction"] = mask.mean()
        span.attrs["thin_universe_fraction"] = thin_universe_fraction(mask)
    return span


def run_falsification(
    backtester: Backtester,
    *,
    placebo_draws: int = DEFAULT_PLACEBO_DRAWS,
    seed: int | None = 0,
) -> FalsificationReport:
    """Run the whole suite against ``backtester``'s configuration."""

    cfg = backtester._base_cfg
    periods_per_year = _periods_per_year(cfg)
    backtest_cfg = cfg.get("backtest", {}) or {}
    risk_cfg = cfg.get("risk", {}) or {}
    starting_nav = float(backtest_cfg.get("starting_nav", 1_000_000.0))
    target_vol = float(risk_cfg.get("target_portfolio_vol", 0.10))

    base_result = _run(backtester)
    prices = base_result.prices
    if prices is None or prices.empty:
        raise ValueError("Backtest produced no price history to falsify against")

    signals = backtester._build_signals(prices, cfg)

    notes = [
        "Benchmarks and stress variants use the same data, dates, and costs as "
        "the strategy.",
        "The placebo path is vectorised and gross of the order queue: it tests "
        "whether the signal's direction carried information, not whether it "
        "could be executed.",
    ]
    if not risk_cfg.get("periods_per_year"):
        notes.append(
            "risk.periods_per_year is unset, so metrics annualise by 252. Set it "
            "to 365 for a 7-day calendar."
        )

    try:
        returns = base_result.nav.pct_change().dropna()
        intervals = bootstrap_confidence_intervals(
            returns,
            metrics=["sharpe", "max_drawdown"],
            n_samples=min(1_000, max(200, placebo_draws)),
            seed=seed,
            periods_per_year=periods_per_year,
        )
        ci_frame = pd.DataFrame(
            [
                {
                    "metric": name,
                    "mean": interval.mean,
                    "lower": interval.lower,
                    "upper": interval.upper,
                }
                for name, interval in intervals.items()
            ]
        ).set_index("metric")
    except Exception as exc:  # pragma: no cover
        logger.warning("Bootstrap intervals failed: %s", exc)
        ci_frame = pd.DataFrame()

    placebo = run_placebo(
        prices,
        signals,
        periods_per_year=periods_per_year,
        target_vol=target_vol,
        starting_nav=starting_nav,
        draws=placebo_draws,
        seed=seed,
    )
    if placebo.attrs.get("is_coarse"):
        combinations = placebo.attrs.get("distinct_sign_combinations")
        notes.append(
            f"The placebo flips one sign per instrument, so with "
            f"{prices.shape[1]} instruments there are only {combinations} "
            "distinct outcomes. Its percentile is correspondingly coarse and "
            "extra draws do not refine it."
        )

    cost_stress = run_cost_stress(backtester, cfg, periods_per_year=periods_per_year)
    if cost_stress.attrs.get("is_frictionless"):
        notes.append(
            "Transaction costs are configured at zero, so the cost stress rows "
            "are identical and the net figures above are gross. Configure "
            "execution costs before drawing any conclusion from them."
        )
    elif cost_stress.attrs.get("costs_barely_bite"):
        effect = cost_stress.attrs.get("cost_effect", 0.0)
        turnover = cost_stress.attrs.get("turnover", 0.0)
        notes.append(
            f"Quadrupling transaction costs changed annual return by only "
            f"{effect:.2%} despite turnover of {turnover:.0f}x, so the cost "
            "model is calibrated too small to constrain this strategy and the "
            "net figures are effectively gross. The engine's slippage is a "
            "tick count times a single universe-wide tick value, which cannot "
            "express a proportional spread across instruments whose prices "
            "differ by orders of magnitude. See docs/CRYPTO_DATA.md."
        )

    return FalsificationReport(
        variants=run_benchmark_battery(
            backtester,
            prices=prices,
            base_result=base_result,
            periods_per_year=periods_per_year,
            starting_nav=starting_nav,
            target_vol=target_vol,
        ),
        vol_scaling_2x2=run_vol_scaling_2x2(backtester, periods_per_year=periods_per_year),
        signal_delay=run_signal_delay(backtester, periods_per_year=periods_per_year),
        cost_stress=cost_stress,
        vol_lookback=run_vol_lookback(backtester, periods_per_year=periods_per_year),
        leave_one_out=run_leave_one_out(backtester, prices, periods_per_year=periods_per_year),
        placebo=placebo,
        attribution=attribution_by_asset(base_result),
        side_attribution=attribution_by_side(base_result),
        data_span=data_span_table(prices, backtester._eligibility_mask),
        confidence_intervals=ci_frame,
        notes=notes,
    )
