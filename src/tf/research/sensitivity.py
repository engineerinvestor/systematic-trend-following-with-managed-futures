"""Sensitivity analysis utilities for parameter sweeps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from ..eval.metrics import TRADING_DAYS, performance_summary
from ..engine.backtester import Backtester


def _periods_per_year(backtester: "Backtester") -> int:
    """Annualisation basis the backtester's config sized positions on."""

    risk_cfg = backtester._base_cfg.get("risk", {}) or {}
    return int(risk_cfg.get("periods_per_year", TRADING_DAYS))


def _to_overrides(parameter: str, value) -> dict:
    parts = parameter.split(".")
    overrides: dict = {}
    current = overrides
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    return overrides


def _merge_overrides(base: Mapping | None, extra: Mapping | None) -> dict:
    if not base and not extra:
        return {}

    def _merge(lhs: MutableMapping, rhs: Mapping) -> MutableMapping:
        for key, value in rhs.items():
            if isinstance(value, Mapping) and isinstance(lhs.get(key), MutableMapping):
                lhs[key] = _merge(lhs[key], value)  # type: ignore[index]
            elif isinstance(value, Mapping):
                lhs[key] = _merge({}, value)
            else:
                lhs[key] = value
        return lhs

    merged: MutableMapping = {}
    if base:
        merged = _merge({}, base)
    if extra:
        merged = _merge(merged, extra)
    return dict(merged)


def compute_metric_sensitivity(
    backtester: Backtester,
    parameter: str,
    values: Iterable,
    *,
    metric: str = "Sharpe",
    base_overrides: Mapping[str, object] | None = None,
    seed: int | None = None,
    value_adapter: Callable[[object], object] | None = None,
) -> pd.DataFrame:
    """Return a dataframe of the requested metric against parameter values."""

    records = []
    for value in values:
        override_value = value_adapter(value) if value_adapter else value
        overrides = _to_overrides(parameter, override_value)
        combined = _merge_overrides(base_overrides, overrides)
        res = backtester.run(parameter_overrides=combined, seed=seed)
        summary = performance_summary(
            res.nav,
            trades=res.trades,
            periods_per_year=_periods_per_year(backtester),
        )
        records.append({"parameter": value, metric: summary.get(metric, float("nan"))})
    frame = pd.DataFrame(records)
    return frame.set_index("parameter")


def lookback_sensitivity(
    backtester: Backtester,
    lookbacks: Sequence[int],
    *,
    metric: str = "Sharpe",
    seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate performance metric for a range of lookback values."""

    values = [{"signals": {"momentum": {"lookbacks": [lb]}}} for lb in lookbacks]
    records = []
    for lb, overrides in zip(lookbacks, values, strict=True):
        res = backtester.run(parameter_overrides=overrides, seed=seed)
        summary = performance_summary(res.nav, trades=res.trades, periods_per_year=_periods_per_year(backtester))
        records.append({"lookback": lb, metric: summary.get(metric, float("nan"))})
    return pd.DataFrame(records).set_index("lookback")


def vol_target_sensitivity(
    backtester: Backtester,
    targets: Sequence[float],
    *,
    metric: str = "Sharpe",
    seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate performance metric for a range of volatility targets."""

    records = []
    for target in targets:
        overrides = {"risk": {"target_portfolio_vol": float(target)}}
        res = backtester.run(parameter_overrides=overrides, seed=seed)
        summary = performance_summary(res.nav, trades=res.trades, periods_per_year=_periods_per_year(backtester))
        records.append({"vol_target": target, metric: summary.get(metric, float("nan"))})
    return pd.DataFrame(records).set_index("vol_target")
