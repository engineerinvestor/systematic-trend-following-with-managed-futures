"""Portfolio sizing utilities."""

from __future__ import annotations

from typing import Mapping

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..risk.vol import ewma_vol, rolling_volatility


def _min_step(contract_rounding: float | Mapping[str, float], symbols) -> float:
    if isinstance(contract_rounding, Mapping):
        steps = [float(contract_rounding.get(sym, 1.0)) for sym in symbols]
        return min(steps) if steps else 1.0
    return float(contract_rounding)


def _resolve_contract_step(contract_rounding: float | Mapping[str, float], symbol: str) -> float:
    if isinstance(contract_rounding, Mapping):
        step = float(contract_rounding.get(symbol, 1.0))
    else:
        step = float(contract_rounding)
    return 1.0 if step <= 0 else step


def _round_series_to_contracts(
    series: pd.Series, contract_rounding: float | Mapping[str, float]
) -> pd.Series:
    rounded = pd.Series(index=series.index, dtype=float)
    for sym, value in series.items():
        step = _resolve_contract_step(contract_rounding, sym)
        if not np.isfinite(value):
            value = 0.0
        rounded[sym] = np.round(value / step) * step
    return rounded


def _apply_max_asset_weight(
    weights: pd.DataFrame, max_asset_weight: float | None
) -> pd.DataFrame:
    """Cap each instrument's share of that day's gross exposure.

    Water-filling on shares with gross preserved: names over the cap are fixed
    at it and the excess is redistributed pro-rata over the uncapped names,
    repeating until every share respects the cap. Gross exposure is unchanged,
    because this constraint is about concentration, not de-risking. A single
    clip cannot do this: clipping shrinks gross, which shrinks the cap, so a
    [0.8, 0.1, 0.1] book clipped once at 0.5 ends at 71% of gross.

    Rows where the cap is infeasible (fewer than ``1 / cap`` active names,
    e.g. one live instrument against a 50% cap) are left unchanged: the only
    mathematical solution there is zero, and silently liquidating the book is
    worse than a violated concentration preference.
    """

    if max_asset_weight is None or max_asset_weight <= 0 or weights.empty:
        return weights
    cap = float(max_asset_weight)
    if cap >= 1.0:
        return weights

    values = weights.to_numpy(dtype=float, copy=True)
    for row in range(values.shape[0]):
        absolute = np.abs(values[row])
        gross = absolute.sum()
        if gross <= 0:
            continue
        active = absolute > 0
        if active.sum() * cap < 1.0 - 1e-12:
            continue  # infeasible: every allocation of this row violates the cap

        shares = absolute / gross
        capped = np.zeros_like(shares, dtype=bool)
        for _ in range(values.shape[1]):
            over = (shares > cap + 1e-12) & ~capped
            if not over.any():
                break
            capped |= over
            shares[capped] = cap
            remainder = 1.0 - cap * capped.sum()
            free = ~capped & active
            free_total = shares[free].sum()
            if free_total <= 0:
                break
            shares[free] *= remainder / free_total
        values[row] = np.sign(values[row]) * shares * gross

    return pd.DataFrame(values, index=weights.index, columns=weights.columns)


def _scale_to_gross_limit(weights: pd.DataFrame, gross_limit: float | None) -> pd.DataFrame:
    if gross_limit is None or gross_limit <= 0:
        return weights
    gross = weights.abs().sum(axis=1)
    scale = pd.Series(1.0, index=weights.index)
    mask = gross > gross_limit
    if mask.any():
        scale.loc[mask] = gross_limit / gross.loc[mask]
    return weights.mul(scale, axis=0)


def _apply_sector_caps(
    budgets: pd.DataFrame,
    sector_map: Mapping[str, str] | None,
    sector_caps: Mapping[str, float] | None,
) -> pd.DataFrame:
    if not sector_map or not sector_caps:
        return budgets

    capped = budgets.copy()
    symbol_to_sector = {sym: sector_map[sym] for sym in capped.columns if sym in sector_map}
    for sector, cap in sector_caps.items():
        sector_symbols = [sym for sym, sec in symbol_to_sector.items() if sec == sector]
        if not sector_symbols:
            continue
        total = capped[sector_symbols].sum(axis=1)
        mask = total > cap
        if mask.any():
            scale = cap / total
            scale = scale.where(mask, 1.0)
            capped.loc[mask, sector_symbols] = capped.loc[mask, sector_symbols].mul(
                scale.loc[mask], axis=0
            )
    return capped


def _compute_volatility(
    prices: pd.DataFrame,
    *,
    vol_model: str,
    vol_lookback: int,
    ewma_lambda: float,
    min_vol_periods: int,
    volatility: pd.DataFrame | None,
    periods_per_year: int = 252,
    ewma_center_of_mass: float | None = None,
    vol_warmup: str = "mask",
) -> pd.DataFrame:
    if volatility is not None:
        vol = volatility.copy()
    else:
        # NaN returns (before an instrument's first price) must stay NaN: a
        # fillna(0.0) here fabricated a pre-listing run of zero returns that
        # satisfied the EWMA's min_periods, releasing a near-zero volatility
        # estimate on the first real trading day and defeating the warm-up mask.
        returns = prices.pct_change()
        if vol_model == "ewma":
            vol = ewma_vol(
                returns,
                lam=ewma_lambda,
                min_periods=min_vol_periods,
                periods_per_year=periods_per_year,
                center_of_mass=ewma_center_of_mass,
            )
        elif vol_model == "rolling":
            vol = rolling_volatility(
                returns,
                window=vol_lookback,
                min_periods=min_vol_periods,
                periods_per_year=periods_per_year,
            )
        else:
            raise ValueError(f"Unknown volatility model: {vol_model}")
    vol = vol.reindex(index=prices.index, columns=prices.columns)
    vol = vol.replace(0.0, np.nan).ffill()
    if vol_warmup == "bfill":
        # Pre-0.10 behaviour, retained only to reproduce older results: it fills
        # the warm-up window with the first volatility estimate computed from
        # data that had not yet arrived, which is lookahead bias.
        vol = vol.bfill()
    elif vol_warmup != "mask":
        raise ValueError(
            f"Unknown vol_warmup: {vol_warmup!r}. Expected 'mask' or 'bfill'."
        )
    # Under "mask" the warm-up stays NaN, which sizing turns into a zero weight,
    # so an instrument holds no position until its volatility estimate is warm.
    return vol.fillna(0.0)


def _risk_budget(
    signals: pd.DataFrame,
    allocator: str,
) -> pd.DataFrame:
    abs_signals = signals.abs()
    if allocator == "erc":
        active = (abs_signals > 0).astype(float)
        row_sums = active.sum(axis=1).replace(0.0, np.nan)
        return active.div(row_sums, axis=0).fillna(0.0)
    if allocator != "proportional":
        raise ValueError(f"Unknown risk allocator: {allocator}")
    row_sums = abs_signals.sum(axis=1).replace(0.0, np.nan)
    return abs_signals.div(row_sums, axis=0).fillna(0.0)


def _apply_rebalance_threshold(
    desired_contracts: pd.DataFrame,
    contract_rounding: float | Mapping[str, float],
    threshold: float,
    prev_positions: pd.Series | Mapping[str, float] | None,
    threshold_mode: str = "contracts",
) -> pd.DataFrame:
    final = pd.DataFrame(0.0, index=desired_contracts.index, columns=desired_contracts.columns)
    if prev_positions is None:
        prev = pd.Series(0.0, index=desired_contracts.columns, dtype=float)
    else:
        if not isinstance(prev_positions, pd.Series):
            prev = pd.Series(prev_positions, dtype=float)
        else:
            prev = prev_positions.astype(float)
        prev = prev.reindex(desired_contracts.columns).fillna(0.0)

    threshold = max(float(threshold), 0.0)
    for ts in desired_contracts.index:
        target = desired_contracts.loc[ts].fillna(0.0).copy()
        delta = target - prev
        if threshold > 0:
            if threshold_mode == "fraction":
                # Threshold as a fraction of the position, so its meaning does
                # not swing five orders of magnitude across a universe whose
                # contract sizes do (0.05 contracts is 0.3% of NAV in BTC and
                # three cents in XRP).
                scale = pd.concat([target.abs(), prev.abs()], axis=1).max(axis=1)
                small = delta.abs() < threshold * scale
            else:
                small = delta.abs() < threshold
            target.loc[small] = prev.loc[small]
        rounded = _round_series_to_contracts(target, contract_rounding)
        final.loc[ts] = rounded
        prev = rounded
    return final


def volatility_target_positions(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    point_values: Mapping[str, float],
    *,
    capital: float = 1_000_000.0,
    target_portfolio_vol: float = 0.15,
    gross_exposure_limit: float | None = 3.0,
    sector_map: Mapping[str, str] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    contract_rounding: float | Mapping[str, float] = 1.0,
    rebalance_threshold: float = 0.25,
    rebalance_threshold_mode: str = "contracts",
    prev_positions: pd.Series | Mapping[str, float] | None = None,
    vol_model: str = "ewma",
    vol_lookback: int = 63,
    ewma_lambda: float = 0.94,
    min_vol_periods: int = 20,
    risk_allocator: str = "proportional",
    max_position_weight: float | None = None,
    max_notional_weight: float | None = None,
    max_asset_weight: float | None = None,
    volatility: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    ewma_center_of_mass: float | None = None,
    vol_warmup: str = "mask",
) -> pd.DataFrame:
    """Size positions to a portfolio volatility target.

    ``max_position_weight`` caps an instrument's share of the RISK BUDGET
    (pre-vol-division), which is what the config names it maps
    (``max_instrument_vol_weight``) describe; ``max_notional_weight`` caps the
    post-division notional weight, the behaviour this parameter previously
    had. Standalone vol contributions are summed as if correlations were 1.0;
    there is no covariance matrix, so realised portfolio volatility lands
    below the target by roughly ``sqrt((1 + (n-1) * rho) / n)`` for average
    pairwise correlation ``rho``. Treat ``target_portfolio_vol`` as an
    upper-bound calibration, not a realised-vol promise.
    """

    if prices.empty:
        raise ValueError("Price history is empty")
    if set(prices.columns) != set(signals.columns):
        signals = signals.reindex(columns=prices.columns, fill_value=0.0)
    signals = signals.reindex(index=prices.index).fillna(0.0)
    prices = prices.sort_index()

    vol = _compute_volatility(
        prices,
        vol_model=vol_model,
        vol_lookback=vol_lookback,
        ewma_lambda=ewma_lambda,
        min_vol_periods=min_vol_periods,
        volatility=volatility,
        periods_per_year=periods_per_year,
        ewma_center_of_mass=ewma_center_of_mass,
        vol_warmup=vol_warmup,
    ).replace(0.0, np.nan)

    risk_budget = _risk_budget(signals, allocator=risk_allocator)
    risk_budget = _apply_sector_caps(risk_budget, sector_map, sector_caps)

    if max_position_weight is not None and max_position_weight > 0:
        # A risk-share cap. Applying this value as a notional clip (the old
        # behaviour) with the shipped 0.04 bound every instrument roughly 3x
        # below its uncapped weight and held realised vol near 1% against a
        # 15% target.
        risk_budget = risk_budget.clip(upper=float(max_position_weight))

    risk_target = risk_budget * float(target_portfolio_vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = risk_target.div(vol)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights * np.sign(signals)

    if max_notional_weight is not None and max_notional_weight > 0:
        weights = weights.clip(upper=max_notional_weight, lower=-max_notional_weight)

    weights = _apply_max_asset_weight(weights, max_asset_weight)
    weights = _scale_to_gross_limit(weights, gross_exposure_limit)

    capital = float(capital)
    notional = weights * capital
    contracts = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for sym in prices.columns:
        pv = float(point_values.get(sym, 1.0))
        denom = (prices[sym] * pv).replace(0.0, np.nan)
        contracts[sym] = notional[sym].div(denom)
    contracts = contracts.fillna(0.0)

    if (
        rebalance_threshold_mode == "contracts"
        and rebalance_threshold > 0
        and 2 * rebalance_threshold <= _min_step(contract_rounding, prices.columns)
    ):
        logger.warning(
            "rebalance_threshold %.4g is inert: any change it would suppress "
            "also rounds away at contract_step. Raise it or set "
            "rebalance_threshold_mode: fraction.",
            rebalance_threshold,
        )

    final_positions = _apply_rebalance_threshold(
        contracts,
        contract_rounding=contract_rounding,
        threshold=rebalance_threshold,
        prev_positions=prev_positions,
        threshold_mode=rebalance_threshold_mode,
    )
    return final_positions
