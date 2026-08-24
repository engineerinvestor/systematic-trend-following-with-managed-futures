"""Reference trend-following strategies for cryptocurrency.

Each preset implements a methodology described in public research. None
reproduces, or claims to reproduce, any proprietary strategy or product; the
names describe the method, not any manager or fund. See ``CRYPTO_SPEC.md`` for
the evidence table behind each one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..signals.breakout import donchian_breakout
from ..signals.momentum import timeseries_momentum
from ..signals.moving_average import moving_average_crossover, price_minus_ma
from .horizons import REPLICATION_HORIZONS, TSMOM_HORIZONS, resolve_horizons

#: Direction modes. ``long_short`` takes the classical managed-futures position
#: on both sides; ``long_flat`` holds cash instead of shorting, which is how most
#: listed crypto trend products are built.
DIRECTION_MODES = ("long_short", "long_flat")

#: EWMA volatility centre of mass used by the TSMOM presets (MOP use 60 days).
TSMOM_VOL_COM = 60.0

#: EWMA volatility centre of mass used by the replication ensemble.
REPLICATION_VOL_COM = 40.0

#: Per-instrument volatility normalisation from the TSMOM literature. This is a
#: signal-scaling convention, not a recommendation to run a portfolio at 40
#: percent volatility; portfolio-level scaling is applied on top.
INSTRUMENT_VOL_TARGET = 0.40


def apply_direction(signals: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Apply a direction mode to a signal frame."""

    if direction not in DIRECTION_MODES:
        raise ValueError(
            f"Unknown direction: {direction!r}. Expected one of {DIRECTION_MODES}."
        )
    if direction == "long_flat":
        return signals.clip(lower=0.0)
    return signals


@dataclass(frozen=True)
class Preset:
    """A named signal construction plus the risk defaults it expects."""

    name: str
    description: str
    builder: Callable[..., pd.DataFrame]
    horizons: Sequence[str | int]
    vol_center_of_mass: float
    reference: str
    options: Mapping[str, object] = field(default_factory=dict)

    def build(
        self,
        prices: pd.DataFrame,
        *,
        freq: str = "daily",
        direction: str = "long_short",
        horizons: Sequence[str | int] | None = None,
        lag: int = 1,
        **overrides: object,
    ) -> pd.DataFrame:
        """Return the preset's signal frame for ``prices``."""

        resolved = resolve_horizons(horizons or self.horizons, freq)
        options = dict(self.options)
        options.update(overrides)
        signals = self.builder(prices, horizons=resolved, lag=lag, **options)
        return apply_direction(signals, direction)


def _tsmom_builder(
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int],
    lag: int = 1,
    skip_last_n: int = 0,
    transform: str = "sign",
    weighting: str = "equal",
) -> pd.DataFrame:
    if skip_last_n:
        raise ValueError(
            "Time-series momentum uses no skip period. skip_last_n must be 0; "
            f"got {skip_last_n}. The skip-month convention belongs to "
            "cross-sectional momentum."
        )
    return timeseries_momentum(
        prices,
        lookbacks=tuple(horizons),
        skip_last_n=0,
        transform=transform,
        weighting=weighting,
        lag=lag,
    )


def _multisystem_builder(
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int],
    lag: int = 1,
    systems: Sequence[str] = ("total_return", "pmac", "dmac", "breakout"),
) -> pd.DataFrame:
    """Average four trend systems across a ladder of horizons.

    Weights are equal across systems and horizons. Fitting weights would need a
    replication target and a long joint history; crypto has neither, so
    estimated weights would be noise. Replication research also reports that the
    choice among these system families made little difference, which argues for
    diversifying simple systems rather than tuning one.
    """

    unknown = [s for s in systems if s not in _SYSTEM_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown trend systems: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_SYSTEM_BUILDERS))}"
        )

    frames: list[pd.DataFrame] = []
    for system in systems:
        builder = _SYSTEM_BUILDERS[system]
        for horizon in horizons:
            frames.append(builder(prices, horizon, lag))

    if not frames:
        raise ValueError("At least one system and horizon are required")

    total = frames[0].copy()
    for frame in frames[1:]:
        total = total.add(frame, fill_value=0.0)
    return total / float(len(frames))


def _system_total_return(prices: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    return timeseries_momentum(
        prices, lookbacks=(horizon,), skip_last_n=0, transform="sign", lag=lag
    )


def _system_pmac(prices: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    return price_minus_ma(prices, window=horizon, transform="sign", lag=lag)


def _system_dmac(prices: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    fast = max(int(horizon / 4), 2)
    if fast >= horizon:
        # Horizons of 2 or 3 bars cannot support a fast/slow split.
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    raw = moving_average_crossover(prices, fast=fast, slow=horizon, lag=lag)
    return raw.apply(np.sign)


def _system_breakout(prices: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    return donchian_breakout(prices, window=horizon, lag=lag)


_SYSTEM_BUILDERS: Mapping[str, Callable[[pd.DataFrame, int, int], pd.DataFrame]] = {
    "total_return": _system_total_return,
    "pmac": _system_pmac,
    "dmac": _system_dmac,
    "breakout": _system_breakout,
}


def _long_flat_ma_builder(
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int],
    lag: int = 1,
) -> pd.DataFrame:
    """Price versus a single long moving average, the shape of listed products."""

    window = int(horizons[0])
    return price_minus_ma(prices, window=window, transform="sign", lag=lag)


PRESETS: Mapping[str, Preset] = {
    "mop2012_tsmom": Preset(
        name="mop2012_tsmom",
        description=(
            "Single-horizon time-series momentum: the sign of the trailing "
            "12-month return, volatility scaled."
        ),
        builder=_tsmom_builder,
        horizons=("365D",),
        vol_center_of_mass=TSMOM_VOL_COM,
        reference="Moskowitz, Ooi, Pedersen (2012), Time Series Momentum, JFE 104(2)",
    ),
    "tsmom_1_3_12": Preset(
        name="tsmom_1_3_12",
        description=(
            "Equal-weighted 1, 3, and 12-month time-series momentum, the "
            "ensemble used to characterise managed-futures returns."
        ),
        builder=_tsmom_builder,
        horizons=TSMOM_HORIZONS,
        vol_center_of_mass=TSMOM_VOL_COM,
        reference="Hurst, Ooi, Pedersen (2013), Demystifying Managed Futures, JOIM 11(3)",
    ),
    "bottom_up_multisystem": Preset(
        name="bottom_up_multisystem",
        description=(
            "Equal-weighted ensemble of total-return momentum, price minus "
            "moving average, dual moving average, and Donchian breakout, across "
            "thirteen horizons from one week to one year."
        ),
        builder=_multisystem_builder,
        horizons=REPLICATION_HORIZONS,
        vol_center_of_mass=REPLICATION_VOL_COM,
        reference="ReSolve Asset Management, How to Replicate Trend Following Managed Futures",
    ),
    "btc_long_flat": Preset(
        name="btc_long_flat",
        description=(
            "Price versus its 200-day moving average, long or flat. The shape of "
            "most listed crypto trend products, included as a benchmark."
        ),
        builder=_long_flat_ma_builder,
        horizons=(200,),
        vol_center_of_mass=TSMOM_VOL_COM,
        reference="Global X BTRN and comparable listed long/flat crypto trend products",
    ),
}


def get_preset(name: str) -> Preset:
    """Return a preset by name."""

    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown preset: {name!r}. Available: {', '.join(sorted(PRESETS))}"
        ) from exc


def build_signals(
    prices: pd.DataFrame,
    preset: str,
    *,
    freq: str = "daily",
    direction: str = "long_short",
    horizons: Sequence[str | int] | None = None,
    lag: int = 1,
    **overrides: object,
) -> pd.DataFrame:
    """Build the signal frame for a named preset."""

    spec = get_preset(preset)
    if spec.name == "btc_long_flat" and direction != "long_flat":
        raise ValueError(
            "btc_long_flat is a long/flat benchmark; use tsmom presets for "
            "long/short comparisons."
        )
    return spec.build(
        prices,
        freq=freq,
        direction=direction,
        horizons=horizons,
        lag=lag,
        **overrides,
    )
