"""Core daily backtest engine and helper utilities."""

from __future__ import annotations

import copy
import logging
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..portfolio.sizing import volatility_target_positions
from ..eval.metrics import TRADING_DAYS
from ..signals.momentum import timeseries_momentum

logger = logging.getLogger(__name__)
from .execution import (
    Order,
    RollEngine,
    participation_slippage,
    resolve_adv_frame,
)


#: Keys each config block is known to read. Anything else is warned about:
#: metadata rejects typos loudly, and a silently ignored risk knob is worse.
_KNOWN_CONFIG_KEYS = {
    "data": {
        "calendar", "prefer", "prefer_prices", "price_preference", "strict",
        "allow_partial_history", "root_dir",
    },
    "signals": {
        "preset", "horizons", "direction", "lag", "momentum", "disable_trend",
        "systems", "skip_last_n", "transform", "weighting", "symbol",
    },
    "risk": {
        "vol_model", "vol_lookback", "ewma_lambda", "ewma_center_of_mass",
        "min_vol_periods", "periods_per_year", "target_portfolio_vol",
        "gross_exposure_limit", "rebalance_threshold", "rebalance_threshold_mode",
        "risk_allocator", "max_instrument_weight", "max_instrument_vol_weight",
        "max_notional_weight", "max_asset_weight", "sector_caps",
        "disable_vol_scaling", "disable_vol_scaling_gross", "vol_warmup",
    },
    "execution": {
        "order_type", "adv_limit_pct", "default_adv_contracts",
        "average_daily_volume", "adv_contracts", "roll_schedule",
        "commission_per_contract", "impact", "tick_value", "min_slippage_ticks",
        "cost_multiplier", "spread_bps",
    },
    "backtest": {
        "start", "end", "starting_nav", "seed", "results_dir", "rebalance",
        "check_reproducibility", "compound_capital",
    },
}


def _warn_unknown_keys(cfg: Mapping[str, object]) -> None:
    for block, known in _KNOWN_CONFIG_KEYS.items():
        section = cfg.get(block)
        if not isinstance(section, Mapping):
            continue
        unknown = sorted(str(key) for key in section if key not in known)
        if unknown:
            logger.warning(
                "Config block '%s' has keys nothing reads: %s. A typo here "
                "silently falls back to defaults.",
                block,
                ", ".join(unknown),
            )


def _is_daily_calendar(name: object) -> bool:
    """Return whether a config calendar name denotes a 7-day calendar."""

    from ..data.calendar import TradingCalendar

    return TradingCalendar.from_name(None if name is None else str(name)).freq == "daily"


def _deep_update(base: dict, overrides: Mapping[str, object]) -> dict:
    """Recursively merge ``overrides`` into ``base`` without mutating inputs."""

    updated = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(updated.get(key), Mapping):
            updated[key] = _deep_update(updated[key], value)  # type: ignore[assignment]
        else:
            updated[key] = copy.deepcopy(value)
    return updated


def _assign_nested(container: dict, dotted_key: str, value: object) -> None:
    """Assign ``value`` into ``container`` following ``dot.separated`` keys."""

    parts = dotted_key.split(".")
    current = container
    for part in parts[:-1]:
        current = current.setdefault(part, {})  # type: ignore[assignment]
    current[parts[-1]] = value


def expand_parameter_grid(grid: Mapping[str, Sequence[object]]) -> list[dict]:
    """Expand a mapping of dot-separated keys to lists of values into overrides."""

    if not grid:
        return []

    keys = list(grid.keys())
    values_product = list(product(*(grid[key] for key in keys)))
    scenarios: list[dict] = []
    for combo in values_product:
        overrides: dict[str, object] = {}
        for key, value in zip(keys, combo, strict=True):
            _assign_nested(overrides, key, value)
        scenarios.append(overrides)
    return scenarios


def _hash_config(cfg: Mapping[str, object]) -> str:
    """Return a deterministic hash for the provided configuration mapping."""

    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BacktestResults:
    """Container for primary backtest artefacts."""

    nav: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    config_hash: str
    orders: pd.DataFrame | None = None
    costs: pd.DataFrame | None = None
    pnl: pd.Series | None = None
    cash: pd.Series | None = None
    ledger: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    point_values: dict[str, float] | None = None
    sector_map: dict[str, str] | None = None
    #: Point-in-time eligibility mask this run traded under, or None.
    eligibility_mask: pd.DataFrame | None = None


class Backtester:
    """Daily event loop backtester supporting parameter sweeps and splits."""

    def __init__(
        self,
        prices: pd.DataFrame,
        universe_meta,
        cfg: dict,
        *,
        volumes: pd.DataFrame | None = None,
    ):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("Backtester prices must be indexed by a DatetimeIndex")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("Backtester price index must be sorted ascending")
        self.prices = prices
        #: Optional per-instrument traded volume in USD, aligned like prices.
        #: Feeds the eligibility rule's min_adv_usd filter.
        self._volumes = volumes
        self.universe_meta = universe_meta  # list of dicts
        self._base_cfg = copy.deepcopy(cfg)
        self.results_dir = Path(cfg["backtest"].get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._point_values = {
            u["symbol"]: float(u.get("point_value", 1.0)) for u in self.universe_meta
        }
        self._sector_map = {u["symbol"]: u.get("sector", "Unknown") for u in self.universe_meta}
        #: Last run's eligibility mask, for report tooling. Written per-run, so
        #: sharing one Backtester across threads is not supported; use
        #: BacktestResults.eligibility_mask for anything that must be race-free.
        self._eligibility_mask: pd.DataFrame | None = None
        self._contract_steps = {
            u["symbol"]: float(u.get("contract_step", 1.0)) for u in self.universe_meta
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        parameter_overrides: Mapping[str, object] | None = None,
        walk_forward_split: Mapping[str, object] | None = None,
        seed: int | None = None,
        check_reproducibility: bool | None = None,
    ) -> BacktestResults:
        """Execute a single backtest run.

        Parameters
        ----------
        parameter_overrides:
            Nested mapping overriding configuration values.  This is the
            primary entry point for grid-search sweeps.
        walk_forward_split:
            Optional mapping containing ``start``/``end`` boundaries that will
            override the base configuration window.
        seed:
            Optional random seed overriding the configuration value.
        check_reproducibility:
            When ``True`` the engine replays the run with the same seed and
            verifies that all primary artefacts match.
        """

        cfg = self._prepare_run_config(
            overrides=parameter_overrides, split=walk_forward_split, seed=seed
        )

        if check_reproducibility is None:
            check_reproducibility = bool(cfg.get("backtest", {}).get("check_reproducibility", False))

        config_hash = _hash_config(cfg)
        result = self._simulate_once(cfg, config_hash=config_hash)

        if check_reproducibility:
            repeat = self._simulate_once(cfg, config_hash=config_hash)
            if not self._results_equal(result, repeat):
                raise RuntimeError("Backtest run is not reproducible under the provided seed")

        return result

    def run_parameter_grid(
        self,
        parameter_grid: Mapping[str, Sequence[object]] | None,
        *,
        base_overrides: Mapping[str, object] | None = None,
        walk_forward_splits: Sequence[Mapping[str, object]] | None = None,
        seeds: Sequence[int] | int | None = None,
        check_reproducibility: bool | None = None,
        n_jobs: int | None = None,
        prefer: str | None = None,
    ) -> list[BacktestResults]:
        """Iterate through a parameter grid and optional walk-forward splits."""

        overrides_list = expand_parameter_grid(parameter_grid or {})
        if not overrides_list:
            overrides_list = [None]

        if walk_forward_splits:
            splits = list(walk_forward_splits)
        else:
            splits = [None]

        if seeds is None:
            seeds_list: Sequence[int | None] = [None]
        elif isinstance(seeds, int):
            seeds_list = [seeds]
        else:
            seeds_list = list(seeds)

        tasks: list[tuple[Mapping[str, object] | None, Mapping[str, object] | None, int | None]] = []
        for split in splits:
            for overrides in overrides_list:
                if base_overrides and overrides:
                    merged = _deep_update(base_overrides, overrides)
                elif base_overrides:
                    merged = copy.deepcopy(base_overrides)
                else:
                    merged = overrides
                for run_seed in seeds_list:
                    tasks.append((merged, split, run_seed))

        if n_jobs is None or n_jobs <= 1 or len(tasks) <= 1:
            results = [
                self.run(
                    parameter_overrides=overrides,
                    walk_forward_split=split,
                    seed=run_seed,
                    check_reproducibility=check_reproducibility,
                )
                for overrides, split, run_seed in tasks
            ]
            return results

        try:
            from joblib import Parallel, delayed
        except ImportError:  # pragma: no cover - optional dependency fallback
            results = [
                self.run(
                    parameter_overrides=overrides,
                    walk_forward_split=split,
                    seed=run_seed,
                    check_reproducibility=check_reproducibility,
                )
                for overrides, split, run_seed in tasks
            ]
            return results

        prefer = prefer or "processes"
        parallel = Parallel(n_jobs=n_jobs, prefer=prefer)
        return parallel(
            delayed(self.run)(
                parameter_overrides=overrides,
                walk_forward_split=split,
                seed=run_seed,
                check_reproducibility=check_reproducibility,
            )
            for overrides, split, run_seed in tasks
        )

    def run_walk_forward(
        self,
        *,
        insample: int,
        oos: int,
        step: int | None = None,
        anchored: bool = True,
        parameter_overrides: Mapping[str, object] | None = None,
        seed: int | None = None,
        check_reproducibility: bool | None = None,
        n_jobs: int | None = None,
        prefer: str | None = None,
    ) -> list["WalkForwardResult"]:
        """Run anchored or rolling walk-forward evaluation across the price history."""

        from ..research.walkforward import (
            WalkForwardResult,
            generate_walk_forward_windows,
            windows_to_splits,
        )

        windows = generate_walk_forward_windows(
            self.prices.index,
            insample=insample,
            oos=oos,
            step=step,
            anchored=anchored,
        )
        if not windows:
            return []

        splits = windows_to_splits(windows)
        seeds: Sequence[int] | int | None
        seeds = [seed] if seed is not None else None

        results = self.run_parameter_grid(
            parameter_grid=None,
            base_overrides=parameter_overrides,
            walk_forward_splits=splits,
            seeds=seeds,
            check_reproducibility=check_reproducibility,
            n_jobs=n_jobs,
            prefer=prefer,
        )

        if len(results) != len(windows):
            raise RuntimeError("Unexpected mismatch between walk-forward windows and results")

        return [
            WalkForwardResult(
                window=window,
                backtest=result,
                periods_per_year=int(
                    (self._base_cfg.get("risk", {}) or {}).get(
                        "periods_per_year", TRADING_DAYS
                    )
                ),
            )
            for window, result in zip(windows, results, strict=True)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_run_config(
        self,
        *,
        overrides: Mapping[str, object] | None,
        split: Mapping[str, object] | None,
        seed: int | None,
    ) -> dict:
        cfg = copy.deepcopy(self._base_cfg)
        if overrides:
            cfg = _deep_update(cfg, overrides)
        if split:
            start = split.get("start")
            end = split.get("end")
            backtest_cfg = cfg.setdefault("backtest", {})
            if start is not None:
                backtest_cfg["start"] = str(pd.Timestamp(start).date())
            if end is not None:
                backtest_cfg["end"] = str(pd.Timestamp(end).date())
        if seed is not None:
            cfg.setdefault("backtest", {})["seed"] = int(seed)
        return cfg

    def _results_equal(self, lhs: BacktestResults, rhs: BacktestResults) -> bool:
        return (
            lhs.nav.equals(rhs.nav)
            and lhs.positions.equals(rhs.positions)
            and lhs.trades.equals(rhs.trades)
            and (lhs.orders.equals(rhs.orders) if lhs.orders is not None else rhs.orders is None)
            and (lhs.costs.equals(rhs.costs) if lhs.costs is not None else rhs.costs is None)
            and (lhs.pnl.equals(rhs.pnl) if lhs.pnl is not None else rhs.pnl is None)
            and (lhs.cash.equals(rhs.cash) if lhs.cash is not None else rhs.cash is None)
            and (lhs.prices.equals(rhs.prices) if lhs.prices is not None else rhs.prices is None)
            and (lhs.ledger.equals(rhs.ledger) if lhs.ledger is not None else rhs.ledger is None)
        )

    @staticmethod
    def _preset_vol_com(cfg: Mapping[str, object]) -> float | None:
        """Volatility centre of mass the config's preset was published with.

        Used only when ``risk.ewma_center_of_mass`` is not set, so a preset run
        matches its literature convention (60 for the TSMOM presets, 40 for the
        replication ensemble) instead of silently taking the RiskMetrics 0.94
        default.
        """

        sig_cfg = cfg.get("signals", {}) or {}
        preset = sig_cfg.get("preset")
        if not preset:
            return None
        from ..crypto.presets import get_preset

        try:
            return float(get_preset(str(preset)).vol_center_of_mass)
        except KeyError:
            return None

    def _build_signals(self, prices: pd.DataFrame, cfg: Mapping[str, object]) -> pd.DataFrame:
        """Return the signal frame for ``prices``.

        Without a ``signals.preset`` key this is the historical multi-horizon
        momentum call, so existing configs behave exactly as before.
        """

        sig_cfg = cfg.get("signals", {}) or {}
        if sig_cfg.get("disable_trend"):
            # Used by the falsification suite to isolate how much of a result
            # comes from the trend signal rather than from volatility scaling.
            return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)

        preset = sig_cfg.get("preset")
        if preset:
            from ..crypto.presets import build_signals, get_preset

            data_cfg = cfg.get("data", {}) or {}
            freq = "daily" if _is_daily_calendar(data_cfg.get("calendar")) else "weekday"
            reserved = {"preset", "horizons", "direction", "lag", "momentum", "disable_trend"}
            accepted = get_preset(str(preset)).accepted_options()
            options = {}
            for key, value in sig_cfg.items():
                if key in reserved:
                    continue
                if key in accepted:
                    options[key] = value
                else:
                    # A leftover key from another preset (a benchmark override
                    # switching presets inherits the base config's block) must
                    # not detonate deep inside the builder, but it must not
                    # vanish silently either.
                    logger.warning(
                        "Ignoring signals.%s: preset %r does not accept it",
                        key,
                        preset,
                    )
            lag = int(sig_cfg.get("lag", 1))
            if lag < 1:
                raise ValueError(
                    f"signals.lag must be at least 1 in a backtest config, got {lag}. "
                    "Zero-lag signals are only available to library callers doing "
                    "research on the signal itself."
                )
            return build_signals(
                prices,
                str(preset),
                freq=freq,
                direction=str(sig_cfg.get("direction", "long_short")),
                horizons=sig_cfg.get("horizons"),
                lag=lag,
                **options,
            )

        momentum_cfg = sig_cfg.get("momentum", {}) or {}
        return timeseries_momentum(
            prices,
            lookbacks=tuple(momentum_cfg.get("lookbacks", [63, 126, 252])),
            skip_last_n=int(momentum_cfg.get("skip_last_n", 20)),
            lag=max(int(sig_cfg.get("lag", 1)), 1),
        )

    def _build_targets(
        self,
        prices: pd.DataFrame,
        cfg: Mapping[str, object],
        *,
        capital: float,
    ) -> pd.DataFrame:
        """Return lagged target positions in contracts."""

        risk_cfg = cfg.get("risk", {}) or {}
        signals = self._build_signals(prices, cfg)

        mask = self._eligibility_mask_for(prices, cfg)
        self._eligibility_mask = mask
        if mask is not None:
            # Masking the signals (not just the final targets) lets the risk
            # budget renormalise over the eligible names; zeroing sized targets
            # instead silently ran the portfolio below its volatility target
            # whenever anything was ineligible.
            signals = signals.where(mask, 0.0)

        max_weight = risk_cfg.get(
            "max_instrument_weight", risk_cfg.get("max_instrument_vol_weight")
        )
        periods_per_year = int(risk_cfg.get("periods_per_year", TRADING_DAYS))

        if risk_cfg.get("disable_vol_scaling"):
            # No volatility information enters sizing: a constant fake vol makes
            # weight_i = budget_i * gross. The constant is chosen so the book is
            # fully invested (gross = 1) rather than the accidental
            # target_vol-sized book a vol of 1.0 produced, which made the 2x2's
            # CAGR and drawdown cells leverage artefacts. Note the budget is
            # proportional to |signal| under the default allocator, so weights
            # are equal-notional only when signals are equal-magnitude.
            gross_level = float(risk_cfg.get("disable_vol_scaling_gross", 1.0))
            constant = float(risk_cfg.get("target_portfolio_vol", 0.15)) / gross_level
            volatility = pd.DataFrame(
                constant, index=prices.index, columns=prices.columns
            )
        else:
            volatility = None

        targets = volatility_target_positions(
            prices=prices,
            signals=signals,
            point_values=self._point_values,
            capital=capital,
            target_portfolio_vol=risk_cfg.get("target_portfolio_vol", 0.15),
            gross_exposure_limit=risk_cfg.get("gross_exposure_limit", 3.0),
            sector_map=self._sector_map,
            sector_caps=risk_cfg.get("sector_caps"),
            contract_rounding=self._contract_steps,
            rebalance_threshold=risk_cfg.get("rebalance_threshold", 0.25),
            rebalance_threshold_mode=str(
                risk_cfg.get("rebalance_threshold_mode", "contracts")
            ),
            vol_model=risk_cfg.get("vol_model", "ewma"),
            vol_lookback=risk_cfg.get("vol_lookback", 63),
            ewma_lambda=risk_cfg.get("ewma_lambda", 0.94),
            min_vol_periods=risk_cfg.get("min_vol_periods", 20),
            risk_allocator=risk_cfg.get("risk_allocator", "proportional"),
            max_position_weight=max_weight,
            max_notional_weight=risk_cfg.get("max_notional_weight"),
            max_asset_weight=risk_cfg.get("max_asset_weight"),
            volatility=volatility,
            periods_per_year=periods_per_year,
            ewma_center_of_mass=risk_cfg.get(
                "ewma_center_of_mass", self._preset_vol_com(cfg)
            ),
            vol_warmup=str(risk_cfg.get("vol_warmup", "mask")),
        )

        if mask is not None:
            targets = targets.where(mask, 0.0)

        # Timing, measured end to end: a price event on day T first changes the
        # position at day T+2's fill. The signal's internal one-bar lag makes
        # row T reflect prices through T-1; this shift pairs with the event
        # loop reading targets.iloc[i + 1] when submitting at day i for a fill
        # at day i+1. Removing either the shift or the iloc[i + 1] alone would
        # introduce a one-day lookahead; removing both would trade one day
        # faster at the cost of re-deriving this analysis. The net two-bar
        # delay is conservative and is documented in CRYPTO_SPEC.md.
        return targets.shift(1).reindex_like(prices).fillna(0.0)

    def _eligibility_mask_for(
        self,
        prices: pd.DataFrame,
        cfg: Mapping[str, object],
    ) -> pd.DataFrame | None:
        """Build the point-in-time eligibility mask for the backtest window.

        The mask is computed on the FULL raw price history handed to the
        backtester, then sliced to the window. Computing it on the window alone
        counted history from the backtest start date, forcing even an
        instrument with a decade of prior data to sit out min_history +
        entry_lag at the beginning of every run (and of every walk-forward
        fold). Raw history also keeps genuine gaps visible to the staleness
        rule, which forward-filled window prices cannot.
        """

        universe_cfg = cfg.get("universe", {}) or {}
        rule_cfg = universe_cfg.get("eligibility")
        if not rule_cfg:
            return None

        from ..data.eligibility import EligibilityRule, build_eligibility_mask

        rule = EligibilityRule.from_config(rule_cfg)
        listing_dates = {
            item["symbol"]: item.get("listing_date")
            for item in self.universe_meta
            if item.get("listing_date")
        }
        full = self.prices.reindex(columns=prices.columns)
        mask = build_eligibility_mask(
            full,
            rule,
            listing_dates=listing_dates or None,
            volumes=self._volumes,
        )
        return mask.reindex(index=prices.index, columns=prices.columns).fillna(False)

    def _simulate_once(self, cfg: Mapping[str, object], *, config_hash: str) -> BacktestResults:
        backtest_cfg = cfg.get("backtest", {})
        start = backtest_cfg.get("start")
        end = backtest_cfg.get("end")
        if start is None or end is None:
            raise ValueError("Backtest configuration requires 'start' and 'end' dates")

        results_dir = Path(backtest_cfg.get("results_dir", str(self.results_dir)))
        results_dir.mkdir(parents=True, exist_ok=True)

        prices = self.prices.loc[str(start) : str(end)].copy()
        data_cfg = cfg.get("data", {}) or {}
        prices = prices.dropna(how="all", axis=1)
        if data_cfg.get("allow_partial_history"):
            # Instruments listed part-way through the window keep their leading
            # NaNs. Dropping any-NaN rows would truncate the whole backtest to
            # the shortest history in the universe, so a BTC-and-SOL run would
            # collapse to SOL's window and silently discard years of BTC data.
            prices = prices.ffill().dropna(how="all")
        else:
            prices = prices.ffill().dropna()

        if prices.empty:
            raise ValueError("No price data available for requested window")

        # Note: no global RNG seeding. The simulation is deterministic; all
        # synthetic-data and bootstrap randomness uses numpy Generator objects
        # seeded explicitly. The old np.random.seed here protected nothing and
        # stomped process-global state under threaded parameter grids.

        exec_cfg = cfg.get("execution", {})
        _warn_unknown_keys(cfg)

        order_type = str(exec_cfg.get("order_type", "market_next_close"))
        if order_type == "market_next_open":
            # Historic configs used this name, but the data model is close-only:
            # there is no open price anywhere, and fills have always been at the
            # next session's close.
            logger.warning(
                "execution.order_type 'market_next_open' is filled at the next "
                "CLOSE; the engine has no open prices. Renaming the config key "
                "to 'market_next_close' silences this warning."
            )
        elif order_type != "market_next_close":
            raise ValueError(
                f"Unsupported execution.order_type: {order_type!r}. The engine "
                "fills at the next session's close ('market_next_close')."
            )

        capital = float(backtest_cfg.get("starting_nav", 1_000_000.0))
        compound_capital = bool(backtest_cfg.get("compound_capital", True))
        commission = float(exec_cfg.get("commission_per_contract", 0.0))
        impact_cfg = exec_cfg.get("impact", {})
        impact_k = float(impact_cfg.get("k", 0.0))
        impact_alpha = float(impact_cfg.get("alpha", 1.0))
        tick_value = float(exec_cfg.get("tick_value", 1.0))
        cost_multiplier = float(exec_cfg.get("cost_multiplier", 1.0))
        spread_bps_cfg = exec_cfg.get("spread_bps", {}) or {}
        if isinstance(spread_bps_cfg, (int, float)):
            spread_bps = {sym: float(spread_bps_cfg) for sym in prices.columns}
        else:
            spread_bps = {sym: float(spread_bps_cfg.get(sym, 0.0)) for sym in prices.columns}
        # Per-instrument dollar-per-tick from metadata tick_size x point_value.
        # A single universe-wide tick value cannot serve instruments whose
        # prices span orders of magnitude; that made a plausible-looking crypto
        # cost configuration nearly frictionless.
        tick_values = {}
        for item in self.universe_meta:
            sym = item.get("symbol")
            tick_size = item.get("tick_size")
            if sym is not None and tick_size is not None:
                tick_values[sym] = float(tick_size) * float(item.get("point_value", 1.0))
        min_slippage_ticks = float(exec_cfg.get("min_slippage_ticks", 0.0))
        adv_limit_pct = float(exec_cfg.get("adv_limit_pct", 0.0))
        default_adv = float(exec_cfg.get("default_adv_contracts", 10_000.0))
        adv_source = exec_cfg.get("average_daily_volume")
        if adv_source is None:
            adv_source = exec_cfg.get("adv_contracts")
        roll_schedule = exec_cfg.get("roll_schedule")
        roll_engine = RollEngine(roll_schedule)

        # Signals and targets are pre-computed to keep the event loop light weight.
        targets = self._build_targets(prices, cfg, capital=capital)

        point_values = pd.Series(self._point_values, index=prices.columns, dtype=float).fillna(1.0)
        adv_frame = resolve_adv_frame(
            adv_source,
            prices.index,
            prices.columns,
            default=default_adv,
        )

        dates = prices.index
        positions = pd.DataFrame(0.0, index=dates, columns=prices.columns)
        nav = pd.Series(index=dates, dtype=float)
        cash_series = pd.Series(index=dates, dtype=float)
        pnl_series = pd.Series(0.0, index=dates, dtype=float)
        costs = pd.DataFrame(0.0, index=dates, columns=["trading", "roll", "total"])

        current_pos = pd.Series(0.0, index=prices.columns, dtype=float)
        pending_orders: dict[str, list[Order]] = {sym: [] for sym in prices.columns}
        orders_log: list[dict[str, object]] = []
        trades_log: list[dict[str, object]] = []
        ledger_rows: list[dict[str, object]] = []

        cash = capital
        first_ts = dates[0]
        first_prices = prices.loc[first_ts]
        first_asset_value = float(
            (current_pos * first_prices * point_values).fillna(0.0).sum()
        )
        nav.loc[first_ts] = cash + first_asset_value
        prev_nav_for_sizing = float(nav.loc[first_ts])

        def _round_targets(desired: pd.Series) -> pd.Series:
            rounded = {}
            for sym, value in desired.items():
                step = float(self._contract_steps.get(sym, 1.0))
                if step <= 0:
                    step = 1.0
                rounded[sym] = float(np.round(float(value) / step) * step)
            return pd.Series(rounded)

        cash_series.loc[first_ts] = cash
        pnl_series.loc[first_ts] = 0.0
        positions.loc[first_ts] = current_pos.copy()
        ledger_rows.append(
            {
                "ts": first_ts,
                "nav": nav.loc[first_ts],
                "cash": cash,
                "asset_value": first_asset_value,
                "pnl": 0.0,
                "trading_cost": 0.0,
                "roll_cost": 0.0,
                "total_cost": 0.0,
            }
        )

        order_counter = 0

        def _log_order(order: Order) -> None:
            orders_log.append(
                {
                    "order_id": order.order_id,
                    "submitted": order.submitted,
                    "symbol": order.symbol,
                    "qty": order.initial_qty,
                    "reason": order.reason,
                    "target": order.target,
                }
            )

        def _submit_order(order: Order) -> None:
            pending_orders[order.symbol].append(order)
            _log_order(order)

        def _submit_rebalance_orders(submit_ts: pd.Timestamp, desired: pd.Series | None) -> None:
            nonlocal order_counter
            if desired is None:
                return
            desired = desired.fillna(0.0)
            # Cancel-and-replace: a new target supersedes any unfilled
            # rebalance quantity. The old FIFO kept a stale order queued ahead
            # of its own reversal, so under an ADV cap a target flip first
            # bought to maximum long on a day the target was flat, then
            # unwound: phantom round trips at double cost, worst exactly when
            # capacity is tight. Roll orders keep their place in the queue.
            for sym in prices.columns:
                pending_orders[sym] = [
                    o for o in pending_orders[sym] if o.reason.startswith("roll")
                ]
            if compound_capital:
                # Targets were sized off starting capital before the loop began.
                # Scaling by the live NAV keeps risk proportional to equity, so
                # the equity curve is geometric and the geometric CAGR the
                # metrics report describes the process that generated it.
                # Without this a run that doubles NAV trades at half the
                # intended risk per unit of equity in its second half.
                scale = max(prev_nav_for_sizing, 0.0) / capital
                desired = _round_targets(desired * scale)
            pending_totals = {
                sym: sum(o.qty for o in pending_orders[sym]) for sym in prices.columns
            }
            for sym in prices.columns:
                target_qty = float(desired.get(sym, 0.0))
                projected = current_pos[sym] + pending_totals[sym]
                delta = target_qty - projected
                if abs(delta) <= 1e-8:
                    continue
                order_counter += 1
                order = Order(
                    order_id=order_counter,
                    submitted=submit_ts,
                    symbol=sym,
                    qty=float(delta),
                    reason="rebalance",
                    target=target_qty,
                )
                pending_totals[sym] += delta
                _submit_order(order)

        def _submit_roll_orders(submit_ts: pd.Timestamp) -> None:
            nonlocal order_counter
            for symbol, qty in roll_engine.orders_for(submit_ts, current_pos):
                if abs(qty) <= 1e-8:
                    continue
                order_counter += 1
                order = Order(
                    order_id=order_counter,
                    submitted=submit_ts,
                    symbol=symbol,
                    qty=float(qty),
                    reason="roll",
                    target=None,
                )
                _submit_order(order)

        if len(dates) > 1:
            _submit_rebalance_orders(first_ts, targets.iloc[1])

        prev_nav = float(nav.loc[first_ts])

        for i in range(1, len(dates)):
            ts = dates[i]
            price_today = prices.loc[ts]
            adv_today = adv_frame.loc[ts]
            day_trading_cost = 0.0
            day_roll_cost = 0.0

            for sym in prices.columns:
                queue = pending_orders[sym]
                if not queue:
                    continue
                adv_value = float(adv_today.get(sym, np.nan))
                if adv_limit_pct > 0 and np.isfinite(adv_limit_pct):
                    capacity = float(adv_limit_pct * adv_value)
                else:
                    capacity = np.inf
                remaining_capacity = capacity
                updated_queue: list[Order] = []
                for order in queue:
                    if remaining_capacity <= 1e-12 and np.isfinite(remaining_capacity):
                        updated_queue.append(order)
                        continue
                    fill_qty = float(order.qty)
                    if np.isfinite(remaining_capacity):
                        allowed = min(abs(order.qty), remaining_capacity)
                        if allowed <= 1e-12:
                            updated_queue.append(order)
                            continue
                        fill_qty = np.sign(order.qty) * allowed
                    if abs(fill_qty) <= 1e-12:
                        updated_queue.append(order)
                        continue

                    price = float(price_today[sym])
                    pv = float(point_values[sym])
                    commission_cost = commission * abs(fill_qty)
                    slippage_cost = participation_slippage(
                        qty=fill_qty,
                        adv=adv_value,
                        k=impact_k,
                        alpha=impact_alpha,
                        tick_value=tick_values.get(sym, tick_value),
                        min_ticks=min_slippage_ticks,
                    )
                    # Proportional half-spread in basis points of notional, the
                    # only cost form that scales sanely across instruments whose
                    # prices differ by orders of magnitude.
                    spread_cost = (
                        abs(fill_qty) * price * pv * spread_bps.get(sym, 0.0) / 10_000.0
                    )
                    trade_cost = (
                        commission_cost + slippage_cost + spread_cost
                    ) * cost_multiplier
                    if order.reason.startswith("roll"):
                        day_roll_cost += trade_cost
                    else:
                        day_trading_cost += trade_cost

                    notional = fill_qty * price * pv
                    cash -= notional + trade_cost
                    current_pos[sym] += fill_qty
                    order.qty -= fill_qty
                    if np.isfinite(remaining_capacity):
                        remaining_capacity = max(0.0, remaining_capacity - abs(fill_qty))

                    trades_log.append(
                        {
                            "ts": ts,
                            "symbol": sym,
                            "qty": fill_qty,
                            "price": price,
                            "notional": notional,
                            "slippage": slippage_cost,
                            "commission": commission_cost,
                            "cost": trade_cost,
                            "reason": order.reason,
                            "order_id": order.order_id,
                            "partial": abs(order.qty) > 1e-9,
                            "cash_after": cash,
                        }
                    )

                    if abs(order.qty) > 1e-9:
                        updated_queue.append(order)
                pending_orders[sym] = updated_queue

            # fillna guards the leading NaN prices of a not-yet-listed
            # instrument, where a zero position times NaN would poison NAV.
            asset_value = float(
                (current_pos * price_today * point_values).fillna(0.0).sum()
            )
            nav_today = cash + asset_value
            pnl_today = nav_today - prev_nav
            prev_nav = nav_today
            prev_nav_for_sizing = nav_today

            nav.loc[ts] = nav_today
            cash_series.loc[ts] = cash
            pnl_series.loc[ts] = pnl_today
            costs.loc[ts, "trading"] = day_trading_cost
            costs.loc[ts, "roll"] = day_roll_cost
            costs.loc[ts, "total"] = day_trading_cost + day_roll_cost
            positions.loc[ts] = current_pos.copy()
            ledger_rows.append(
                {
                    "ts": ts,
                    "nav": nav_today,
                    "cash": cash,
                    "asset_value": asset_value,
                    "pnl": pnl_today,
                    "trading_cost": day_trading_cost,
                    "roll_cost": day_roll_cost,
                    "total_cost": day_trading_cost + day_roll_cost,
                }
            )

            if i + 1 < len(dates):
                _submit_roll_orders(ts)
                _submit_rebalance_orders(ts, targets.iloc[i + 1])

        trades_df = pd.DataFrame(trades_log)
        if not trades_df.empty:
            trades_df.sort_values(["ts", "order_id"], inplace=True)

        orders_df = pd.DataFrame(orders_log)
        if not orders_df.empty:
            orders_df.sort_values(["submitted", "order_id"], inplace=True)

        ledger = pd.DataFrame(ledger_rows).set_index("ts")

        return BacktestResults(
            nav=nav.dropna(),
            positions=positions.dropna(how="all"),
            trades=trades_df.reset_index(drop=True),
            config_hash=config_hash,
            orders=orders_df.reset_index(drop=True),
            costs=costs,
            pnl=pnl_series,
            cash=cash_series,
            ledger=ledger,
            prices=prices,
            point_values=dict(self._point_values),
            eligibility_mask=self._eligibility_mask,
            sector_map=dict(self._sector_map),
        )
