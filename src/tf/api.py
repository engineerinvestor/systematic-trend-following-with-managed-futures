"""High-level helpers for notebooks and scripts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collections.abc import Mapping, MutableMapping, Sequence

import pandas as pd
import yaml

from .data.calendar import TradingCalendar
from .data.ingest import load_prices_or_generate
from .engine.backtester import BacktestResults, Backtester, expand_parameter_grid
from .research.walkforward import WalkForwardResult


#: Seed used for synthetic prices when a config omits ``backtest.seed``.
#: Previously ``None`` was forwarded, making synthetic runs irreproducible.
DEFAULT_PRICE_SEED = 42


@dataclass(frozen=True)
class BacktestContext:
    """Runtime artefacts that are useful in notebooks."""

    config: dict[str, Any]
    config_path: Path | None
    config_dir: Path | None
    universe: list[dict[str, Any]]
    prices: pd.DataFrame
    price_seed: int | None

    @property
    def results_dir(self) -> Path:
        base = self.config.get("backtest", {}).get("results_dir", "./results")
        return Path(base)


@dataclass(frozen=True)
class ParameterSweepScenario:
    """Description of a single sweep iteration."""

    grid_overrides: dict[str, Any] | None
    base_overrides: dict[str, Any] | None
    combined_overrides: dict[str, Any] | None
    seed: int | None


@dataclass(frozen=True)
class ParameterSweepResult:
    """Combine a sweep scenario with the produced backtest results."""

    scenario: ParameterSweepScenario
    backtest: BacktestResults


def load_config(path: str | Path) -> dict[str, Any]:
    """Return a deep copy of the YAML configuration file."""

    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise TypeError("Configuration file must define a mapping at the top level")
    return copy.deepcopy(data)


def load_universe(config: Mapping[str, Any], *, base_path: Path | None = None) -> list[dict[str, Any]]:
    """Load the universe metadata from the path stored in ``config``."""

    universe_cfg = config.get("universe", {})
    if not isinstance(universe_cfg, Mapping):
        raise TypeError("'universe' configuration must be a mapping")
    assets_file = universe_cfg.get("assets_file")
    if assets_file is None:
        raise KeyError("Universe configuration must provide 'assets_file'")

    assets_path = Path(str(assets_file))
    if not assets_path.is_absolute():
        candidates: list[Path] = []
        if base_path is not None:
            candidates.append((base_path / assets_path).resolve())
            parent = base_path.parent
            if parent != base_path:
                candidates.append((parent / assets_path).resolve())
        candidates.append((Path.cwd() / assets_path).resolve())

        for candidate in candidates:
            if candidate.exists():
                assets_path = candidate
                break
        else:
            if base_path is not None:
                assets_path = (base_path / assets_path).resolve()
            else:
                assets_path = (Path.cwd() / assets_path).resolve()

    if not assets_path.exists():
        raise FileNotFoundError(f"Universe asset file not found: {assets_path}")
    assets_data = yaml.safe_load(assets_path.read_text())
    symbols = assets_data.get("symbols")
    if not isinstance(symbols, list):
        raise TypeError("Universe file must contain a 'symbols' list")
    return [dict(symbol) for symbol in symbols]


def merge_overrides(base: Mapping[str, Any] | None, extra: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Recursively merge ``extra`` on top of ``base`` without mutating inputs."""

    if not base and not extra:
        return None

    def _merge(lhs: MutableMapping[str, Any], rhs: Mapping[str, Any]) -> MutableMapping[str, Any]:
        for key, value in rhs.items():
            if isinstance(value, Mapping) and isinstance(lhs.get(key), MutableMapping):
                lhs[key] = _merge(lhs[key], value)  # type: ignore[index]
            elif isinstance(value, Mapping):
                lhs[key] = _merge({}, value)
            else:
                lhs[key] = copy.deepcopy(value)
        return lhs

    merged: MutableMapping[str, Any] = {}
    if base:
        merged = _merge({}, base)
    if extra:
        merged = _merge(merged, extra)
    return dict(merged)


def _prepare_context(
    config: dict[str, Any] | str | Path,
    *,
    prices: pd.DataFrame | None = None,
    universe: Sequence[Mapping[str, Any]] | None = None,
    price_seed: int | None = None,
) -> BacktestContext:
    if isinstance(config, (str, Path)):
        cfg_path = Path(config)
        cfg_data = load_config(cfg_path)
        config_dir = cfg_path.parent.resolve()
    else:
        cfg_path = None
        cfg_data = copy.deepcopy(config)
        config_dir = None

    if universe is None:
        universe_list = load_universe(cfg_data, base_path=config_dir)
    else:
        universe_list = [dict(item) for item in universe]

    backtest_cfg = cfg_data.get("backtest", {})
    if not isinstance(backtest_cfg, Mapping):
        raise TypeError("'backtest' configuration must be a mapping")

    start = backtest_cfg.get("start")
    end = backtest_cfg.get("end")
    if start is None or end is None:
        raise KeyError("Backtest configuration requires 'start' and 'end'")

    resolved_price_seed = price_seed if price_seed is not None else backtest_cfg.get("seed")

    data_cfg = cfg_data.get("data", {})
    prefer_prices = "auto"
    calendar_name = None
    strict_prices = False
    if isinstance(data_cfg, Mapping):
        prefer_prices = str(
            data_cfg.get("prefer_prices", data_cfg.get("price_preference", data_cfg.get("prefer", "auto")))
        ).lower()
        calendar_name = data_cfg.get("calendar")
        strict_prices = bool(data_cfg.get("strict", False))

    if prices is None:
        prices = load_prices_or_generate(
            universe_list,
            str(start),
            str(end),
            seed=DEFAULT_PRICE_SEED
            if resolved_price_seed is None
            else int(resolved_price_seed),
            prefer=prefer_prices,
            calendar=TradingCalendar.from_name(calendar_name),
            strict=strict_prices,
        )
    else:
        prices = prices.copy()

    return BacktestContext(
        config=cfg_data,
        config_path=cfg_path,
        config_dir=config_dir,
        universe=[dict(item) for item in universe_list],
        prices=prices,
        price_seed=None if resolved_price_seed is None else int(resolved_price_seed),
    )


def prepare_backtester(
    config: dict[str, Any] | str | Path,
    *,
    prices: pd.DataFrame | None = None,
    universe: Sequence[Mapping[str, Any]] | None = None,
    price_seed: int | None = None,
) -> tuple[Backtester, BacktestContext]:
    """Return a backtester instance alongside the resolved context."""

    context = _prepare_context(config, prices=prices, universe=universe, price_seed=price_seed)
    backtester = Backtester(context.prices, context.universe, context.config)
    return backtester, context


def run_backtest(
    config: dict[str, Any] | str | Path,
    *,
    parameter_overrides: Mapping[str, Any] | None = None,
    seed: int | None = None,
    price_seed: int | None = None,
    check_reproducibility: bool | None = None,
) -> tuple[BacktestResults, BacktestContext]:
    """Execute a single backtest and return the results and context."""

    backtester, context = prepare_backtester(config, price_seed=price_seed)
    results = backtester.run(
        parameter_overrides=parameter_overrides,
        seed=seed,
        check_reproducibility=check_reproducibility,
    )
    return results, context


def _resolve_seeds(seeds: Sequence[int] | int | None) -> list[int | None]:
    if seeds is None:
        return [None]
    if isinstance(seeds, int):
        return [seeds]
    return list(seeds)


def run_parameter_sweep(
    config: dict[str, Any] | str | Path,
    *,
    parameter_grid: Mapping[str, Sequence[Any]] | None = None,
    base_overrides: Mapping[str, Any] | None = None,
    seeds: Sequence[int] | int | None = None,
    price_seed: int | None = None,
    check_reproducibility: bool | None = None,
    n_jobs: int | None = None,
    prefer: str | None = None,
) -> tuple[list[ParameterSweepResult], BacktestContext]:
    """Run a parameter grid sweep and capture the metadata for each iteration."""

    backtester, context = prepare_backtester(config, price_seed=price_seed)
    overrides_list = expand_parameter_grid(parameter_grid or {})
    if not overrides_list:
        overrides_list = [None]
    seeds_list = _resolve_seeds(seeds)

    scenarios: list[ParameterSweepScenario] = []
    for overrides in overrides_list:
        for seed_value in seeds_list:
            combined = merge_overrides(base_overrides, overrides)
            scenarios.append(
                ParameterSweepScenario(
                    grid_overrides=None if overrides is None else copy.deepcopy(overrides),
                    base_overrides=None if base_overrides is None else copy.deepcopy(base_overrides),
                    combined_overrides=None if combined is None else copy.deepcopy(combined),
                    seed=None if seed_value is None else int(seed_value),
                )
            )

    results = backtester.run_parameter_grid(
        parameter_grid=parameter_grid,
        base_overrides=base_overrides,
        seeds=seeds,
        check_reproducibility=check_reproducibility,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    if len(results) != len(scenarios):
        raise RuntimeError("Mismatch between generated scenarios and backtest results")

    sweep_results = [
        ParameterSweepResult(scenario=scenario, backtest=result)
        for scenario, result in zip(scenarios, results, strict=True)
    ]
    return sweep_results, context


def run_walk_forward(
    config: dict[str, Any] | str | Path,
    *,
    insample: int,
    oos: int,
    step: int | None = None,
    anchored: bool = True,
    parameter_overrides: Mapping[str, Any] | None = None,
    seed: int | None = None,
    price_seed: int | None = None,
    check_reproducibility: bool | None = None,
    n_jobs: int | None = None,
    prefer: str | None = None,
) -> tuple[list[WalkForwardResult], BacktestContext]:
    """Run a walk-forward study for the requested configuration."""

    backtester, context = prepare_backtester(config, price_seed=price_seed)
    results = backtester.run_walk_forward(
        insample=insample,
        oos=oos,
        step=step,
        anchored=anchored,
        parameter_overrides=parameter_overrides,
        seed=seed,
        check_reproducibility=check_reproducibility,
        n_jobs=n_jobs,
        prefer=prefer,
    )
    return results, context


def serialise_sweep_scenarios(results: Sequence[ParameterSweepResult]) -> list[dict[str, Any]]:
    """Return JSON-serialisable metadata for sweep scenarios."""

    payload: list[dict[str, Any]] = []
    for item in results:
        scenario = item.scenario
        payload.append(
            {
                "grid_overrides": scenario.grid_overrides,
                "base_overrides": scenario.base_overrides,
                "combined_overrides": scenario.combined_overrides,
                "seed": scenario.seed,
                "config_hash": item.backtest.config_hash,
            }
        )
    return payload


def export_sweep_metadata(results: Sequence[ParameterSweepResult], path: Path | str) -> Path:
    """Persist sweep metadata as a JSON file."""

    payload = serialise_sweep_scenarios(results)
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2))
    return path


__all__ = [
    "BacktestContext",
    "ParameterSweepResult",
    "ParameterSweepScenario",
    "export_sweep_metadata",
    "load_config",
    "load_universe",
    "merge_overrides",
    "prepare_backtester",
    "run_backtest",
    "run_parameter_sweep",
    "run_walk_forward",
    "serialise_sweep_scenarios",
]
