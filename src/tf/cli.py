"""Command line interface for the trend-following toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich import print as rprint

from . import api
from .engine.backtester import BacktestResults
from .eval.analytics import (
    compute_exposures,
    compute_pnl_contributions,
    compute_roll_cost_breakdown,
    compute_sector_contributions,
    normalise_totals,
)
from .eval.metrics import TRADING_DAYS, compute_rolling_metrics, performance_summary
from .report.exporter import write_html, write_pdf
from .report.plots import (
    plot_contribution_breakdown,
    plot_correlation_heatmap,
    plot_drawdown,
    plot_equity,
    plot_exposure_timeseries,
    plot_histogram,
    plot_rolling_stats,
)


def _export_table(
    data: pd.Series | pd.DataFrame,
    base_path: Path,
    label: str,
    *,
    index: bool = True,
) -> None:
    if isinstance(data, pd.Series):
        data.to_csv(base_path.with_suffix(".csv"))
        frame = data.to_frame(name=data.name or label)
    else:
        data.to_csv(base_path.with_suffix(".csv"), index=index)
        frame = data
    try:
        frame.to_parquet(base_path.with_suffix(".parquet"))
    except Exception as exc:  # pragma: no cover - optional dependency
        rprint(f"[yellow]Skipped Parquet export for {label}: {exc}[/yellow]")


def _parse_mapping_arg(value: str | None, *, label: str) -> dict[str, Any] | None:
    if value is None:
        return None

    candidate = Path(value)
    text: str
    if candidate.exists():
        text = candidate.read_text()
    else:
        text = value

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(text)

    if parsed is None:
        return None
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must evaluate to a mapping (dict-like) structure")
    return dict(parsed)


def _prepare_run_directory(context: api.BacktestContext, run_id: str) -> Path:
    outdir = context.results_dir / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _periods_per_year_from(config: Mapping[str, Any]) -> int:
    """Annualisation basis from ``risk.periods_per_year`` (default 252).

    Positions are sized on this basis, so reports must use the same one; a
    crypto config sized at 365 and reported at 252 understates volatility and
    Sharpe by about 17 percent.
    """

    risk_cfg = config.get("risk", {})
    if isinstance(risk_cfg, Mapping):
        try:
            return int(risk_cfg.get("periods_per_year", TRADING_DAYS))
        except (TypeError, ValueError):
            return TRADING_DAYS
    return TRADING_DAYS


def _write_backtest_outputs(
    result: BacktestResults,
    *,
    config: Mapping[str, Any],
    outdir: Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)

    _export_table(result.nav.rename("nav"), outdir / "nav", "NAV")
    _export_table(result.positions, outdir / "positions", "Positions")
    _export_table(result.trades, outdir / "trades", "Trades", index=False)

    if result.cash is not None:
        _export_table(result.cash.rename("cash"), outdir / "cash", "Cash")
    if result.ledger is not None:
        _export_table(result.ledger, outdir / "ledger", "Ledger")

    contributions = compute_pnl_contributions(result.prices, result.positions, result.point_values)
    sector_contrib = compute_sector_contributions(contributions, result.sector_map)
    exposures = compute_exposures(result.prices, result.positions, result.point_values)
    roll_costs = compute_roll_cost_breakdown(result.trades, result.costs)
    report_cfg = config.get("report", {}) if isinstance(config.get("report", {}), Mapping) else {}
    rolling_window = int(report_cfg.get("rolling_window", 126))
    periods_per_year = _periods_per_year_from(config)
    rolling_stats = compute_rolling_metrics(
        result.nav, window=rolling_window, periods_per_year=periods_per_year
    )

    if not contributions.empty:
        _export_table(contributions, outdir / "contributions", "Contributions")
    if not sector_contrib.empty:
        _export_table(sector_contrib, outdir / "sector_contributions", "Sector Contributions")
    if not exposures.empty:
        _export_table(exposures, outdir / "exposures", "Exposures")
    if not roll_costs.empty:
        _export_table(roll_costs, outdir / "roll_costs", "Roll Costs")
    if not rolling_stats.empty:
        _export_table(rolling_stats, outdir / "rolling_stats", "Rolling Metrics")

    summary = performance_summary(
        result.nav,
        pnl=result.pnl,
        trades=result.trades,
        periods_per_year=periods_per_year,
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    charts: list[tuple[str, str]] = []
    equity_path = outdir / "equity.png"
    plot_equity(result.nav, equity_path)
    charts.append(("Equity Curve", equity_path.name))

    drawdown_path = outdir / "drawdown.png"
    plot_drawdown(result.nav, drawdown_path)
    if drawdown_path.exists():
        charts.append(("Underwater Curve", drawdown_path.name))

    if not rolling_stats.empty:
        rolling_path = outdir / "rolling_stats.png"
        plot_rolling_stats(rolling_stats, rolling_path, window=rolling_window)
        if rolling_path.exists():
            charts.append((f"Rolling {rolling_window}-Day Stats", rolling_path.name))

    instrument_totals = contributions.sum() if not contributions.empty else pd.Series(dtype=float)
    if not instrument_totals.empty:
        top_instruments = normalise_totals(instrument_totals, top_n=10)
        contrib_path = outdir / "instrument_contributions.png"
        plot_contribution_breakdown(top_instruments, contrib_path, title="Top Instrument Contributions")
        if contrib_path.exists():
            charts.append(("Instrument Contributions", contrib_path.name))

    if not sector_contrib.empty:
        sector_totals = normalise_totals(sector_contrib.sum(), top_n=10)
        sector_path = outdir / "sector_contributions.png"
        plot_contribution_breakdown(sector_totals, sector_path, title="Sector Contributions")
        if sector_path.exists():
            charts.append(("Sector Contributions", sector_path.name))

    if not exposures.empty:
        exposure_path = outdir / "exposures.png"
        plot_exposure_timeseries(exposures, exposure_path)
        if exposure_path.exists():
            charts.append(("Instrument Exposures", exposure_path.name))

    returns = result.nav.pct_change().dropna()
    if not returns.empty:
        hist_path = outdir / "return_histogram.png"
        plot_histogram(returns, hist_path, title="Daily Return Distribution")
        if hist_path.exists():
            charts.append(("Return Distribution", hist_path.name))

    if result.prices is not None:
        instrument_returns = result.prices.pct_change().dropna()
        heatmap_path = outdir / "return_correlation_heatmap.png"
        plot_correlation_heatmap(
            instrument_returns,
            heatmap_path,
            title="Instrument Return Correlation",
        )
        if heatmap_path.exists():
            charts.append(("Instrument Correlation", heatmap_path.name))

    tables: list[tuple[str, pd.DataFrame | pd.Series | Mapping[str, Any]]] = []
    if not roll_costs.empty:
        tables.append(("Roll Cost Breakdown", roll_costs))
    if not instrument_totals.empty:
        tables.append(("Instrument Contribution Totals", instrument_totals.to_frame(name="PnL")))
    if not sector_contrib.empty:
        tables.append(("Sector Contribution Totals", sector_contrib.sum().to_frame(name="PnL")))

    metadata_payload: dict[str, Any] = {
        "window": f"{result.nav.index.min().date()} → {result.nav.index.max().date()}",
        "config_hash": result.config_hash,
    }
    if metadata:
        metadata_payload.update({str(key): str(value) for key, value in metadata.items()})

    write_html(outdir, summary, tables=tables, charts=charts, metadata=metadata_payload)
    write_pdf(outdir, summary, tables=tables, charts=charts, metadata=metadata_payload)
    return summary


def cmd_run(args: argparse.Namespace) -> None:
    overrides = _parse_mapping_arg(args.overrides, label="overrides")
    result, context = api.run_backtest(
        args.config,
        parameter_overrides=overrides,
        seed=args.seed,
        price_seed=args.price_seed,
    )
    outdir = _prepare_run_directory(context, args.run_id)
    metadata = {"run_id": args.run_id}
    seed_value = args.seed if args.seed is not None else context.config.get("backtest", {}).get("seed")
    if seed_value is not None:
        metadata["seed"] = seed_value
    _write_backtest_outputs(result, config=context.config, outdir=outdir, metadata=metadata)
    rprint(f"[bold green]Run complete[/bold green]. Results in {outdir}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = api.load_config(args.config)
    backtest_cfg = cfg.get("backtest", {})
    results_dir = Path(backtest_cfg.get("results_dir", "./results"))
    if args.run_id == "last":
        candidates = sorted(
            [p for p in results_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No runs found.")
            sys.exit(1)
        run_dir = candidates[0]
    else:
        run_dir = results_dir / args.run_id
    print(f"Report available at: {run_dir / 'report.html'}")


def cmd_sweep(args: argparse.Namespace) -> None:
    parameter_grid = _parse_mapping_arg(args.grid, label="parameter grid")
    if not parameter_grid:
        raise ValueError("Parameter grid must not be empty for a sweep")
    base_overrides = _parse_mapping_arg(args.base_overrides, label="base overrides")

    seeds: Sequence[int] | None
    if args.seeds is None or len(args.seeds) == 0:
        seeds = None
    else:
        seeds = args.seeds

    sweep_results, context = api.run_parameter_sweep(
        args.config,
        parameter_grid=parameter_grid,
        base_overrides=base_overrides,
        seeds=seeds,
        price_seed=args.price_seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
    )

    base_dir = _prepare_run_directory(context, args.run_id)
    api.export_sweep_metadata(sweep_results, base_dir / "scenarios.json")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(sweep_results, start=1):
        scenario_dir = base_dir / f"scenario-{idx:03d}"
        metadata = {
            "run_id": args.run_id,
            "scenario": idx,
            "seed": item.scenario.seed,
        }
        summary = _write_backtest_outputs(
            item.backtest,
            config=context.config,
            outdir=scenario_dir,
            metadata={
                **metadata,
                "grid_overrides": json.dumps(item.scenario.grid_overrides, default=str)
                if item.scenario.grid_overrides
                else "{}",
            },
        )
        record = {"scenario": idx, "seed": item.scenario.seed}
        record["grid_overrides"] = json.dumps(item.scenario.grid_overrides, default=str)
        record.update(summary)
        records.append(record)

    if records:
        summary_frame = pd.DataFrame(records).set_index("scenario")
        _export_table(summary_frame, base_dir / "sweep_summary", "Sweep Summary")

    rprint(f"[bold green]Sweep complete[/bold green]. Results in {base_dir}")


def cmd_walkforward(args: argparse.Namespace) -> None:
    parameter_overrides = _parse_mapping_arg(args.parameter_overrides, label="parameter overrides")
    anchored = not args.rolling

    wf_results, context = api.run_walk_forward(
        args.config,
        insample=args.insample,
        oos=args.oos,
        step=args.step,
        anchored=anchored,
        parameter_overrides=parameter_overrides,
        seed=args.seed,
        price_seed=args.price_seed,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
    )

    if not wf_results:
        rprint("[yellow]No walk-forward windows generated.[/yellow]")
        return

    base_dir = _prepare_run_directory(context, args.run_id)
    records: list[dict[str, Any]] = []

    for idx, wf in enumerate(wf_results, start=1):
        window = wf.window
        oos_label = f"{window.oos_start.date()}_{window.oos_end.date()}"
        fold_dir = base_dir / f"fold-{idx:03d}_{oos_label}"
        metadata = {
            "fold": idx,
            "insample": f"{window.insample_start.date()} → {window.insample_end.date()}",
            "oos": f"{window.oos_start.date()} → {window.oos_end.date()}",
        }
        summary = _write_backtest_outputs(wf.backtest, config=context.config, outdir=fold_dir, metadata=metadata)
        _export_table(wf.insample_nav.rename("nav"), fold_dir / "insample_nav", "In-sample NAV")
        _export_table(wf.oos_nav.rename("nav"), fold_dir / "oos_nav", "Out-of-sample NAV")
        (fold_dir / "insample_summary.json").write_text(json.dumps(wf.insample_summary, indent=2))
        (fold_dir / "oos_summary.json").write_text(json.dumps(wf.oos_summary, indent=2))

        record: dict[str, Any] = {
            "fold": idx,
            "train_start": str(window.insample_start.date()),
            "train_end": str(window.insample_end.date()),
            "test_start": str(window.oos_start.date()),
            "test_end": str(window.oos_end.date()),
        }
        record.update({f"IS_{k}": v for k, v in wf.insample_summary.items()})
        record.update({f"OOS_{k}": v for k, v in wf.oos_summary.items()})
        record.update(summary)
        records.append(record)

    summary_frame = pd.DataFrame(records).set_index("fold")
    _export_table(summary_frame, base_dir / "walkforward_summary", "Walk-forward Summary")
    rprint(f"[bold green]Walk-forward complete[/bold green]. Results in {base_dir}")


def cmd_crypto_falsify(args: argparse.Namespace) -> None:
    """Run the falsification suite and write its report bundle."""

    from .eval.falsification import run_falsification

    backtester, context = api.prepare_backtester(
        args.config, price_seed=args.price_seed
    )
    report = run_falsification(
        backtester, placebo_draws=args.placebo_draws, seed=args.seed
    )

    outdir = _prepare_run_directory(context, args.run_id)
    tables = report.tables()
    for title, frame in tables:
        if frame.empty:
            continue
        slug = title.lower().replace(" ", "_").replace(".", "").replace("/", "_")
        _export_table(frame.reset_index(), outdir / slug, title)

    metadata = {"run_id": args.run_id, "report": "falsification"}
    for index, note in enumerate(report.notes, start=1):
        metadata[f"note_{index}"] = note

    write_html(
        outdir,
        _headline_summary(report),
        tables=[(title, frame) for title, frame in tables if not frame.empty],
        charts=[],
        metadata=metadata,
    )

    rprint(f"[bold green]Falsification report written[/bold green] to {outdir}")
    for note in report.notes:
        rprint(f"[yellow]Note:[/yellow] {note}")


def _headline_summary(report) -> dict[str, float]:
    """Pull the strategy row out of the benchmark table for the report header."""

    variants = report.variants
    if variants.empty or "strategy" not in variants.index:
        return {}
    row = variants.loc["strategy"]
    return {str(k): float(v) for k, v in row.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tf", description="Trend Following Backtester")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a backtest")
    p_run.add_argument("--config", default="configs/base.yaml")
    p_run.add_argument("--run-id", default="run-001")
    p_run.add_argument("--overrides", help="JSON/YAML mapping or path with parameter overrides")
    p_run.add_argument("--seed", type=int)
    p_run.add_argument("--price-seed", type=int, help="Override the synthetic price seed")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Show latest report path")
    p_rep.add_argument("--config", default="configs/base.yaml")
    p_rep.add_argument("--run-id", default="last", help="'last' or specific run-id")
    p_rep.set_defaults(func=cmd_report)

    p_sweep = sub.add_parser("sweep", help="Run a parameter grid sweep")
    p_sweep.add_argument("--config", default="configs/base.yaml")
    p_sweep.add_argument("--run-id", default="sweep-001")
    p_sweep.add_argument("--grid", required=True, help="JSON/YAML mapping or file describing the grid")
    p_sweep.add_argument("--base-overrides", help="Optional baseline overrides applied to every scenario")
    p_sweep.add_argument("--seeds", nargs="*", type=int, help="Optional list of seeds to iterate")
    p_sweep.add_argument("--n-jobs", type=int)
    p_sweep.add_argument("--prefer", choices=["processes", "threads"], help="Parallel backend preference")
    p_sweep.add_argument("--price-seed", type=int, help="Override the synthetic price seed")
    p_sweep.set_defaults(func=cmd_sweep)

    p_wf = sub.add_parser("walkforward", help="Run a walk-forward analysis")
    p_wf.add_argument("--config", default="configs/base.yaml")
    p_wf.add_argument("--run-id", default="walk-001")
    p_wf.add_argument("--insample", type=int, required=True, help="In-sample window length in trading days")
    p_wf.add_argument("--oos", type=int, required=True, help="Out-of-sample window length in trading days")
    p_wf.add_argument("--step", type=int, help="Step between successive walk-forward windows")
    p_wf.add_argument("--rolling", action="store_true", help="Use rolling rather than anchored windows")
    p_wf.add_argument("--parameter-overrides", help="JSON/YAML mapping or file with overrides")
    p_wf.add_argument("--seed", type=int, help="Simulation seed applied to each walk-forward fold")
    p_wf.add_argument("--price-seed", type=int, help="Override the synthetic price seed")
    p_wf.add_argument("--n-jobs", type=int)
    p_wf.add_argument("--prefer", choices=["processes", "threads"], help="Parallel backend preference")
    p_wf.set_defaults(func=cmd_walkforward)

    p_crypto = sub.add_parser("crypto", help="Cryptocurrency extension commands")
    crypto_sub = p_crypto.add_subparsers(dest="crypto_cmd", required=True)

    p_fals = crypto_sub.add_parser(
        "falsify",
        help="Run the falsification suite: benchmarks, stress tests, and a placebo",
    )
    p_fals.add_argument("--config", default="configs/crypto/tsmom_1_3_12.yaml")
    p_fals.add_argument("--run-id", default="falsify-001")
    p_fals.add_argument(
        "--placebo-draws",
        type=int,
        default=1_000,
        help="Randomised-signal draws (lower is faster and noisier)",
    )
    p_fals.add_argument("--seed", type=int, default=0)
    p_fals.add_argument("--price-seed", type=int, help="Override the synthetic price seed")
    p_fals.set_defaults(func=cmd_crypto_falsify)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:  # pragma: no cover - defensive user feedback
        rprint(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
