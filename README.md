# tf-trend (starter repo)

A professional-grade **systematic trend-following (managed futures)** research and backtesting scaffold in Python, developed and maintained by Engineer Investor ([@egr_investor](https://x.com/egr_investor)).

## Quick Start

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Smoke test on synthetic data
tf run --config configs/base.yaml

# Produce a basic HTML report
tf report --run-id last

# Run a quick parameter sweep (JSON or YAML grid)
tf sweep --config configs/base.yaml --grid '{"signals.momentum.lookbacks": [[63], [126]]}'

# Walk-forward evaluation with 1-year train / 3-month test windows
tf walkforward --config configs/base.yaml --run-id walk-252x63 --insample 252 --oos 63
```

## Command Line Interface

The `tf` console entrypoint exposes four production-ready commands that cover
single backtests, reporting and both research workflows introduced in phase 8:

| Command | Purpose |
|---------|---------|
| `tf run` | Execute a single backtest and write the full artefact bundle (NAV, positions, trades, charts, summaries). |
| `tf report` | Print the path to the latest HTML/PDF report so it can be opened from automation or notebooks. |
| `tf sweep` | Expand a parameter grid (YAML/JSON) and execute each scenario with optional seed control for robustness studies. |
| `tf walkforward` | Run anchored or rolling walk-forward experiments with configurable train/test splits and per-fold exports. |

All commands accept the same YAML configuration files used by the Python API and
honour override mappings for rapid experimentation.

## Python API & notebooks

The `tf.api` module exposes helpers designed for notebooks and exploratory scripts:

```python
from tf import api

result, context = api.run_backtest("configs/base.yaml")
print(context.universe[0]["symbol"], context.results_dir)

sweep_results, context = api.run_parameter_sweep(
    "configs/base.yaml",
    parameter_grid={"signals.momentum.lookbacks": [[63], [126], [252]]},
    price_seed=123,
)

walkforward_results, _ = api.run_walk_forward(
    "configs/base.yaml",
    insample=252,
    oos=63,
    rolling=True,
)
```

Four worked notebook examples are available in `examples/notebooks/`:

* `quick_start.ipynb` – end-to-end workflow from configuration to analytics.
* `parameter_study.ipynb` – grid search and sweep metadata analysis.
* `attribution.ipynb` – inspect contributions, exposures and roll costs.
* `systematic_trend_following_showcase.ipynb` – comprehensive tour of the full research toolkit.

## Cryptocurrency extension

`tf-trend[crypto]` applies the same engine to cryptocurrency, with the pieces
crypto actually needs: a 7-day calendar, 365-day annualisation, point-in-time
universe eligibility, and a falsification suite that runs by default.

```bash
tf run --config configs/crypto/tsmom_1_3_12.yaml --run-id my-run
tf crypto falsify --config configs/crypto/tsmom_1_3_12.yaml --run-id my-run-falsify
```

Four reference strategies ship, each traceable to published research and none
named for any manager or product:

| Preset | Method |
|---|---|
| `mop2012_tsmom` | Sign of the trailing 12-month return, volatility scaled |
| `tsmom_1_3_12` | Equal-weighted 1, 3, and 12-month time-series momentum |
| `bottom_up_multisystem` | Four trend systems across thirteen horizons |
| `btc_long_flat` | 200-day price versus moving average, long or flat |

Every preset runs long/short or long/flat, so the choice stays an empirical
question rather than a built-in assumption.

`tf crypto falsify` is the part worth running. It places a result beside
benchmarks that require no skill, re-runs it at higher costs and later signals,
separates what the trend signal contributed from what volatility scaling
contributed, and compares it against a placebo with randomised signal directions.
It also reports when its own tests cannot bite, for example when transaction
costs are configured at zero.

- Methodology and evidence: [`CRYPTO_SPEC.md`](CRYPTO_SPEC.md)
- Data schema, sourcing, licensing, and modelling limits: [`docs/CRYPTO_DATA.md`](docs/CRYPTO_DATA.md)

No market data ships with this repository. The shipped configs run on synthetic
prices so they work offline; switch `data.prefer` to `auto` with `data.strict:
true` and supply your own data before drawing any conclusion.

## Documentation set

Additional user guides live under `docs/`:

* `docs/ADD_INSTRUMENT.md` documents the metadata required when onboarding a new contract.
* `docs/HOW_ROLL_WORKS.md` walks through the roll schedule, continuous series builder and analytics.
* `docs/COST_MODEL_CALIBRATION.md` explains how to calibrate commissions, slippage and ADV assumptions.
* `docs/CRYPTO_DATA.md` covers the crypto data schema, sourcing constraints and modelling limits.

The top-level `README` pairs with `SPEC.md` for a condensed architectural overview.

## Architecture overview

```mermaid
flowchart TD
    subgraph DataLayer[Data & Calendars]
        vendors["Raw Vendors / Local Files"]
        ingest["Ingestion, Validation, Calendars"]
        store["Research-Ready Prices"]
        vendors --> ingest --> store
    end

    subgraph ResearchLayer[Signals & Risk]
        signals["Signal Library (Momentum / MA / Breakout)"]
        risk["Risk & Allocation"]
        ResearchContext["Universe & Config"]
        store --> signals
        ResearchContext --> signals
        signals --> risk
    end

    subgraph Engine[Backtest Engine]
        simulate["Daily Simulation Loop"]
        costs["Execution & Cost Models"]
        risk --> simulate
        simulate --> costs
        ResearchContext --> simulate
    end

    subgraph Results[Analytics & Reporting]
        artefacts["Trades / NAV / Exposures"]
        analytics["Metrics & Attribution"]
        reports["HTML / Charts"]
        costs --> artefacts
        simulate --> artefacts
        artefacts --> analytics --> reports
    end

    CLI["CLI & API"]
    ResearchContext --> CLI
    CLI --> ResearchContext
    CLI --> simulate
    CLI --> analytics
```

## Data quality & QA utilities

Phase 9 adds explicit QA helpers for market data edge cases:

* `tf.data.validators.validate_price_data` enforces monotonic indices, minimum price checks and configurable gap limits.
* `tf.data.validators.detect_trading_suspensions` summarises multi-day trading halts so they can be investigated.
* `tf.data.validators.detect_limit_moves` returns a boolean mask for potential limit-up/down events using percentage thresholds.

Refer to the `tests/test_data_layer.py` suite for concrete usage patterns and expectations around suspensions, holiday gaps and limit moves.

## Layout

```
src/tf/
  data/
  signals/
  risk/
  portfolio/
  engine/
  costs/
  eval/
  report/
  cli.py
configs/
tests/
examples/
```

This is a scaffold with runnable defaults (synthetic data generation) and room to extend.
See `SPEC.md` for the system requirements summary.

## Market data options

The starter configuration now understands contract metadata and can pull daily
settlements from Yahoo! Finance when you prefer real market history.  Populate
``configs/universe.yaml`` with the contracts you care about and run:

```bash
tf run --config configs/base.yaml  # attempts Yahoo first, falls back to synthetic
```

If the Yahoo download fails (e.g. you are offline) the loader automatically
reverts to the reproducible synthetic data generator so unit tests and quick
examples continue to work without an internet connection.

## Testing & quality gates

The automated test suite exercises the full stack, including CLI entry points,
parameter sweeps, walk-forward splits and the new data quality audits.  Run it
locally with:

```bash
pytest -q
```

CI and local developers alike should expect the suite to finish in under a
minute on modern hardware.

## Packaging & releases

The project is packaged as `tf-trend` with dependencies pinned in `pyproject.toml`.
Install via `pip install tf-trend` once a distribution has been built, or use
`pip install -e .` for editable development installs.

Versioned releases are tracked in `CHANGELOG.md`.  Update both the changelog and
the package version before cutting a new release tag to keep artefacts in sync.
