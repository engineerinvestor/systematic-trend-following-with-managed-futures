# Changelog

## [Unreleased]

### Added
- MIT license file and attribution to Engineer Investor ([@egr_investor](https://x.com/egr_investor)).
- `CRYPTO_SPEC.md`, the specification for the `tf-trend[crypto]` extension.
- Seven-day calendar support: `TradingCalendar` takes a `freq` of `"weekday"` or
  `"daily"`, and `TradingCalendar.from_name` resolves config names such as
  `CRYPTO_DAILY`. `config["data"]["calendar"]` is now read and threaded through
  price loading; it was previously ignored.
- `periods_per_year` on `performance_summary`, `compute_rolling_metrics`, and
  `bootstrap_confidence_intervals`, plus `metrics.CALENDAR_DAYS` (365). Defaults
  stay at 252, so existing results are unchanged.
- `risk.vol.com_to_lambda` and an `ewma_vol(center_of_mass=...)` argument, so
  volatility windows can be stated as a center of mass as the trend-following
  literature does.
- `max_asset_weight` in `volatility_target_positions`, capping any one
  instrument's share of gross exposure. Sector caps do not constrain a single
  name within its own sector.
- `load_prices_or_generate(strict=True)` and a `data.strict` config key, which
  disable the synthetic fallback entirely.
- `generate_synthetic_prices` gained `freq`, per-symbol `vol` and `mu`, a
  `common_factor` for cross-sectional correlation, and `start_offsets` for
  staggered listing histories.
- `.gitignore`, `crypto` and `dev` extras, and pytest `testpaths`/`pythonpath`.
- `tf.data.cme_crypto`: CME crypto contract specifications, the 29 May 2026
  move to around-the-clock trading encoded as a dated session regime, session
  and fill-lag helpers, and quarterly roll-schedule construction feeding the
  existing continuous-series builders.
- `tf.data.crypto`: reference-price validation that rejects calendar gaps and
  non-positive prices, spot and local-CSV universe builders, and a `data_span`
  report of per-instrument history.
- `tf.costs.crypto`: per-instrument and per-era spread assumptions, commission
  and participation impact, and `scale_execution_costs` for cost stress tests.
- `ContractMetadata` gained `tick_size`, `listing_date`, `calendar`, `venue`,
  `contract_size`, and `session_regime_changes`. Unknown keys are still
  rejected, so configuration typos still surface immediately.
- `tf.crypto`: four reference presets (`mop2012_tsmom`, `tsmom_1_3_12`,
  `bottom_up_multisystem`, `btc_long_flat`), each available long/short and
  long/flat, plus calendar-aware horizon resolution so `"365D"` means 365 bars
  on a 7-day calendar and 252 on a weekday one.
- `signals.breakout.donchian_breakout`: a Donchian entry rule in {-1, 0, +1}
  computed from the bands *before* the current bar.
- `signals.moving_average.price_minus_ma`: price against its own moving
  average, the PMAC system named in trend-following research.
- `timeseries_momentum` gained `transform` (`"sign"` for the published TSMOM
  construction) and `weighting` (`"equal"` for the 1/3/12-month ensemble).
- `signals.preset`, `signals.direction`, `signals.disable_trend`, and
  `risk.disable_vol_scaling` config keys. The last two decompose a result into
  its trend-signal and volatility-scaling contributions.
- `tf.data.eligibility`: point-in-time universe membership from `min_history`,
  `max_stale_days`, `entry_lag`, `evaluation_frequency`, and listing dates,
  evaluated only on data available at each evaluation date, plus
  `thin_universe_fraction` for reporting how often the universe was too small
  to be diversified.
- `data.allow_partial_history`, which keeps instruments whose history starts
  part-way through the window. Without it the engine truncates the backtest to
  the intersection of all histories, so one late listing discards years of data
  from everything else.
- `risk.max_asset_weight`, capping any one instrument's share of gross exposure.

### Changed
- Updated project metadata, documentation, and spec to reflect full signal coverage and current maintainer contacts.
- Report modules now select the Agg matplotlib backend. Importing `pyplot` chose
  a GUI backend, so three CLI tests failed with `ModuleNotFoundError: _tkinter`
  on any machine without Tk. The package only writes charts to files.
- `api` now forwards a default price seed (42) when `backtest.seed` is absent.
  It previously forwarded `None`, making synthetic runs irreproducible.

### Fixed
- **Lookahead bias in position sizing.** `_compute_volatility` back-filled the
  volatility warm-up window with the first estimate computed from data that had
  not yet arrived. The warm-up is now masked, so an instrument holds no position
  until its volatility estimate is warm. This changes backtest results in the
  first `min_vol_periods` observations; pass `vol_warmup="bfill"` to reproduce
  the old numbers.
- **`channel_breakout` could not detect a breakout.** Its rolling window
  included the current bar, so price was always inside the channel and the raw
  ratio was confined to [-0.5, +0.5], bounded near 0.165 after the tanh, an
  order of magnitude below what momentum emits. It is a channel-position
  oscillator and is now documented as one; use `donchian_breakout` for an entry
  rule. `price_vs_sma` is likewise documented as an SMA-versus-SMA comparison
  rather than price versus a moving average, which its name implies.
- **Silent substitution of synthetic data for failed real data.** `ingest` caught
  every exception, including data-quality validation failures, and returned
  generated prices with only a log warning, so a malformed CSV became fabricated
  results. Validation errors now propagate; only vendor and network failures fall
  back, and `strict=True` disables the fallback outright.

## [0.9.0] - 2024-05-07

### Added
- Data quality audits covering trading suspensions, holiday gaps and limit-up/down days.
- User guides for onboarding instruments, understanding roll mechanics and calibrating the cost model.
- Version constant exported via `tf.__version__` for downstream tooling.

### Changed
- README expanded with CLI/API guidance, QA workflow and release process.
- Dependencies pinned in `pyproject.toml` to guarantee reproducible installations.

## [0.8.0] - 2024-04-15

### Added
- End-to-end CLI (`run`, `report`, `sweep`, `walkforward`).
- Python API helpers (`run_backtest`, `run_parameter_sweep`, `run_walk_forward`).
- Example notebooks for quick start, parameter studies and attribution analysis.
