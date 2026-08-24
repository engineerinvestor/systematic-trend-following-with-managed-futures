"""Plotting utilities for evaluation and reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

# Reports are written to files, never displayed, so the package must not
# require a GUI backend (importing it fails on headless machines and in CI).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def plot_equity(nav: pd.Series, path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    nav.plot(ax=ax, color="#2E86AB", lw=1.5)
    ax.set_title("Equity Curve (NAV)")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(nav: pd.Series, path: Path | str) -> None:
    if nav.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    dd.plot(ax=ax, color="#C0392B", lw=1.5)
    ax.set_title("Underwater Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.fill_between(dd.index, dd, 0, color="#C0392B", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_rolling_stats(
    rolling_stats: pd.DataFrame,
    path: Path | str,
    *,
    window: int,
) -> None:
    if rolling_stats.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rolling_stats.index, rolling_stats["rolling_vol"], color="#8E44AD", label="Rolling Vol")
    ax1.set_ylabel("Volatility", color="#8E44AD")
    ax1.tick_params(axis="y", labelcolor="#8E44AD")

    ax2 = ax1.twinx()
    ax2.plot(rolling_stats.index, rolling_stats["rolling_sharpe"], color="#27AE60", label="Rolling Sharpe")
    ax2.set_ylabel("Sharpe", color="#27AE60")
    ax2.tick_params(axis="y", labelcolor="#27AE60")

    ax1.set_title(f"Rolling {window}-Day Volatility & Sharpe")
    ax1.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_contribution_breakdown(
    totals: pd.Series,
    path: Path | str,
    *,
    title: str,
) -> None:
    if totals.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_totals = totals.sort_values()
    sorted_totals.plot(kind="barh", ax=ax, color="#2471A3")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("PnL Contribution")
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_exposure_timeseries(
    exposures: pd.DataFrame,
    path: Path | str,
    *,
    top_n: int = 5,
) -> None:
    if exposures.empty:
        return

    mean_abs = exposures.abs().mean().sort_values(ascending=False)
    selected = exposures[mean_abs.head(top_n).index].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4))

    def _one_sided(series: pd.Series) -> bool:
        return (series >= 0).all() or (series <= 0).all()

    if all(_one_sided(selected[col]) for col in selected.columns):
        selected.plot.area(ax=ax, alpha=0.7)
    else:
        selected.plot(ax=ax, lw=1.5)

    ax.set_title(f"Top {top_n} Instrument Exposures")
    ax.set_xlabel("Date")
    ax.set_ylabel("Notional Exposure")
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
    values: pd.Series,
    path: Path | str,
    *,
    title: str,
    bins: int = 50,
) -> None:
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="#1ABC9C", edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    path: Path | str,
    *,
    title: str,
) -> None:
    if returns.empty or returns.shape[1] < 2:
        return

    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_curve(
    curve: pd.Series | pd.DataFrame,
    path: Path | str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if isinstance(curve, pd.DataFrame):
        if curve.empty:
            return
        if curve.shape[1] == 0:
            return
        series = curve.iloc[:, 0]
    else:
        series = curve
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series.index, series.values, marker="o", color="#34495E")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(path), bbox_inches="tight")
    plt.close(fig)
