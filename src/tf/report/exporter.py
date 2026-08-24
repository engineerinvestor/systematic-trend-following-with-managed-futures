"""Reporting helpers to serialise metrics and charts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import pandas as pd
from tabulate import tabulate

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def _format_value(value: float) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _to_dataframe(table: Mapping | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, pd.Series):
        return table.to_frame(name=table.name or "value")
    return pd.Series(table).to_frame(name="value")


def write_html(
    report_dir: Path | str,
    summary: Mapping[str, float],
    *,
    tables: Sequence[tuple[str, Mapping | pd.Series | pd.DataFrame]] | None = None,
    charts: Sequence[tuple[str, str]] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Render an HTML report containing metrics, tables and charts."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [(metric, _format_value(value)) for metric, value in summary.items()]
    summary_table = tabulate(summary_rows, headers=["Metric", "Value"], tablefmt="html")

    extra_tables = ""
    for title, table in tables or []:
        frame = _to_dataframe(table)
        extra_tables += f"<h2>{title}</h2>\n{frame.to_html(classes='dataframe', float_format=lambda x: f'{x:,.4f}')}"

    charts_html = ""
    for title, filename in charts or []:
        charts_html += (
            f"<figure><img src='{filename}' alt='{title}' width='800'/><figcaption>{title}</figcaption></figure>"
        )

    metadata_html = ""
    if metadata:
        metadata_rows = tabulate(metadata.items(), headers=["Key", "Value"], tablefmt="html")
        metadata_html = f"<h2>Run Metadata</h2>\n{metadata_rows}"

    html = f"""
    <html>
      <head>
        <meta charset='utf-8'/>
        <title>Trend Following Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #fafafa; }}
          table {{ border-collapse: collapse; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; }}
          th {{ background-color: #f0f0f0; }}
          figure {{ margin: 24px 0; }}
          figcaption {{ text-align: center; font-style: italic; margin-top: 8px; }}
        </style>
      </head>
      <body>
        <h1>Trend Following Backtest Report</h1>
        {metadata_html}
        <h2>Performance Summary</h2>
        {summary_table}
        {extra_tables}
        <h2>Visualisations</h2>
        {charts_html}
      </body>
    </html>
    """

    report_path = report_dir / "report.html"
    report_path.write_text(html)
    return report_path


def write_pdf(
    report_dir: Path | str,
    summary: Mapping[str, float],
    *,
    tables: Sequence[tuple[str, Mapping | pd.Series | pd.DataFrame]] | None = None,
    charts: Sequence[tuple[str, str]] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Create a PDF companion report composed of the saved charts."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "report.pdf"

    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        y = 0.95
        ax.text(0.02, y, "Trend Following Backtest Report", fontsize=16, weight="bold", transform=ax.transAxes)
        y -= 0.05
        if metadata:
            ax.text(0.02, y, "Metadata:", fontsize=12, weight="bold", transform=ax.transAxes)
            y -= 0.04
            for key, value in metadata.items():
                ax.text(0.04, y, f"{key}: {value}", fontsize=11, transform=ax.transAxes)
                y -= 0.035
        ax.text(0.02, y, "Summary Metrics:", fontsize=12, weight="bold", transform=ax.transAxes)
        y -= 0.04
        for metric, value in summary.items():
            ax.text(0.04, y, f"{metric}: {_format_value(value)}", fontsize=11, transform=ax.transAxes)
            y -= 0.03
        if tables:
            for title, table in tables:
                ax.text(0.02, y, f"{title}:", fontsize=12, weight="bold", transform=ax.transAxes)
                y -= 0.035
                frame = _to_dataframe(table)
                preview = frame.head(10).to_string(float_format="{:.4f}".format)
                for line in preview.splitlines():
                    ax.text(0.04, y, line, fontsize=9, transform=ax.transAxes, family="monospace")
                    y -= 0.025
                y -= 0.01
        pdf.savefig(fig)
        plt.close(fig)

        for title, filename in charts or []:
            chart_path = report_dir / filename
            if not chart_path.exists():
                continue
            image = plt.imread(chart_path)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(title)
            pdf.savefig(fig)
            plt.close(fig)

    return pdf_path
