"""
Interactive Plotly-based trade visualization for the HTML report.

Renders price action for the backtested instrument with a marker for every
trade the strategy took (long/short entries, winning/losing exits, and a
dotted connector between each pair), plus the equity curve on a secondary
axis -- fully pannable and zoomable in the browser, with a range slider for
scanning the whole period.

Design note: this intentionally does NOT depend on the `plotly` Python
package. It builds the same `data` / `layout` structures Plotly.py would
produce, as plain JSON-serializable dicts, and lets Plotly.js (loaded from
its CDN by the report template) render them client-side. That keeps the
Python dependency footprint the same as the rest of app/reports (see
charts.py) and keeps the PyInstaller-frozen desktop build lean. The one
tradeoff: viewing the "Trade Visualization" tab requires an internet
connection to fetch Plotly.js; every other tab/section of the report
remains fully self-contained and offline, exactly as before.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.backtest.execution import Trade

# Plotly renders fine well past this many points, but the report is meant to
# open instantly -- downsample the underlying price/equity lines while
# leaving every individual trade marker untouched (those matter far more
# than the exact shape of the price line between them).
MAX_LINE_POINTS = 6000


def _downsample(timestamps: list, values: list[float], max_points: int) -> tuple[list, list[float]]:
    n = len(values)
    if n <= max_points:
        return timestamps, values
    step = n / max_points
    idx = [int(i * step) for i in range(max_points)]
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [timestamps[i] for i in idx], [values[i] for i in idx]


def _ts(x: Any) -> str:
    """ISO-8601 string -- a format Plotly's date axis parses natively."""
    return pd.Timestamp(x).isoformat()


def _hover_text(t: Trade, role: str) -> str:
    side = "Long" if t.direction == 1 else "Short"
    if role == "entry":
        return f"{side} entry<br>Time: {_ts(t.entry_time)}<br>Price: {t.entry_price:,.5f}"
    outcome = "Win" if t.pnl > 0 else ("Loss" if t.pnl < 0 else "Breakeven")
    return (
        f"{side} exit &mdash; {outcome}<br>"
        f"Time: {_ts(t.exit_time)}<br>"
        f"Price: {t.exit_price:,.5f}<br>"
        f"P&amp;L: ${t.pnl:,.2f} ({t.pnl_pct:,.2f}%)<br>"
        f"Reason: {t.exit_reason}"
    )


def build_trade_chart_payload(
    price_df: pd.DataFrame,
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    instrument: str = "",
) -> dict[str, Any]:
    """Builds the Plotly `data` + `layout` dicts as plain Python structures."""

    ts_all = price_df["timestamp"].tolist()
    close_all = [float(v) for v in price_df["close"].tolist()]
    ts_ds, close_ds = _downsample(ts_all, close_all, MAX_LINE_POINTS)

    long_x, long_y, long_txt = [], [], []
    short_x, short_y, short_txt = [], [], []
    win_x, win_y, win_txt = [], [], []
    loss_x, loss_y, loss_txt = [], [], []
    connector_x, connector_y = [], []

    for t in trades:
        if t.direction == 1:
            long_x.append(_ts(t.entry_time))
            long_y.append(t.entry_price)
            long_txt.append(_hover_text(t, "entry"))
        else:
            short_x.append(_ts(t.entry_time))
            short_y.append(t.entry_price)
            short_txt.append(_hover_text(t, "entry"))

        if t.pnl >= 0:
            win_x.append(_ts(t.exit_time))
            win_y.append(t.exit_price)
            win_txt.append(_hover_text(t, "exit"))
        else:
            loss_x.append(_ts(t.exit_time))
            loss_y.append(t.exit_price)
            loss_txt.append(_hover_text(t, "exit"))

        connector_x.extend([_ts(t.entry_time), _ts(t.exit_time), None])
        connector_y.extend([t.entry_price, t.exit_price, None])

    data: list[dict[str, Any]] = [
        {
            "type": "scattergl",
            "mode": "lines",
            "name": f"{instrument} price".strip() or "Price",
            "x": [_ts(t) for t in ts_ds],
            "y": close_ds,
            "line": {"color": "#2f6fed", "width": 1.1},
            "hoverinfo": "x+y",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Trade path",
            "x": connector_x,
            "y": connector_y,
            "line": {"color": "#b8bdc5", "width": 1, "dash": "dot"},
            "hoverinfo": "skip",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Long entry",
            "x": long_x,
            "y": long_y,
            "text": long_txt,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {"symbol": "triangle-up", "size": 10, "color": "#16a34a",
                       "line": {"width": 1, "color": "#0a5c2a"}},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Short entry",
            "x": short_x,
            "y": short_y,
            "text": short_txt,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {"symbol": "triangle-down", "size": 10, "color": "#D9A441",
                       "line": {"width": 1, "color": "#8a6a1c"}},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Winning exit",
            "x": win_x,
            "y": win_y,
            "text": win_txt,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {"symbol": "circle", "size": 7, "color": "#16a34a",
                       "line": {"width": 1, "color": "#0a5c2a"}},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "name": "Losing exit",
            "x": loss_x,
            "y": loss_y,
            "text": loss_txt,
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {"symbol": "circle", "size": 7, "color": "#dc2626",
                       "line": {"width": 1, "color": "#7a1414"}},
        },
    ]

    if len(equity_curve):
        eq_ts, eq_vals = _downsample(
            equity_curve["timestamp"].tolist(),
            [float(v) for v in equity_curve["equity"].tolist()],
            MAX_LINE_POINTS,
        )
        data.append({
            "type": "scattergl",
            "mode": "lines",
            "name": "Equity",
            "x": [_ts(t) for t in eq_ts],
            "y": eq_vals,
            "yaxis": "y2",
            "line": {"color": "#111827", "width": 1.3},
            "opacity": 0.85,
        })

    layout = {
        "margin": {"t": 10, "l": 60, "r": 64, "b": 40},
        "hovermode": "closest",
        "dragmode": "zoom",
        "showlegend": True,
        "legend": {"orientation": "h", "y": 1.12, "font": {"size": 11}},
        "xaxis": {
            "title": "Time",
            "type": "date",
            "rangeslider": {"visible": True, "thickness": 0.08},
        },
        "yaxis": {"title": "Price", "side": "left"},
        "yaxis2": {"title": "Equity ($)", "overlaying": "y", "side": "right", "showgrid": False},
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "font": {"family": "-apple-system, Segoe UI, Roboto, Arial, sans-serif", "size": 11, "color": "#14161a"},
    }

    return {"data": data, "layout": layout}


def build_trade_chart_html(
    price_df: pd.DataFrame | None,
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    instrument: str = "",
    div_id: str = "t58-trade-chart",
) -> str:
    """Returns the tab body: a chart container plus the payload it needs.

    Rendering itself is deferred to the report's tab-switch JS (see
    generator.py) so Plotly never initializes into a hidden, zero-size
    <div> -- it draws the first time the user actually opens this tab.
    """
    if price_df is None or not len(price_df):
        return '<p class="muted">No price data was retained for this run, so the interactive trade chart is unavailable.</p>'
    if not trades:
        return '<p class="muted">This strategy took no trades over the given data, so there is nothing to plot.</p>'

    payload = build_trade_chart_payload(price_df, trades, equity_curve, instrument=instrument)
    payload_json = json.dumps(payload, default=str)

    return f"""
<p class="muted">Every trade the strategy took, plotted against price -- drag to zoom, use the range slider below the chart to scan the full period, and hover any marker for trade detail. Equity curve is on the right axis.</p>
<div id="{div_id}" style="width:100%; height:640px;"></div>
<script>window.__t58TradeChartPayload = {payload_json};</script>
""".strip()
