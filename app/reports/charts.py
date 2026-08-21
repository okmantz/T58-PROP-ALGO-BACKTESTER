"""
Minimal, dependency-free SVG chart generation for the HTML report.

No matplotlib / plotting library is required: these functions emit plain
<svg> markup strings that get embedded directly into the self-contained
report.html, so the report stays a single file with no external assets
and no extra Python dependencies.
"""
from __future__ import annotations

import math


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def svg_line_chart(
    values: list[float],
    width: int = 760,
    height: int = 260,
    color: str = "#2f6fed",
    fill: str = "#2f6fed22",
    title: str = "",
    y_label: str = "",
) -> str:
    """A simple single-series line chart (e.g. an equity curve)."""
    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 16, 28
    plot_w = max(width - pad_left - pad_right, 10)
    plot_h = max(height - pad_top - pad_bottom, 10)

    if not values:
        return f'<svg width="{width}" height="{height}"><text x="20" y="20">No data.</text></svg>'

    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1
        hi += 1
    span = hi - lo

    n = len(values)
    step = plot_w / max(n - 1, 1)

    def px(i: int) -> float:
        return pad_left + i * step

    def py(v: float) -> float:
        return pad_top + plot_h - ((v - lo) / span) * plot_h

    points = [(px(i), py(v)) for i, v in enumerate(values)]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = path + f" L {points[-1][0]:.1f},{pad_top + plot_h:.1f} L {points[0][0]:.1f},{pad_top + plot_h:.1f} Z"

    # 4 horizontal gridlines + labels
    gridlines = []
    for i in range(5):
        gy = pad_top + plot_h * i / 4
        gv = hi - span * i / 4
        gridlines.append(
            f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{width - pad_right}" y2="{gy:.1f}" '
            f'stroke="#e3e6ea" stroke-width="1"/>'
            f'<text x="{pad_left - 8}" y="{gy + 4:.1f}" font-size="10" fill="#666" text-anchor="end">{gv:,.0f}</text>'
        )

    zero_line = ""
    if lo < 0 < hi:
        zy = py(0)
        zero_line = (
            f'<line x1="{pad_left}" y1="{zy:.1f}" x2="{width - pad_right}" y2="{zy:.1f}" '
            f'stroke="#b8bdc5" stroke-width="1" stroke-dasharray="3,3"/>'
        )

    title_el = f'<text x="{pad_left}" y="14" font-size="12" font-weight="700" fill="#111">{title}</text>' if title else ""
    ylabel_el = (
        f'<text x="14" y="{pad_top + plot_h / 2:.1f}" font-size="10" fill="#666" '
        f'transform="rotate(-90 14 {pad_top + plot_h / 2:.1f})" text-anchor="middle">{y_label}</text>'
        if y_label else ""
    )

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {title_el}
  {ylabel_el}
  {''.join(gridlines)}
  {zero_line}
  <path d="{area}" fill="{fill}" stroke="none"/>
  <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
</svg>
""".strip()


def svg_histogram(
    values: list[float],
    bins: int = 24,
    width: int = 760,
    height: int = 260,
    color: str = "#2f6fed",
    title: str = "",
    x_label: str = "",
    markers: dict[str, float] | None = None,
) -> str:
    """A histogram of a distribution (e.g. Monte Carlo simulated returns),
    with optional labeled vertical marker lines (e.g. median / p95)."""
    pad_left, pad_right, pad_top, pad_bottom = 40, 16, 16, 34
    plot_w = max(width - pad_left - pad_right, 10)
    plot_h = max(height - pad_top - pad_bottom, 10)

    if not values:
        return f'<svg width="{width}" height="{height}"><text x="20" y="20">No data.</text></svg>'

    lo, hi = min(values), max(values)
    if lo == hi:
        lo -= 1
        hi += 1
    span = hi - lo
    bin_w = span / bins

    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / span * bins)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1

    bars = []
    bar_w = plot_w / bins
    for i, c in enumerate(counts):
        bx = pad_left + i * bar_w
        bh = (c / max_count) * plot_h
        by = pad_top + plot_h - bh
        bars.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bar_w - 1, 0):.1f}" height="{bh:.1f}" fill="{color}" opacity="0.85"/>')

    def px_of_value(v: float) -> float:
        return pad_left + ((v - lo) / span) * plot_w

    marker_els = []
    if markers:
        palette = ["#111111", "#D9A441", "#43D17A", "#F05B63"]
        for i, (label, v) in enumerate(markers.items()):
            if v is None or not (lo <= v <= hi):
                continue
            mx = px_of_value(v)
            col = palette[i % len(palette)]
            marker_els.append(
                f'<line x1="{mx:.1f}" y1="{pad_top}" x2="{mx:.1f}" y2="{pad_top + plot_h}" '
                f'stroke="{col}" stroke-width="1.5" stroke-dasharray="4,3"/>'
                f'<text x="{mx:.1f}" y="{pad_top + plot_h + 14}" font-size="9" fill="{col}" text-anchor="middle">{label}</text>'
            )

    axis_labels = []
    for i in range(5):
        v = lo + span * i / 4
        ax = pad_left + plot_w * i / 4
        axis_labels.append(
            f'<text x="{ax:.1f}" y="{pad_top + plot_h + 26}" font-size="9" fill="#666" text-anchor="middle">{v:,.1f}</text>'
        )

    title_el = f'<text x="{pad_left}" y="14" font-size="12" font-weight="700" fill="#111">{title}</text>' if title else ""
    xlabel_el = (
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" font-size="9" fill="#666" text-anchor="middle">{x_label}</text>'
        if x_label else ""
    )

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {title_el}
  <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" stroke="#ccc" stroke-width="1"/>
  {''.join(bars)}
  {''.join(marker_els)}
  {''.join(axis_labels)}
  {xlabel_el}
</svg>
""".strip()
