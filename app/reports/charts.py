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

    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return f'<svg width="{width}" height="{height}"><text x="20" y="20">No finite values to plot.</text></svg>'
    values = clean

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

    clean = [v for v in values if math.isfinite(v)]
    dropped = len(values) - len(clean)
    if not clean:
        return (
            f'<svg width="{width}" height="{height}"><text x="20" y="20">'
            f'No finite values to plot ({dropped} non-finite value(s) excluded).</text></svg>'
        )
    values = clean

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

    note_el = (
        f'<text x="{pad_left}" y="{height - 4}" font-size="8" fill="#b33">'
        f'{dropped} non-finite simulation(s) excluded from this chart.</text>'
        if dropped else ""
    )

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
  {note_el}
</svg>
""".strip()


def svg_heatmap(
    a_values: list[float],
    b_values: list[float],
    grid: list[list[float]],
    a_label: str = "",
    b_label: str = "",
    width: int = 620,
    height: int = 460,
    title: str = "",
) -> str:
    """
    A 2D parameter-sensitivity heatmap: grid[i][j] is the metric value at
    (a_values[i], b_values[j]). Colored red (worst) -> yellow -> green
    (best) so a person can see at a glance whether a strategy sits on a
    stable plateau or a narrow, fragile ridge.
    """
    pad_left, pad_right, pad_top, pad_bottom = 70, 20, 34, 50
    n_a, n_b = len(a_values), len(b_values)
    if not n_a or not n_b:
        return f'<svg width="{width}" height="{height}"><text x="20" y="20">No data.</text></svg>'

    plot_w = max(width - pad_left - pad_right, 10)
    plot_h = max(height - pad_top - pad_bottom, 10)
    cell_w = plot_w / n_b
    cell_h = plot_h / n_a

    flat = [v for row in grid for v in row if math.isfinite(v)]
    lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)
    if lo == hi:
        lo -= 1
        hi += 1

    def color_for(v: float) -> str:
        if not math.isfinite(v):
            return "#cccccc"
        t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
        # red (low) -> yellow (mid) -> green (high)
        if t < 0.5:
            r, g, b = 217, int(60 + t * 2 * (196 - 60)), 60
        else:
            t2 = (t - 0.5) * 2
            r, g, b = int(217 - t2 * (217 - 60)), int(196 - t2 * (196 - 180) + t2 * 20), 60
        return f"rgb({r},{g},{b})"

    cells = []
    for i in range(n_a):
        for j in range(n_b):
            v = grid[i][j]
            x = pad_left + j * cell_w
            y = pad_top + i * cell_h
            cells.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{color_for(v)}" stroke="#ffffff" stroke-width="1"/>'
            )

    a_axis_labels = []
    for i, av in enumerate(a_values):
        y = pad_top + i * cell_h + cell_h / 2 + 3
        a_axis_labels.append(f'<text x="{pad_left - 6}" y="{y:.1f}" font-size="9" fill="#444" text-anchor="end">{av:g}</text>')

    b_axis_labels = []
    for j, bv in enumerate(b_values):
        x = pad_left + j * cell_w + cell_w / 2
        b_axis_labels.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 14}" font-size="9" fill="#444" '
            f'text-anchor="middle" transform="rotate(45 {x:.1f} {pad_top + plot_h + 14})">{bv:g}</text>'
        )

    title_el = f'<text x="{pad_left}" y="14" font-size="12" font-weight="700" fill="#111">{title}</text>' if title else ""
    a_label_el = (
        f'<text x="14" y="{pad_top + plot_h / 2:.1f}" font-size="10" fill="#666" '
        f'transform="rotate(-90 14 {pad_top + plot_h / 2:.1f})" text-anchor="middle">{a_label}</text>'
        if a_label else ""
    )
    b_label_el = (
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" font-size="10" fill="#666" text-anchor="middle">{b_label}</text>'
        if b_label else ""
    )

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {title_el}
  {''.join(cells)}
  {''.join(a_axis_labels)}
  {''.join(b_axis_labels)}
  {a_label_el}
  {b_label_el}
</svg>
""".strip()


def svg_multi_line_chart(
    series: list[tuple[str, list[float], str]],
    width: int = 760,
    height: int = 260,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> str:
    """
    Multiple labeled line series on shared axes (e.g. best-fitness vs
    mean-fitness per generation, for the Iterative Refinement convergence
    chart). `series` is a list of (label, values, color) tuples; all value
    lists must be the same length (one point per x-position, e.g. one per
    generation).
    """
    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 16, 40
    plot_w = max(width - pad_left - pad_right, 10)
    plot_h = max(height - pad_top - pad_bottom, 10)

    clean_series = []
    for label, values, color in series:
        clean = [v for v in values if math.isfinite(v)]
        if clean:
            clean_series.append((label, values, color, clean))

    if not clean_series:
        return f'<svg width="{width}" height="{height}"><text x="20" y="20">No data.</text></svg>'

    n = max(len(values) for _, values, _, _ in clean_series)
    all_clean = [v for _, _, _, clean in clean_series for v in clean]
    lo, hi = min(all_clean), max(all_clean)
    if lo == hi:
        lo -= 1
        hi += 1
    span = hi - lo
    step = plot_w / max(n - 1, 1)

    def px(i: int) -> float:
        return pad_left + i * step

    def py(v: float) -> float:
        return pad_top + plot_h - ((v - lo) / span) * plot_h

    gridlines = []
    for i in range(5):
        gy = pad_top + plot_h * i / 4
        gv = hi - span * i / 4
        gridlines.append(
            f'<line x1="{pad_left}" y1="{gy:.1f}" x2="{width - pad_right}" y2="{gy:.1f}" '
            f'stroke="#e3e6ea" stroke-width="1"/>'
            f'<text x="{pad_left - 8}" y="{gy + 4:.1f}" font-size="10" fill="#666" text-anchor="end">{gv:,.1f}</text>'
        )

    paths = []
    legend = []
    for i, (label, values, color, _clean) in enumerate(clean_series):
        pts = [(px(j), py(v)) for j, v in enumerate(values) if math.isfinite(v)]
        if len(pts) >= 2:
            path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            paths.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            paths.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
        lx = pad_left + i * 150
        legend.append(
            f'<rect x="{lx}" y="{height - 12}" width="10" height="10" fill="{color}"/>'
            f'<text x="{lx + 14}" y="{height - 3}" font-size="10" fill="#333">{label}</text>'
        )

    title_el = f'<text x="{pad_left}" y="14" font-size="12" font-weight="700" fill="#111">{title}</text>' if title else ""
    ylabel_el = (
        f'<text x="14" y="{pad_top + plot_h / 2:.1f}" font-size="10" fill="#666" '
        f'transform="rotate(-90 14 {pad_top + plot_h / 2:.1f})" text-anchor="middle">{y_label}</text>'
        if y_label else ""
    )
    xlabel_el = (
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{pad_top + plot_h + 22}" font-size="9" '
        f'fill="#666" text-anchor="middle">{x_label}</text>'
        if x_label else ""
    )

    return f"""
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {title_el}
  {ylabel_el}
  {''.join(gridlines)}
  {''.join(paths)}
  {xlabel_el}
  {''.join(legend)}
</svg>
""".strip()
