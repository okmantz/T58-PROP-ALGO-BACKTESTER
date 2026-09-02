"""
Regime Survival Matrix.

app.validation.regime_testing already answers one narrow question well:
"if this strategy only ever traded ONE volatility tercile, would it still
be profitable" -- via independent segment-by-segment re-backtesting. This
module answers a different, complementary question a trader actually asks
after a strategy has already passed evaluation: "of the trades this
EXACT strategy actually took, which kind of market condition were the
losers concentrated in -- and can I just gate the strategy off in that
condition going forward."

That's a trade-ATTRIBUTION question, not a re-validation question, so the
approach here is deliberately different and much cheaper: run the
backtest ONCE over the whole dataset (the strategy sees its real,
continuous history, exactly as it would live), classify every BAR into a
regime along four independent dimensions, then attribute each of the
strategy's own trades to whichever regime was active at its entry time
(the same "as-of" attribution app.ai.research_loop.diagnose_failure
already uses for its single ATR-based check). Segment re-backtesting
(regime_testing.py's approach) is still the right tool for asking "is
this edge real across genuinely different markets" -- this module is the
right tool for "which regimes should this specific, already-built
strategy avoid."

Four classification dimensions, every one computed causally (only ever
looking backward from each bar, never forward):

    trend        -- signed EMA-slope, ATR-normalized, qcut into 5 bins
                    (strong_bearish .. strong_bullish)
    volatility   -- ATR, qcut into 5 bins (very_low .. extreme) -- same
                    "quantile over the WHOLE dataset" convention as
                    regime_testing.py, for the same reason: the question
                    is "how does it do across genuinely different
                    environments actually present in this data", not
                    "how does it do when volatility is currently
                    changing" (a signal-level question, not a regime one)
    session      -- fixed UTC clock-time windows (Asia/London/NY Open/
                    NY/Power Hour) -- see _SESSION_WINDOWS. Assumes
                    timestamps are UTC, matching this app's existing
                    data-import convention (app.data.importer looks for
                    columns literally named "UTC"); DST is not modeled,
                    exactly like the existing session_time_effect family
                    in app.search.strategy_space.
    environment  -- a heuristic 5-way split (trending/ranging/breakout/
                    compression/expansion) from Bollinger Bandwidth level
                    + its own rate of change, crossed with trend strength.
                    This is a labeling heuristic, not a rigorous
                    detector -- see _label_environment for the exact,
                    documented priority order.

Every dimension's classifier is fit ONCE on the historical dataset and
its cut points (RegimeThresholds) are returned so the SAME cut points --
not a fresh quantile split -- can be reapplied to new/live bars later
(see classify_latest_regime). That's what makes "the strategy can simply
turn itself off during bad regimes" an actual, implementable statement
rather than just a backtest curiosity: the cut points a live deployment
would gate on are fit once, in-sample, and then held fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.base import Strategy
from app.strategy.indicators import atr as _atr_ind
from app.strategy.indicators import bollinger, ema

# ---------------------------------------------------------------------------
# Fixed session windows (UTC, no DST) -- ordered by clock hour, covering all
# 24 hours exactly once. Approximate, widely-used retail-trading convention;
# tune to your own feed's timestamp convention if it differs.
# ---------------------------------------------------------------------------
_SESSION_WINDOWS: list[tuple[str, int, int]] = [
    ("asia", 0, 7),
    ("london", 7, 12),
    ("ny_open", 12, 14),
    ("ny", 14, 20),
    ("power_hour", 20, 24),
]

_TREND_NAMES = ["strong_bearish", "weak_bearish", "neutral", "weak_bullish", "strong_bullish"]
_VOL_NAMES = ["very_low", "low", "normal", "high", "extreme"]
_ENV_NAMES = ["compression", "expansion", "breakout", "trending", "ranging"]
_DIMENSION_NAMES = {"trend", "volatility", "session", "environment"}


# ---------------------------------------------------------------------------
# Thresholds -- fit once in-sample, reusable on future/live bars
# ---------------------------------------------------------------------------

@dataclass
class RegimeThresholds:
    trend_edges: list[float]              # qcut bin edges (ascending), ends replaced with +/-inf
    volatility_edges: list[float]
    env_bandwidth_p20: float              # bandwidth value at the 20th percentile (compression cutoff)
    env_bandwidth_delta_p80: float        # bandwidth-change value at the 80th percentile (expansion cutoff)
    env_trend_strength_p50: float         # |trend slope| value at the 50th percentile
    env_trend_strength_p60: float         # |trend slope| value at the 60th percentile
    atr_period: int
    trend_lookback: int
    bandwidth_period: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: dict) -> "RegimeThresholds":
        return RegimeThresholds(**d)


def _edges_from_qcut(series: pd.Series, q: int, names: list[str]) -> tuple[list[float], list[str]]:
    """qcut's bin edges, with the outer edges widened to +/-inf so the
    SAME edges classify values outside the original historical range
    (a live bar more extreme than anything seen in-sample) instead of
    producing NaN. Falls back to fewer bins on low-variety data
    (duplicates='drop'), same convention as regime_testing.py."""
    valid = series.dropna()
    bucketed = pd.qcut(valid, q=q, duplicates="drop")
    categories = list(bucketed.cat.categories)
    edges = [c.left for c in categories] + [categories[-1].right]
    edges[0], edges[-1] = -np.inf, np.inf
    out_names = names if len(categories) == len(names) else [
        f"bucket_{i + 1}_of_{len(categories)}" for i in range(len(categories))
    ]
    return edges, out_names


def _label_with_edges(series: pd.Series, edges: list[float], names: list[str]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=names, include_lowest=True)


# ---------------------------------------------------------------------------
# Per-dimension labeling
# ---------------------------------------------------------------------------

def _trend_slope_normalized(df: pd.DataFrame, lookback: int, atr_period: int) -> pd.Series:
    """Signed, ATR-normalized EMA slope: how many ATRs of directional
    movement per bar, over `lookback` bars. Purely backward-looking."""
    trend_ema = ema(df["close"], period=max(lookback // 2, 2))
    slope = trend_ema.diff(lookback) / float(lookback)
    atr_series = _atr_ind(df, period=atr_period).replace(0.0, np.nan)
    return slope / atr_series


def _label_trend(df: pd.DataFrame, lookback: int, atr_period: int,
                  thresholds: RegimeThresholds | None) -> tuple[pd.Series, list[float]]:
    normalized = _trend_slope_normalized(df, lookback, atr_period)
    if thresholds is not None:
        return _label_with_edges(normalized, thresholds.trend_edges, _TREND_NAMES), thresholds.trend_edges
    edges, names = _edges_from_qcut(normalized, 5, _TREND_NAMES)
    return _label_with_edges(normalized, edges, names), edges


def _label_volatility(df: pd.DataFrame, atr_period: int,
                       thresholds: RegimeThresholds | None) -> tuple[pd.Series, list[float]]:
    atr_series = _atr_ind(df, period=atr_period)
    if thresholds is not None:
        return _label_with_edges(atr_series, thresholds.volatility_edges, _VOL_NAMES), thresholds.volatility_edges
    edges, names = _edges_from_qcut(atr_series, 5, _VOL_NAMES)
    return _label_with_edges(atr_series, edges, names), edges


def _label_session(df: pd.DataFrame) -> pd.Series:
    """Stateless -- fixed clock windows, nothing to fit. Assumes
    df['timestamp'] is UTC (see module docstring)."""
    hours = pd.to_datetime(df["timestamp"]).dt.hour
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, start, end in _SESSION_WINDOWS:
        out[(hours >= start) & (hours < end)] = name
    return out.astype("category")


def label_session(df: pd.DataFrame) -> pd.Series:
    """Public entry point for _label_session -- used standalone by
    app.ai.research_loop.diagnose_failure's session-concentration check,
    which needs just this one dimension rather than the full four-way
    label_regimes() classification."""
    return _label_session(df)


def _label_environment(
    df: pd.DataFrame, bandwidth_period: int, trend_lookback: int, atr_period: int,
    thresholds: RegimeThresholds | None,
) -> tuple[pd.Series, dict]:
    """Heuristic 5-way market-environment split. Priority order (checked
    top to bottom, first match wins) -- documented explicitly because
    this is a labeling heuristic, not a rigorous detector:

      1. compression  -- Bollinger Bandwidth in its own bottom 20th
                          percentile (a volatility squeeze)
      2. breakout      -- bandwidth widening fast (>= 80th percentile of
                          its own period-over-period change) AND trend
                          strength at/above its median (a directional
                          expansion, not just noise widening both ways)
      3. expansion     -- bandwidth widening fast but WITHOUT the
                          directional confirmation above (volatility
                          expanding without yet resolving into a trend)
      4. trending      -- trend strength in its own top 40th percentile
                          (persistent directional movement, whether or
                          not bandwidth is currently expanding)
      5. ranging       -- none of the above (the default/residual case)
    """
    _, bb_mid, _ = bollinger(df["close"], period=bandwidth_period)
    bb_upper, _, bb_lower = bollinger(df["close"], period=bandwidth_period)
    bandwidth = (bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)
    bandwidth_delta = bandwidth.diff(bandwidth_period)
    trend_strength = _trend_slope_normalized(df, trend_lookback, atr_period).abs()

    if thresholds is not None:
        bw_p20 = thresholds.env_bandwidth_p20
        bwd_p80 = thresholds.env_bandwidth_delta_p80
        ts_p50 = thresholds.env_trend_strength_p50
        ts_p60 = thresholds.env_trend_strength_p60
    else:
        bw_p20 = float(np.nanpercentile(bandwidth.dropna(), 20)) if bandwidth.notna().any() else 0.0
        bwd_p80 = float(np.nanpercentile(bandwidth_delta.dropna(), 80)) if bandwidth_delta.notna().any() else 0.0
        ts_p50 = float(np.nanpercentile(trend_strength.dropna(), 50)) if trend_strength.notna().any() else 0.0
        ts_p60 = float(np.nanpercentile(trend_strength.dropna(), 60)) if trend_strength.notna().any() else 0.0

    is_compression = bandwidth <= bw_p20
    is_widening = bandwidth_delta >= bwd_p80
    is_directional = trend_strength >= ts_p50
    is_trending = trend_strength >= ts_p60

    labels = np.select(
        [
            is_compression.fillna(False),
            (is_widening & is_directional).fillna(False),
            is_widening.fillna(False),
            is_trending.fillna(False),
        ],
        ["compression", "breakout", "expansion", "trending"],
        default="ranging",
    )
    # Bars with no valid indicator reading yet (warm-up window) can't be
    # honestly classified -- mark them NaN rather than defaulting to
    # "ranging", which would be a fabricated label.
    unclassifiable = bandwidth.isna() | trend_strength.isna()
    out = pd.Series(labels, index=df.index, dtype="object")
    out[unclassifiable] = pd.NA
    edges = {
        "env_bandwidth_p20": bw_p20, "env_bandwidth_delta_p80": bwd_p80,
        "env_trend_strength_p50": ts_p50, "env_trend_strength_p60": ts_p60,
    }
    return out.astype("category"), edges


def label_regimes(
    df: pd.DataFrame,
    atr_period: int = 14,
    trend_lookback: int = 20,
    bandwidth_period: int = 20,
    thresholds: RegimeThresholds | None = None,
) -> tuple[pd.DataFrame, RegimeThresholds]:
    """Labels every bar along all four dimensions. Pass `thresholds` from
    a prior call's return value to reapply the SAME cut points to new
    data (forward test / live) instead of re-fitting quantiles on it."""
    trend_series, trend_edges = _label_trend(df, trend_lookback, atr_period, thresholds)
    vol_series, vol_edges = _label_volatility(df, atr_period, thresholds)
    session_series = _label_session(df)
    env_series, env_edges = _label_environment(df, bandwidth_period, trend_lookback, atr_period, thresholds)

    labels = pd.DataFrame({
        "trend": trend_series, "volatility": vol_series,
        "session": session_series, "environment": env_series,
    }, index=df.index)

    out_thresholds = thresholds or RegimeThresholds(
        trend_edges=trend_edges, volatility_edges=vol_edges,
        atr_period=atr_period, trend_lookback=trend_lookback, bandwidth_period=bandwidth_period,
        **env_edges,
    )
    return labels, out_thresholds


# ---------------------------------------------------------------------------
# Trade attribution + result shape
# ---------------------------------------------------------------------------

@dataclass
class RegimeCellResult:
    dims: dict            # e.g. {"volatility": "high", "environment": "trending"}
    label: str             # human-readable, e.g. "High Vol / Trending"
    n_bars: int
    pct_of_bars: float
    n_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    return_pct: float       # pooled trades' cumulative pnl / initial_balance * 100, chronological
    max_drawdown_pct: float  # peak-to-trough on that same pooled, regime-isolated equity curve
    is_profitable: bool
    recommend_disable: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _pooled_stats(trades_subset: list, initial_balance: float) -> dict:
    if not trades_subset:
        return {"win_rate": 0.0, "profit_factor": 0.0, "net_profit": 0.0, "return_pct": 0.0, "max_drawdown_pct": 0.0}
    ordered = sorted(trades_subset, key=lambda t: t.exit_time)
    pnls = [t.pnl for t in ordered]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    net_profit = gross_profit - gross_loss
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (10.0 if gross_profit > 0 else 0.0)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100.0

    # A regime-ISOLATED hypothetical equity curve: "if these had been the
    # only trades taken, starting from initial_balance" -- not a slice of
    # the real account curve, since other regimes' trades happened in
    # between chronologically. Clearly documented rather than presented
    # as the real account drawdown.
    cum = np.cumsum(pnls)
    equity = initial_balance + cum
    running_max = np.maximum.accumulate(np.concatenate([[initial_balance], equity]))[1:]
    drawdown = (equity - running_max) / running_max
    max_dd_pct = abs(float(drawdown.min())) * 100.0
    return_pct = float(cum[-1]) / initial_balance * 100.0
    return {
        "win_rate": win_rate, "profit_factor": profit_factor, "net_profit": net_profit,
        "return_pct": return_pct, "max_drawdown_pct": max_dd_pct,
    }


def _attribute_trades(trades: list, labels: pd.DataFrame, df: pd.DataFrame, dim_cols: list[str]) -> dict:
    """Maps each trade to the label(s) active at its entry time (as-of,
    never look-ahead -- searchsorted 'right' then step back one bar,
    same convention as diagnose_failure._atr_at), keyed by a tuple of
    that trade's values across `dim_cols`."""
    ts = pd.to_datetime(df["timestamp"])
    by_key: dict[tuple, list] = {}
    for t in trades:
        idx = ts.searchsorted(pd.Timestamp(t.entry_time), side="right") - 1
        if idx < 0 or idx >= len(labels):
            continue
        row = labels.iloc[idx]
        key = tuple(row[c] for c in dim_cols)
        if any(pd.isna(v) for v in key):
            continue
        by_key.setdefault(key, []).append(t)
    return by_key


def _cell_label(dims: dict) -> str:
    return " / ".join(f"{v.replace('_', ' ').title()}" for v in dims.values())


@dataclass
class RegimeMatrixResult:
    primary_dimensions: tuple
    cells: list                 # list[RegimeCellResult] for the primary (usually 2-D) grouping
    single_dimension: dict      # dim_name -> list[RegimeCellResult], for the other dimensions
    thresholds: RegimeThresholds
    min_trades_for_verdict: int
    disable_pf_threshold: float
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_dimensions": list(self.primary_dimensions),
            "cells": [c.to_dict() for c in self.cells],
            "single_dimension": {k: [c.to_dict() for c in v] for k, v in self.single_dimension.items()},
            "thresholds": self.thresholds.to_dict(),
            "min_trades_for_verdict": self.min_trades_for_verdict,
            "disable_pf_threshold": self.disable_pf_threshold,
            "notes": self.notes,
        }

    def disable_regimes(self) -> list["RegimeCellResult"]:
        """Cells this strategy should simply be gated OFF in -- enough
        trades to trust the number, and a losing profit factor. This is
        the actionable output: feed these `dims` dicts to
        is_regime_disabled() against live/forward classify_latest_regime()
        output to skip taking new signals while in one of them."""
        return [c for c in self.cells if c.recommend_disable]

    def render_table(self) -> str:
        dims_label = " x ".join(d.title() for d in self.primary_dimensions)
        lines = [f"Regime Survival Matrix ({dims_label})", ""]
        header = f"{'Regime':<28}{'Trades':>8}{'Win Rate':>10}{'PF':>7}{'Return':>9}{'DD':>8}  Verdict"
        lines.append(header)
        lines.append("-" * len(header))
        for c in sorted(self.cells, key=lambda c: c.net_profit, reverse=True):
            verdict = "DISABLE" if c.recommend_disable else ("keep" if c.n_trades >= self.min_trades_for_verdict else "unproven")
            lines.append(
                f"{c.label:<28}{c.n_trades:>8}{c.win_rate:>9.0f}%{c.profit_factor:>7.2f}"
                f"{c.return_pct:>+8.1f}%{c.max_drawdown_pct:>7.1f}%  {verdict}"
            )
        disabled = self.disable_regimes()
        if disabled:
            lines.append("")
            lines.append("Recommended OFF regimes: " + ", ".join(c.label for c in disabled))
        return "\n".join(lines)


def is_regime_disabled(current_dims: dict, disabled_cells: list[RegimeCellResult]) -> bool:
    """True if `current_dims` (from classify_latest_regime, restricted to
    the same keys the matrix was built on) matches any disabled cell."""
    for cell in disabled_cells:
        if all(current_dims.get(k) == v for k, v in cell.dims.items()):
            return True
    return False


def classify_latest_regime(
    recent_df: pd.DataFrame, thresholds: RegimeThresholds, dims: tuple = ("trend", "volatility", "session", "environment"),
) -> dict:
    """Labels only the LAST bar of `recent_df` (a recent window, long
    enough to warm up the rolling indicators -- a few hundred bars is
    safe) using previously fit `thresholds`, for a live/forward "should
    I let this strategy trade right now" check. Returns
    {dim_name: label} for the requested `dims`, or None for a dim whose
    warm-up window hasn't filled yet."""
    labels, _ = label_regimes(
        recent_df, atr_period=thresholds.atr_period, trend_lookback=thresholds.trend_lookback,
        bandwidth_period=thresholds.bandwidth_period, thresholds=thresholds,
    )
    last = labels.iloc[-1]
    return {d: (last[d] if pd.notna(last[d]) else None) for d in dims}


def build_regime_matrix(
    df: pd.DataFrame,
    trades: list,
    initial_balance: float,
    dimensions: tuple = ("volatility", "environment"),
    atr_period: int = 14,
    trend_lookback: int = 20,
    bandwidth_period: int = 20,
    thresholds: RegimeThresholds | None = None,
    min_trades_for_verdict: int = 15,
    disable_pf_threshold: float = 1.0,
) -> "RegimeMatrixResult | None":
    """Core, backtest-agnostic entry point: given a df and the trades a
    strategy already produced on it (from a single, ordinary
    run_backtest call -- see run_regime_matrix for the convenience
    wrapper that does that call for you), builds the primary
    `dimensions` cross-tab plus a single-dimension breakdown for every
    OTHER classification dimension.

    Returns None (not an empty result) if there aren't enough valid bars
    to classify any regime at all -- callers should treat that as
    "unproven", exactly like app.validation.regime_testing.run_regime_test.
    """
    for d in dimensions:
        if d not in _DIMENSION_NAMES:
            raise ValueError(f"Unknown regime dimension '{d}'. Known: {sorted(_DIMENSION_NAMES)}")
    if df is None or len(df) < max(atr_period, trend_lookback, bandwidth_period) * 5:
        return None

    labels, fit_thresholds = label_regimes(
        df, atr_period=atr_period, trend_lookback=trend_lookback,
        bandwidth_period=bandwidth_period, thresholds=thresholds,
    )
    total_bars = len(labels)

    def _cells_for(dim_cols: list[str]) -> list[RegimeCellResult]:
        valid = labels[dim_cols].dropna()
        if valid.empty:
            return []
        by_key = _attribute_trades(trades, labels, df, dim_cols)
        present_combos = sorted(set(map(tuple, valid.itertuples(index=False, name=None))))
        counts = valid.groupby(dim_cols, observed=True).size()
        cells = []
        for combo in present_combos:
            dims_dict = dict(zip(dim_cols, combo))
            n_bars = int(counts.get(combo if len(dim_cols) > 1 else combo[0], 0))
            trades_subset = by_key.get(combo, [])
            stats = _pooled_stats(trades_subset, initial_balance)
            n_trades = len(trades_subset)
            recommend_disable = n_trades >= min_trades_for_verdict and stats["profit_factor"] < disable_pf_threshold
            cells.append(RegimeCellResult(
                dims=dims_dict, label=_cell_label(dims_dict), n_bars=n_bars,
                pct_of_bars=(n_bars / total_bars * 100.0) if total_bars else 0.0,
                n_trades=n_trades, win_rate=stats["win_rate"], profit_factor=stats["profit_factor"],
                net_profit=stats["net_profit"], return_pct=stats["return_pct"],
                max_drawdown_pct=stats["max_drawdown_pct"],
                is_profitable=bool(n_trades and stats["net_profit"] > 0),
                recommend_disable=recommend_disable,
            ))
        return cells

    primary_cells = _cells_for(list(dimensions))
    other_dims = [d for d in _DIMENSION_NAMES if d not in dimensions]
    single_dimension = {d: _cells_for([d]) for d in other_dims}

    notes = []
    zero_trade_primary = [c.label for c in primary_cells if c.n_trades == 0]
    if zero_trade_primary:
        notes.append(
            f"No trades attributed to: {', '.join(zero_trade_primary)} -- may mean the strategy is "
            "naturally inactive there rather than actively losing, exactly like a zero-trade bucket in "
            "app.validation.regime_testing."
        )
    thin = [c.label for c in primary_cells if 0 < c.n_trades < min_trades_for_verdict]
    if thin:
        notes.append(
            f"Fewer than {min_trades_for_verdict} trades in: {', '.join(thin)} -- treat as unproven, "
            "not as evidence the regime is bad."
        )

    return RegimeMatrixResult(
        primary_dimensions=tuple(dimensions), cells=primary_cells, single_dimension=single_dimension,
        thresholds=fit_thresholds, min_trades_for_verdict=min_trades_for_verdict,
        disable_pf_threshold=disable_pf_threshold, notes=notes,
    )


def run_regime_matrix(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    dimensions: tuple = ("volatility", "environment"),
    atr_period: int = 14,
    trend_lookback: int = 20,
    bandwidth_period: int = 20,
    min_trades_for_verdict: int = 15,
    disable_pf_threshold: float = 1.0,
) -> "RegimeMatrixResult | None":
    """Convenience wrapper: runs ONE ordinary backtest (the strategy's
    real, continuous history -- deliberately NOT segment-by-segment like
    regime_testing.run_regime_test; see module docstring) and attributes
    its trades to regimes. If you already have a BacktestResult (e.g.
    from the same run driving your leaderboard/report), call
    build_regime_matrix(df, bt_result.trades, risk.initial_balance, ...)
    directly instead of paying for a second backtest."""
    bt_result = run_backtest(df, strategy, risk)
    return build_regime_matrix(
        df, bt_result.trades, risk.initial_balance, dimensions=dimensions,
        atr_period=atr_period, trend_lookback=trend_lookback, bandwidth_period=bandwidth_period,
        min_trades_for_verdict=min_trades_for_verdict, disable_pf_threshold=disable_pf_threshold,
    )
