"""
Regime Testing.

Every other validation gate in this app (walk-forward, parameter-
neighborhood robustness, the deflated Sharpe ratio) asks some version of
"does this hold up over TIME." None of them ask the related but distinct
question this module asks: does this strategy only actually work in ONE
KIND of market -- and does the backtest window just happen to have been
mostly that kind?

A strategy that is genuinely profitable across calm, normal, and volatile
conditions is regime-stable. A strategy that is only profitable when
volatility is high (a real and common failure mode for breakout systems)
can still pass every walk-forward fold and look perfectly robust over
time, if the whole backtest window happened to stay in that one regime --
walking forward through time tells you nothing about whether you've
sampled more than one kind of market at all.

Approach: label every bar into one of `n_regimes` volatility terciles
(realized ATR, relative to the WHOLE dataset's own ATR distribution --
not a rolling/relative-to-recent measure, since the question here is "how
does it do across genuinely different environments", not "how does it do
when volatility is changing", which is a signal-level question the
existing ATR-regime entry filter in app.search.strategy_space's
Volatility Breakout family already answers). Every contiguous historical
stretch belonging to the same regime is backtested independently (a trade
can never span two regimes or two disconnected stretches), and every
stretch sharing a regime label is pooled into one combined statistic for
that regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.indicators import atr

_DEFAULT_REGIME_NAMES = {
    2: ["low_volatility", "high_volatility"],
    3: ["low_volatility", "medium_volatility", "high_volatility"],
}


@dataclass
class RegimeBucketResult:
    label: str
    n_bars: int
    n_segments: int      # number of disconnected historical stretches pooled into this bucket
    n_trades: int
    net_profit: float
    profit_factor: float
    win_rate: float
    is_profitable: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RegimeTestResult:
    buckets: list                  # list[RegimeBucketResult], ordered low- to high-volatility
    n_profitable_buckets: int
    n_buckets: int
    regime_stability_pct: float    # 100 * n_profitable_buckets / n_buckets
    is_regime_stable: bool
    min_segment_bars: int
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "n_profitable_buckets": self.n_profitable_buckets,
            "n_buckets": self.n_buckets,
            "regime_stability_pct": self.regime_stability_pct,
            "is_regime_stable": self.is_regime_stable,
            "min_segment_bars": self.min_segment_bars,
            "notes": self.notes,
        }


def _label_volatility_regimes(df: pd.DataFrame, atr_period: int, n_regimes: int) -> tuple[pd.Series, list[str]]:
    """Returns (per-bar regime name Series, ordered low->high regime names
    actually produced). Falls back to fewer regimes than requested
    (`duplicates='drop'`) on low-variety ATR data rather than raising, so
    a data feed with a lot of repeated ATR values still degrades
    gracefully instead of hard-failing the whole regime test."""
    atr_series = atr(df, period=atr_period)
    if atr_series.dropna().shape[0] < n_regimes * 10:
        raise ValueError("Not enough bars with a valid ATR reading to label volatility regimes.")

    bucketed = pd.qcut(atr_series, q=n_regimes, duplicates="drop")
    categories = list(bucketed.cat.categories)   # qcut always returns these in ascending order
    names = _DEFAULT_REGIME_NAMES.get(
        len(categories), [f"regime_{i + 1}_of_{len(categories)}" for i in range(len(categories))]
    )
    name_map = dict(zip(categories, names))
    return bucketed.map(name_map), names


def _contiguous_segments(mask: pd.Series, min_bars: int) -> list[tuple[int, int]]:
    """Start/end (end exclusive) integer-position ranges of every
    contiguous run of True in `mask` at least `min_bars` long."""
    segments: list[tuple[int, int]] = []
    values = mask.to_numpy()
    in_run, start = False, 0
    for i, v in enumerate(values):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            if i - start >= min_bars:
                segments.append((start, i))
    if in_run and len(values) - start >= min_bars:
        segments.append((start, len(values)))
    return segments


def run_regime_test(
    df: pd.DataFrame,
    strategy_builder,
    risk: RiskConfig,
    n_regimes: int = 3,
    atr_period: int = 14,
    min_segment_bars: int = 100,
    min_profitable_buckets: int | None = None,
) -> "RegimeTestResult | None":
    """
    strategy_builder: zero-arg callable returning a FRESH Strategy
    instance per call (same "always build fresh" convention as
    app.search.robustness.run_walk_forward -- some strategy sources cache
    internal state keyed to the data they last saw).

    min_profitable_buckets: how many of the `n_regimes` buckets must be
    profitable to call the strategy regime-stable. Defaults to
    n_regimes - 1 (e.g. 2 of 3) -- a real edge is allowed to go quiet or
    flat in ONE unfavorable regime; failing in more than one is treated as
    "this strategy only works in a specific kind of market."

    Returns None (not a failure) when there isn't enough data to form
    `n_regimes` buckets of at least `min_segment_bars` bars apiece --
    callers should treat that as "unproven", not "unstable", exactly like
    run_walk_forward's own None case.
    """
    if min_profitable_buckets is None:
        min_profitable_buckets = max(1, n_regimes - 1)

    try:
        regime_labels, ordered_names = _label_volatility_regimes(df, atr_period, n_regimes)
    except ValueError:
        return None

    buckets: list[RegimeBucketResult] = []
    any_segment_found = False
    for label in ordered_names:
        mask = (regime_labels == label).fillna(False)
        segments = _contiguous_segments(mask, min_segment_bars)
        if segments:
            any_segment_found = True
        if not segments:
            buckets.append(RegimeBucketResult(
                label=label, n_bars=0, n_segments=0, n_trades=0,
                net_profit=0.0, profit_factor=0.0, win_rate=0.0, is_profitable=False,
            ))
            continue

        pooled_pnls: list[float] = []
        total_bars = 0
        for start, end in segments:
            segment_df = df.iloc[start:end].reset_index(drop=True)
            bt = run_backtest(segment_df, strategy_builder(), risk)
            total_bars += (end - start)
            pooled_pnls.extend(t.pnl for t in bt.trades)

        gross_profit = sum(p for p in pooled_pnls if p > 0)
        gross_loss = abs(sum(p for p in pooled_pnls if p < 0))
        net_profit = gross_profit - gross_loss
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 10.0 if gross_profit > 0 else 0.0
        wins = sum(1 for p in pooled_pnls if p > 0)
        win_rate = (wins / len(pooled_pnls) * 100.0) if pooled_pnls else 0.0

        buckets.append(RegimeBucketResult(
            label=label, n_bars=total_bars, n_segments=len(segments), n_trades=len(pooled_pnls),
            net_profit=net_profit, profit_factor=profit_factor, win_rate=win_rate,
            is_profitable=bool(pooled_pnls and net_profit > 0),
        ))

    if not any_segment_found:
        # Not one regime produced even a single contiguous stretch long
        # enough to backtest -- there's nothing to conclude here, so this
        # is "unproven" (None), not "every regime failed" (which would
        # unfairly report 0% stability for a candidate that was simply
        # never tested).
        return None

    n_profitable = sum(1 for b in buckets if b.is_profitable)
    n_buckets = len(buckets)
    stability_pct = (n_profitable / n_buckets * 100.0) if n_buckets else 0.0

    notes: list[str] = []
    zero_trade = [b.label for b in buckets if b.n_trades == 0]
    if zero_trade:
        notes.append(
            f"No trades at all in regime(s): {', '.join(zero_trade)} -- counted as not profitable here, "
            "though this may just mean the strategy is naturally inactive in that regime rather than "
            "actively losing money in it."
        )

    return RegimeTestResult(
        buckets=buckets,
        n_profitable_buckets=n_profitable,
        n_buckets=n_buckets,
        regime_stability_pct=stability_pct,
        is_regime_stable=n_profitable >= min_profitable_buckets,
        min_segment_bars=min_segment_bars,
        notes=notes,
    )
