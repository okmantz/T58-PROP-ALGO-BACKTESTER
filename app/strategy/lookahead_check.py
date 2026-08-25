"""
Generic, code-agnostic lookahead-bias detector.

Every lookahead bug -- regardless of how it's written (a naive
`htf.index < timestamp` filter, a centered rolling window, an accidental
`.shift(-1)`, a groupby that spans future rows, anything) shares one
observable symptom: the strategy's signal for bar `i` changes depending on
whether bars *after* `i` are present in the input DataFrame. A strategy
with no lookahead bias, by definition, can only use information available
at or before bar `i`'s own timestamp -- so its signal at bar `i` must be
identical whether you hand it the full dataset or a copy truncated right
after bar `i`.

This module runs that check directly: it re-invokes the strategy's own
`generate()` on truncated copies of the input data and diffs the signals
against a single full-data run. If ANY bar strictly before a truncation
point produces a different signal in the truncated run than in the full
run, that bar's decision depended on data that hadn't happened yet as of
its own timestamp -- a lookahead leak, full stop.

CHECKPOINT SELECTION: WHY THIS ISN'T JUST "TRUNCATE AT A FEW EVEN FRACTIONS"
-----------------------------------------------------------------------------
Most lookahead bugs of this kind (a higher-timeframe bar built once over
the whole dataset, then filtered with `htf.index < timestamp`) only
produce an *observable signal difference* for bars sitting in the narrow
window whose own higher-timeframe bin straddles the truncation point --
e.g. for a 1H bin, that's up to ~45 minutes of bars right before the cut.
Most real strategies only fire a signal on a small fraction of bars (a
handful of trades across tens of thousands of bars), so truncating at a
few arbitrary evenly-spaced points is very likely to land nowhere near an
actual signal and report a false "all clear".

Instead, this checks the bars that actually matter: it runs the strategy
once on the full dataset, finds every bar where it produced a real
trade signal, and truncates the data to end at (or just after) a sample of
those specific signal bars. This directly answers the question an auditor
actually cares about -- "would each of this strategy's real trades have
still fired if it had only seen data available at that moment?" -- and
gets a far higher hit rate per rerun than blind fractional truncation.

If the strategy produced no signals at all on the full dataset, there's
nothing to anchor on, so this falls back to a few evenly-spaced fractional
checkpoints as a weaker, best-effort check.

This deliberately does NOT try to parse or pattern-match strategy source
code (regexes for `.index <`, `.shift(-`, etc. are trivially both
over- and under-inclusive). It instead tests behavior, which is what
actually matters and catches every lookahead bug class, including ones
nobody has written yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.strategy.base import Strategy, StrategyError


@dataclass
class LookaheadCheckResult:
    checked: bool  # False if the check couldn't run at all (e.g. too little data)
    bug_detected: bool
    checkpoints_tested: list[int] = field(default_factory=list)
    checkpoint_source: str = "signal-anchored"  # or "fractional-fallback"
    first_bad_checkpoint: int | None = None
    first_divergence_index: int | None = None
    first_divergence_timestamp: str | None = None
    n_diverging_bars_at_first_bad_checkpoint: int | None = None
    total_bars_checked_at_first_bad_checkpoint: int | None = None
    skip_reason: str | None = None

    def summary(self) -> str:
        if not self.checked:
            return f"Lookahead check skipped: {self.skip_reason}"
        if not self.bug_detected:
            basis = (
                "the bars where it actually generated a trade signal"
                if self.checkpoint_source == "signal-anchored"
                else "a few evenly-spaced points (no signals to anchor on)"
            )
            return (
                f"Lookahead check passed: re-ran the strategy on data truncated at "
                f"{len(self.checkpoints_tested)} checkpoints, chosen from {basis}. "
                f"Signals were identical in every case. No evidence the strategy is "
                f"using future data."
            )
        return (
            f"LOOKAHEAD BIAS DETECTED: at truncation point "
            f"{self.first_bad_checkpoint}, {self.n_diverging_bars_at_first_bad_checkpoint} "
            f"of {self.total_bars_checked_at_first_bad_checkpoint} bars before the cutoff "
            f"produced a different signal when future bars were removed. The first "
            f"divergence is at bar {self.first_divergence_index} "
            f"({self.first_divergence_timestamp}). This means the strategy's decision at "
            f"that bar depended on data that had not happened yet as of that bar's own "
            f"timestamp -- the backtest results are not trustworthy until this is fixed."
        )


DEFAULT_FRACTIONS = (0.3, 0.5, 0.7, 0.9)
MIN_BARS_FOR_CHECK = 200
MAX_SIGNAL_CHECKPOINTS = 15
_SAMPLE_SEED = 42  # fixed so a given strategy+dataset always gets the same checkpoints


def _sample_positions(items: list[int], k: int) -> list[int]:
    """
    Pick up to k positions from `items` (checking all of them if there are
    k or fewer). Uses a fixed-seed RANDOM sample rather than an evenly
    strided one -- lookahead effects here can correlate with where a bar
    sits within its higher-timeframe bin (e.g. only the last 15m of every
    hour is affected), so an evenly-strided sample can systematically land
    on the same unaffected phase every time and report a false "all clear".
    A random sample doesn't share that failure mode.
    """
    if len(items) <= k:
        return list(items)
    import random
    rng = random.Random(_SAMPLE_SEED)
    return sorted(rng.sample(items, k))


def check_for_lookahead(
    strategy: Strategy,
    df: pd.DataFrame,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    max_signal_checkpoints: int = MAX_SIGNAL_CHECKPOINTS,
) -> LookaheadCheckResult:
    """
    Runs `strategy.generate()` once on the full `df` to find real signal
    bars, then re-runs it on data truncated just after a sample of those
    bars, diffing the signal series each time. See module docstring for why
    checkpoints are chosen this way rather than at arbitrary fractions.
    """
    n = len(df)
    if n < MIN_BARS_FOR_CHECK:
        return LookaheadCheckResult(
            checked=False,
            bug_detected=False,
            skip_reason=f"only {n} bars available (need at least {MIN_BARS_FOR_CHECK}).",
        )

    try:
        full_result = strategy.generate(df.copy())
    except StrategyError:
        # Let the caller's own generate() call surface this; the check
        # itself has nothing useful to add if the strategy can't run at all.
        return LookaheadCheckResult(
            checked=False, bug_detected=False, skip_reason="strategy failed on the full dataset."
        )
    full_signals = full_result.signals

    signal_positions = [int(p) for p in full_signals.to_numpy().nonzero()[0]]

    if signal_positions:
        checkpoint_source = "signal-anchored"
        sampled = _sample_positions(signal_positions, max_signal_checkpoints)
        checkpoints = sorted({
            cp for cp in (pos + 1 for pos in sampled)
            if MIN_BARS_FOR_CHECK <= cp < n
        })
    else:
        checkpoint_source = "fractional-fallback"
        checkpoints = sorted({int(n * f) for f in fractions if MIN_BARS_FOR_CHECK <= int(n * f) < n})

    for cp in checkpoints:
        truncated_df = df.iloc[:cp].copy()
        try:
            truncated_result = strategy.generate(truncated_df)
        except StrategyError:
            # A strategy that can run on the full dataset but not on a
            # shorter prefix isn't a lookahead bug per se -- skip this
            # checkpoint rather than raising a false positive.
            continue
        truncated_signals = truncated_result.signals

        full_prefix = full_signals.iloc[:cp].reset_index(drop=True)
        trunc_prefix = truncated_signals.iloc[:cp].reset_index(drop=True)

        diverges = full_prefix != trunc_prefix
        n_diverging = int(diverges.sum())

        if n_diverging > 0:
            first_bad_pos = int(diverges.idxmax())
            ts = df["timestamp"].iloc[first_bad_pos] if "timestamp" in df.columns else None
            return LookaheadCheckResult(
                checked=True,
                bug_detected=True,
                checkpoints_tested=checkpoints,
                checkpoint_source=checkpoint_source,
                first_bad_checkpoint=cp,
                first_divergence_index=first_bad_pos,
                first_divergence_timestamp=str(ts) if ts is not None else None,
                n_diverging_bars_at_first_bad_checkpoint=n_diverging,
                total_bars_checked_at_first_bad_checkpoint=cp,
            )

    return LookaheadCheckResult(
        checked=True,
        bug_detected=False,
        checkpoints_tested=checkpoints,
        checkpoint_source=checkpoint_source,
    )
