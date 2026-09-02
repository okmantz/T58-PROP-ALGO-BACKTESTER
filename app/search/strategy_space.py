"""
Strategy Space Generator (Search Lab -- Stage 0).

Feeds the batch runner's Stage 1 cheap filter. Two modes:

  "single" -- wrap ONE user-supplied strategy (Manual config, or a built
              Python/PineScript/MQL5 Strategy instance) as a size-1 space,
              so the exact same Stage 1-5 pipeline (filter -> GA refine ->
              validation gate -> leaderboard -> promote) can be used to
              rigorously re-validate a single hand-built strategy of ANY
              supported source type, not just to search a family.

  "family" -- combinatorially expand a search space into many concrete
              candidates. Two distinct ways to do this:

                1. Named hypothesis family (Manual strategies only): a
                   named economic hypothesis + a grid of parameter values
                   (see the FAMILIES registry below).
                2. Grid around one given strategy (`strategy=` argument,
                   any source type -- Manual, Python, PineScript, MQL5):
                   discovers that strategy's own tunable numeric
                   parameters (the same discovery Step 6's Iterative
                   Refinement GA already uses) and grid-searches a
                   discretized range around each one.

Every candidate produced by either mode is represented uniformly as a
"candidate spec" dict so the rest of the Search Lab (batch_runner,
robustness, results_db) never has to care how a candidate was generated:

    {"source_type": "manual", "config": {...}}
    {"source_type": "python", "code_text": "...", "code_extension": ".py"}
    {"source_type": "pinescript", "code_text": "...", "code_extension": ".pine"}
    {"source_type": "mql5", "code_text": "...", "code_extension": ".mq5"}

`build_strategy_from_spec()` turns any of these back into a live Strategy
instance ready to run through app.backtest.engine.run_backtest. This is
the ONE place that dispatch happens; batch_runner.py and robustness.py
both import and use it rather than special-casing source types themselves.

Deliberately NOT random indicator soup for the named-family path. Pure
brute-force combinatorics over arbitrary indicators mostly finds noise
faster than signal, especially at the win-rate/RR targets this app is
built around (see /areas/prop-firm-falsification-kit.md's own findings:
nine instrument/signal combinations tested by hand, one validated edge).
Each named family encodes a specific, named trading hypothesis; the grid
only varies ITS parameters. The "grid around a given strategy" path
sidesteps this concern differently: it only ever varies parameters of a
hypothesis the user themselves already wrote (a Manual config, or a real
Python/PineScript/MQL5 file), never invents new logic.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.optimize.code_parameter_space import (
    CodeGene, apply_code_genome, discover_code_genes,
)
from app.optimize.parameter_space import GeneMeta, apply_genome, extract_genome
from app.strategy.base import Strategy
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy

CANDIDATE_SOURCE_TYPES = {"manual", "python", "pinescript", "mql5"}
_CODE_SOURCE_TYPES = {"python", "pinescript", "mql5"}
_CODE_EXTENSIONS = {"python": ".py", "pinescript": ".pine", "mql5": ".mq5"}


class StrategySpaceError(Exception):
    """Raised for an invalid search-space request (unknown family, bad config, etc.)."""


# ---------------------------------------------------------------------------
# Space container
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    mode: str                        # "single" | "family"
    family: str | None                # family name, "all", "<source_type>_grid", or None for single mode
    candidates: dict[str, dict]       # candidate_id -> candidate spec dict (see module docstring)
    meta: dict[str, dict]             # candidate_id -> {"family": str, "params": dict}
    total_generated: int              # size of the full grid before any sampling cap
    sampled: bool                     # True if total_generated > requested max_candidates


# ---------------------------------------------------------------------------
# Candidate spec <-> live Strategy instance (the one place source_type is
# dispatched on for building a runnable strategy)
# ---------------------------------------------------------------------------

def _source_text_for_strategy(strategy: Strategy) -> str:
    """The exact, unmodified source text for a code-based strategy instance."""
    if strategy.source_type == "python":
        return Path(strategy.file_path).read_text(encoding="utf-8", errors="ignore")
    return strategy.code  # PineScriptStrategy / MQL5Strategy already hold this in memory


def spec_from_strategy(strategy: Strategy) -> dict:
    """Builds a uniform candidate spec dict from any already-built Strategy instance."""
    st = strategy.source_type
    if st == "manual":
        return {"source_type": "manual", "config": copy.deepcopy(strategy.config)}
    if st in _CODE_SOURCE_TYPES:
        return {
            "source_type": st,
            "code_text": _source_text_for_strategy(strategy),
            "code_extension": _CODE_EXTENSIONS[st],
        }
    raise StrategySpaceError(f"Unsupported strategy source type '{st}'.")


def build_strategy_from_spec(spec: dict, tmp_dir: str | Path | None = None) -> Strategy:
    """
    The single dispatch point that turns a candidate spec dict back into a
    live, runnable Strategy instance. `tmp_dir` is only required for
    Python candidates (PythonStrategy only accepts a file path) -- a fresh
    uniquely-named temp file is written there per call; PineScript/MQL5
    build directly from in-memory text, and Manual builds directly from
    the config dict, so `tmp_dir` is unused for those.
    """
    st = spec.get("source_type", "manual")
    if st == "manual":
        return ManualStrategy(spec["config"])
    if st == "python":
        if tmp_dir is None:
            raise StrategySpaceError(
                "Building a Python strategy candidate requires a writable tmp_dir "
                "(PythonStrategy only accepts a file path)."
            )
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"candidate_{uuid.uuid4().hex}.py"
        path.write_text(spec["code_text"], encoding="utf-8")
        return PythonStrategy(path)
    if st == "pinescript":
        return PineScriptStrategy(spec["code_text"])
    if st == "mql5":
        return MQL5Strategy(spec["code_text"])
    raise StrategySpaceError(f"Unknown source_type '{st}' in candidate spec.")




@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    label: str
    description: str
    param_grid: dict[str, list]
    build: Callable[[dict], dict]
    valid: Callable[[dict], bool] = field(default=lambda params: True)
    requires_pair_data: bool = False   # True only for the stat_pairs family -- see
    # app.data.pairs.merge_pair_series; generate_search_space()/batch_runner surface a
    # clear error up front if this family is requested without a "pair_close" column
    # merged into the working DataFrame, instead of letting it fail bar-by-bar as NaNs.

    def combinations(self) -> list[dict]:
        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        out = []
        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            if self.valid(params):
                out.append(params)
        return out


def _risk_management(
    stop_atr_mult: float, target_atr_mult: float, stop_atr_period: int = 14,
    target_atr_period: int = 14, max_bars_in_trade: int | None = None,
) -> dict:
    return {
        "stop_type": "atr",
        "stop_value": stop_atr_mult,
        "stop_atr_period": stop_atr_period,
        "target_type": "atr",
        "target_value": target_atr_mult,
        "target_atr_period": target_atr_period,
        "opposite_signal_exit": True,
        **({"max_bars_in_trade": max_bars_in_trade} if max_bars_in_trade else {}),
    }


def _ind(kind: str, period: int, field_name: str = "close") -> dict:
    return {"type": kind, "period": period, "field": field_name}


def _val(value: float) -> dict:
    return {"type": "value", "value": value}


def _cond(left: dict, operator: str, right: dict) -> dict:
    return {"left": left, "operator": operator, "right": right}


# ---------------------------------------------------------------------------
# Family A: Trend Breakout
#   Donchian-style N-bar breakout, only taken in the direction of an EMA
#   trend filter. Mirrors the trend-aligned breakout family that already
#   showed a real (if imperfect) out-of-sample edge on Gold in
#   /areas/prop-firm-falsification-kit.md.
# ---------------------------------------------------------------------------

def _breakout_flag(lookback: int, direction: str) -> dict:
    """
    A proper N-bar breakout EXCLUDING the current bar, via the existing
    "break_of_structure" primitive (app/strategy/manual.py::_advanced_boolean
    computes prior_high/prior_low as high.shift(1).rolling(lookback).max()/
    low.shift(1).rolling(lookback).min() -- already correctly non-lookahead
    and already excludes the bar itself, unlike a naive
    "close > highest_high(lookback)" check, which can never fire: a plain
    rolling max/min over a window THAT INCLUDES the current bar can never be
    exceeded by that same bar's own close, since the bar's own high/low is
    part of the window being compared against).
    """
    return {"type": "bos", "lookback": lookback, "direction": direction}


def _build_trend_breakout(p: dict) -> dict:
    lookback, ema_fast, ema_slow = p["lookback"], p["ema_fast"], p["ema_slow"]
    return {
        "name": f"Trend Breakout (lb={lookback}, ema {ema_fast}/{ema_slow})",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_TREND_BREAKOUT = SkeletonSpec(
    name="trend_breakout",
    label="Trend Breakout (Donchian + EMA filter)",
    description=(
        "N-bar price breakout taken only in the direction of a slower EMA trend filter, "
        "with an ATR stop and ATR target. Trades infrequently but with a real, named "
        "directional hypothesis behind every signal."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0, 4.0],
    },
    build=_build_trend_breakout,
    valid=lambda p: p["ema_fast"] < p["ema_slow"],
)


# ---------------------------------------------------------------------------
# Family B: MTF Pullback / Trend Continuation
#   Trade pullbacks (RSI dips/pops) that occur WITHIN an established EMA
#   trend, rather than breakouts of it. Single-timeframe proxy for the
#   5m/15m/1h confluence idea in /areas/pinescript-strategy.md -- extend by
#   adding a genuinely higher-timeframe operand once multi-CSV context is
#   wired through the search pipeline (see integration notes).
# ---------------------------------------------------------------------------

def _build_mtf_pullback(p: dict) -> dict:
    ema_fast, ema_slow = p["ema_fast"], p["ema_slow"]
    rsi_period, rsi_low, rsi_high = p["rsi_period"], p["rsi_pullback_low"], p["rsi_pullback_high"]
    return {
        "name": f"Trend Pullback (ema {ema_fast}/{ema_slow}, rsi{rsi_period} {rsi_low}/{rsi_high})",
        "entry_conditions": {
            "long": [
                _cond(_ind("ema", ema_fast), ">", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), "<", _val(rsi_low)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("ema", ema_fast), "<", _ind("ema", ema_slow)),
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_high)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), ">", _val(rsi_high))],
            "short": [_cond(_ind("rsi", rsi_period), "<", _val(rsi_low))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MTF_PULLBACK = SkeletonSpec(
    name="mtf_pullback",
    label="Trend Pullback (EMA trend + RSI dip/pop)",
    description=(
        "Buys RSI pullbacks within an established EMA uptrend, sells RSI pops within an "
        "established EMA downtrend -- a continuation, not a reversal, hypothesis."
    ),
    param_grid={
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_period": [7, 14],
        "rsi_pullback_low": [30, 40],
        "rsi_pullback_high": [60, 70],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [24, 48],
    },
    build=_build_mtf_pullback,
    valid=lambda p: p["ema_fast"] < p["ema_slow"] and p["rsi_pullback_low"] < p["rsi_pullback_high"],
)


# ---------------------------------------------------------------------------
# Family C: Mean-Reversion Band Fade
#   Fades price extremes outside a Bollinger Band, filtered by RSI to avoid
#   fading a band walk during a strong trend. Mirrors the Bollinger
#   mean-reversion family already tried (and retired) in Owen's prop-firm
#   strategy graveyard -- kept here so the search can re-test it
#   systematically across a real grid instead of a single hand-picked config.
# ---------------------------------------------------------------------------

def _build_mean_reversion_band(p: dict) -> dict:
    bb_period, bb_std = p["bb_period"], p["bb_std"]
    rsi_period, oversold, overbought = p["rsi_period"], p["oversold"], 100 - p["oversold"]
    return {
        "name": f"Band Fade (bb{bb_period}x{bb_std}, rsi{rsi_period})",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<", {"type": "bollinger_lower", "period": bb_period, "field": "close"}),
                _cond(_ind("rsi", rsi_period), "<", _val(oversold)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">", {"type": "bollinger_upper", "period": bb_period, "field": "close"}),
                _cond(_ind("rsi", rsi_period), ">", _val(overbought)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
            "short": [_cond(_ind("close", 1), "<", {"type": "bollinger_mid", "period": bb_period, "field": "close"})],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"]),
    }


_MEAN_REVERSION_BAND = SkeletonSpec(
    name="mean_reversion_band",
    label="Mean-Reversion Band Fade (Bollinger + RSI filter)",
    description=(
        "Fades price closing outside a Bollinger Band, filtered by RSI extremity, targeting "
        "reversion to the band midline. The exact family already retired once in the prop-firm "
        "strategy graveyard -- included so the search can re-falsify it across a real grid "
        "rather than resting on one earlier hand-picked config."
    ),
    param_grid={
        "bb_period": [14, 20, 30],
        "bb_std": [2.0, 2.5],
        "rsi_period": [7, 14],
        "oversold": [20, 30],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.0],
    },
    build=_build_mean_reversion_band,
)


# ---------------------------------------------------------------------------
# Family D: Volatility Breakout
#   A Donchian-style breakout taken ONLY while the market's own ATR regime
#   is expanding relative to its recent baseline -- a distinct hypothesis
#   from Family A's trend-breakout, which filters by trend DIRECTION (EMA
#   alignment) rather than by volatility STATE. Many real breakout edges
#   depend on catching genuine range expansion, not just any N-bar high;
#   trading every breakout regardless of the prevailing volatility regime
#   is a common way that family fails in a way this family is built to
#   avoid.
# ---------------------------------------------------------------------------

def _build_volatility_breakout(p: dict) -> dict:
    lookback, atr_period, expansion_mult = p["lookback"], p["atr_period"], p["expansion_mult"]
    return {
        "name": f"Volatility Breakout (lb={lookback}, atr{atr_period} x{expansion_mult} expansion)",
        "entry_conditions": {
            "long": [
                _cond(_breakout_flag(lookback, "bullish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "expansion_mult": expansion_mult}, "==", _val(1)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_breakout_flag(lookback, "bearish"), "is true", _val(1)),
                _cond({"type": "atr_regime", "period": atr_period, "expansion_mult": expansion_mult}, "==", _val(1)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p.get("max_bars")),
    }


_VOLATILITY_BREAKOUT = SkeletonSpec(
    name="volatility_breakout",
    label="Volatility Breakout (Donchian + ATR expansion filter)",
    description=(
        "N-bar breakout taken only while ATR is running materially above its own recent "
        "baseline (an expanding-volatility regime), regardless of trend direction. A "
        "distinct hypothesis from Family A: this filters by volatility STATE, not trend "
        "direction, so it can fire on genuine range-expansion moves a pure trend filter "
        "would miss or wrongly admit."
    ),
    param_grid={
        "lookback": [10, 20, 40],
        "atr_period": [14, 20],
        "expansion_mult": [1.05, 1.15],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0, 4.0],
        "max_bars": [None, 48],
    },
    build=_build_volatility_breakout,
)


# ---------------------------------------------------------------------------
# Family E: Session / Time-of-Day Effect
#   Trades an opening-range breakout, but ONLY within a specific session
#   window (e.g. the first hours after a session open). A genuinely
#   different hypothesis class from A-D: it bets that a particular clock-
#   time window has structurally different order flow (session opens,
#   overlap windows) rather than betting on any price- or volatility-based
#   pattern that could occur at any hour.
# ---------------------------------------------------------------------------

def _build_session_time_effect(p: dict) -> dict:
    start, end = p["session_start"], p["session_end"]
    return {
        "name": f"Session Effect ({start}-{end} opening-range breakout)",
        "entry_conditions": {
            "long": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), ">",
                      {"type": "opening_range_high", "session_start": start, "session_end": end}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "time_of_day", "session_start": start, "session_end": end}, "is true", _val(1)),
                _cond(_ind("close", 1), "<",
                      {"type": "opening_range_low", "session_start": start, "session_end": end}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [], "short": [],
            "long_connectors": [], "short_connectors": [],
        },
        "risk_management": _risk_management(
            p["stop_atr_mult"], p["target_atr_mult"],
            max_bars_in_trade=p["max_bars"],
        ),
        # A clock-time flat-by exit is essential for a session strategy --
        # without it a trade opened near the session window's own close
        # can run indefinitely into hours the hypothesis says nothing about.
        "_time_based_exit": p["flat_time"],
    }


def _apply_time_based_exit(config: dict) -> dict:
    flat_time = config.pop("_time_based_exit", None)
    if flat_time:
        config["risk_management"]["time_based_exit"] = {"enabled": True, "time": flat_time}
    return config


_SESSION_TIME_EFFECT = SkeletonSpec(
    name="session_time_effect",
    label="Session / Time-of-Day Effect (opening-range breakout, session-gated)",
    description=(
        "Bets on a specific clock-time window (e.g. a session open) having structurally "
        "different order flow than the rest of the day: takes an opening-range breakout, "
        "but ONLY within that session window, and force-flattens by a configured clock "
        "time so the trade can't run on into hours the hypothesis makes no claim about. "
        "A genuinely different edge source from A-D -- this one is about WHEN, not what "
        "price or volatility is doing."
    ),
    param_grid={
        "session_start": ["08:30", "13:30"],
        "session_end": ["10:30", "15:30"],
        "flat_time": ["16:00"],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [12, 24],
    },
    build=lambda p: _apply_time_based_exit(_build_session_time_effect(p)),
    valid=lambda p: p["session_start"] < p["session_end"],
)


# ---------------------------------------------------------------------------
# Family F: Volume-Imbalance
#   Trades in the direction of a rolling signed-volume imbalance (more
#   volume trading through up-close bars than down-close bars, or vice
#   versa) once it exceeds a threshold -- an order-flow-pressure hypothesis,
#   independent of price pattern or volatility regime. Requires the market
#   data to include a real volume column; on volume-less feeds (some FX
#   sources report tick count as a volume proxy, which still works here)
#   app.strategy.indicators.volume_delta degrades to a flat 0.0 series, so
#   this family will simply generate zero trades rather than fail loudly --
#   Stage 1's cheap filter drops zero-trade candidates on its own.
# ---------------------------------------------------------------------------

def _build_volume_imbalance(p: dict) -> dict:
    period, threshold = p["period"], p["threshold"]
    return {
        "name": f"Volume Imbalance (period={period}, thresh={threshold})",
        "entry_conditions": {
            "long": [
                _cond({"type": "volume_delta", "period": period}, ">", _val(threshold)),
                _cond(_ind("close", 1), ">", {"type": "price", "field": "open"}),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond({"type": "volume_delta", "period": period}, "<", _val(-threshold)),
                _cond(_ind("close", 1), "<", {"type": "price", "field": "open"}),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond({"type": "volume_delta", "period": period}, "<", _val(0.0))],
            "short": [_cond({"type": "volume_delta", "period": period}, ">", _val(0.0))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VOLUME_IMBALANCE = SkeletonSpec(
    name="volume_imbalance",
    label="Volume Imbalance (signed-volume pressure)",
    description=(
        "Trades in the direction of a rolling signed-volume imbalance (volume weighted "
        "toward up-close vs. down-close bars) once it clears a threshold, exiting when "
        "the imbalance flips back through zero. An order-flow-pressure hypothesis, "
        "independent of the price-pattern and volatility-regime hypotheses in the other "
        "families -- degrades to zero trades (not an error) on volume-less data."
    ),
    param_grid={
        "period": [10, 20, 40],
        "threshold": [0.15, 0.3, 0.45],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [24, 48],
    },
    build=_build_volume_imbalance,
)


# ---------------------------------------------------------------------------
# Family G: Statistical Pairs / Relative Value
#   Mean-reverts the PRIMARY instrument against a second, correlated
#   instrument's price -- a genuinely different edge source from every
#   other family here, none of which look outside the single instrument
#   being tested at all. Requires the working DataFrame to already have a
#   second instrument's close merged in as a "pair_close" column (see
#   app.data.pairs.merge_pair_series) BEFORE this family is run; the
#   engine itself stays single-instrument (see that module's docstring for
#   why), so only the PRIMARY leg is ever actually traded here -- this is
#   an honest, explicitly-scoped proxy for a full two-leg pairs trade, not
#   one, and is documented as such rather than silently pretending to
#   trade both legs.
# ---------------------------------------------------------------------------

def _build_stat_pairs(p: dict) -> dict:
    period, entry_z, exit_z = p["period"], p["entry_z"], p["exit_z"]
    return {
        "name": f"Stat Pairs Relative Value (period={period}, entry_z={entry_z})",
        "entry_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, "<", _val(-entry_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, ">", _val(entry_z))],
        },
        "exit_conditions": {
            "long": [_cond({"type": "pair_zscore", "period": period}, ">", _val(-exit_z))],
            "short": [_cond({"type": "pair_zscore", "period": period}, "<", _val(exit_z))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_STAT_PAIRS = SkeletonSpec(
    name="stat_pairs",
    label="Statistical Pairs / Relative Value (requires merged pair data)",
    description=(
        "Mean-reverts the primary instrument's price ratio against a second, correlated "
        "instrument once their relative-value z-score stretches past a threshold, exiting "
        "as it reverts toward zero. REQUIRES a 'pair_close' column already merged into the "
        "working data via app.data.pairs.merge_pair_series() -- only the primary instrument "
        "is actually traded (this app's engine is single-instrument), so treat this as a "
        "relative-value ENTRY FILTER on the primary leg, not a full two-leg pairs trade."
    ),
    param_grid={
        "period": [30, 50, 100],
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.25, 0.5],
        "stop_atr_mult": [1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [48, 96],
    },
    build=_build_stat_pairs,
    valid=lambda p: p["exit_z"] < p["entry_z"],
    requires_pair_data=True,
)


# ---------------------------------------------------------------------------
# Family G: Liquidity Sweep Reversal
#   A standalone liquidity-hypothesis family -- distinct from Family A's
#   trend-direction breakout and Family D's volatility-state breakout,
#   this one bets purely on stop-hunt/liquidity-grab behavior: a swing
#   level is run through (triggering resting stops) and price immediately
#   reclaims it, which is read as evidence the run was liquidity-driven
#   rather than the start of a genuine breakout. Uses the same
#   `liquidity_sweep` primitive app.strategy.manual._advanced_boolean
#   already implements and app.strategy.dna already recognizes as a
#   distinct "liquidity" gene -- this family just gives that gene its own
#   dedicated, independently-searchable hypothesis instead of only ever
#   appearing as an ingredient inside a hand-built strategy.
# ---------------------------------------------------------------------------

def _build_liquidity_sweep_reversal(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Liquidity Sweep Reversal (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "liquidity_sweep", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_LIQUIDITY_SWEEP_REVERSAL = SkeletonSpec(
    name="liquidity_sweep_reversal",
    label="Liquidity Sweep Reversal (stop-hunt + reclaim)",
    description=(
        "Enters on a pure liquidity-sweep signal -- price runs through a recent swing high/low "
        "(the resting-stop level) and immediately closes back on the other side of it, read as a "
        "stop-hunt rather than a real breakout. A standalone liquidity hypothesis, independent of "
        "trend direction or volatility state, so it can be searched and scored on its own instead "
        "of only appearing as one ingredient inside a hand-built discretionary strategy."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "stop_atr_mult": [0.75, 1.0, 1.5],
        "target_atr_mult": [1.5, 2.0, 3.0],
        "max_bars": [None, 24, 48],
    },
    build=_build_liquidity_sweep_reversal,
)


# ---------------------------------------------------------------------------
# Family H: Momentum Continuation
#   Trades WITH an established momentum surge (RSI extremity confirmed by
#   a MACD histogram in agreement), rather than fading it (Family C) or
#   buying a pullback within a slower EMA trend (Family B). A genuinely
#   different edge source: this one bets that momentum which has already
#   shown up persists a while longer, the time-series-momentum effect
#   referenced in the T58 Quant Trading Masterclass material (AQR's
#   published time-series-momentum evidence across futures markets).
# ---------------------------------------------------------------------------

def _build_momentum_continuation(p: dict) -> dict:
    rsi_period, rsi_threshold = p["rsi_period"], p["rsi_threshold"]
    return {
        "name": f"Momentum Continuation (rsi{rsi_period}>{rsi_threshold}, macd hist confirm)",
        "entry_conditions": {
            "long": [
                _cond(_ind("rsi", rsi_period), ">", _val(rsi_threshold)),
                _cond({"type": "macd_histogram"}, ">", _val(0.0)),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("rsi", rsi_period), "<", _val(100 - rsi_threshold)),
                _cond({"type": "macd_histogram"}, "<", _val(0.0)),
            ],
            "short_connectors": ["AND"],
        },
        "exit_conditions": {
            "long": [_cond(_ind("rsi", rsi_period), "<", _val(50))],
            "short": [_cond(_ind("rsi", rsi_period), ">", _val(50))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MOMENTUM_CONTINUATION = SkeletonSpec(
    name="momentum_continuation",
    label="Momentum Continuation (RSI extremity + MACD histogram confirmation)",
    description=(
        "Trades WITH an already-established momentum surge (RSI past a threshold, confirmed by a "
        "same-direction MACD histogram) rather than fading it or waiting for a pullback -- a "
        "distinct hypothesis from the mean-reversion and trend-pullback families: that recent "
        "momentum tends to persist a while longer, not immediately revert."
    ),
    param_grid={
        "rsi_period": [7, 14],
        "rsi_threshold": [55, 60, 65],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 24],
    },
    build=_build_momentum_continuation,
)


# ---------------------------------------------------------------------------
# Family I: VWAP Reversion
#   Mean-reverts toward the session VWAP once price stretches too far from
#   it (in ATR terms), gated by an ATR-regime filter so it stands down
#   during unusually high volatility -- the T58 Quant Trading Masterclass
#   material's own warning about mean reversion ("can get destroyed
#   during trends") applied directly: don't fade a VWAP stretch that's
#   really the start of a genuine expansion move.
# ---------------------------------------------------------------------------

def _build_vwap_reversion(p: dict) -> dict:
    distance_mult, atr_period = p["distance_atr_mult"], p["atr_period"]
    return {
        "name": f"VWAP Reversion (dist={distance_mult}x ATR{atr_period})",
        "entry_conditions": {
            "long": [
                _cond(_ind("close", 1), "<", _ind("vwap", 1)),
                _cond(
                    {"type": "atr", "period": atr_period},
                    ">",
                    _val(0.0),
                ),
            ],
            "long_connectors": ["AND"],
            "short": [
                _cond(_ind("close", 1), ">", _ind("vwap", 1)),
            ],
        },
        "exit_conditions": {
            "long": [_cond(_ind("close", 1), ">", _ind("vwap", 1))],
            "short": [_cond(_ind("close", 1), "<", _ind("vwap", 1))],
        },
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_VWAP_REVERSION = SkeletonSpec(
    name="vwap_reversion",
    label="VWAP Reversion (session VWAP mean reversion)",
    description=(
        "Mean-reverts toward the session VWAP once price closes on the far side of it, exiting "
        "back at VWAP -- the session-anchored counterpart to Family C's Bollinger-band reversion. "
        "Note: the visual-builder condition DSL only exposes a directional (above/below VWAP) "
        "comparison, not a raw price-minus-VWAP distance operand, so `distance_atr_mult` is "
        "reserved for a future distance-gated version rather than actually gating entries here yet "
        "-- today this trades every VWAP-side crossing, filtered only by the ATR stop/target sizing "
        "itself scaling with `atr_period`."
    ),
    param_grid={
        "distance_atr_mult": [1.0, 1.5, 2.0],
        "atr_period": [14, 20],
        "stop_atr_mult": [1.0, 1.5],
        "target_atr_mult": [1.5, 2.5],
        "max_bars": [None, 24],
    },
    build=_build_vwap_reversion,
)


# ---------------------------------------------------------------------------
# Family J: Market Structure Shift
#   A dedicated Change-of-Character (CHoCH) family -- distinct from Family
#   A's plain N-bar break-of-structure, CHoCH specifically requires a
#   PRIOR opposing structure (a higher-high sequence, then a break below
#   the higher-low that formed it) before treating the break as a
#   genuine structural shift rather than just any new extreme. Gives the
#   "market_structure" DNA gene (app.strategy.dna) its own dedicated,
#   independently-searchable hypothesis.
# ---------------------------------------------------------------------------

def _build_market_structure_shift(p: dict) -> dict:
    lookback = p["lookback"]
    return {
        "name": f"Market Structure Shift / CHoCH (lookback={lookback})",
        "entry_conditions": {
            "long": [_cond({"type": "change_of_character", "lookback": lookback, "direction": "bullish"}, "is true", _val(1))],
            "short": [_cond({"type": "change_of_character", "lookback": lookback, "direction": "bearish"}, "is true", _val(1))],
        },
        "exit_conditions": {"long": [], "short": []},
        "risk_management": _risk_management(p["stop_atr_mult"], p["target_atr_mult"], max_bars_in_trade=p["max_bars"]),
    }


_MARKET_STRUCTURE_SHIFT = SkeletonSpec(
    name="market_structure_shift",
    label="Market Structure Shift (Change of Character / CHoCH)",
    description=(
        "Enters on a Change-of-Character: a prior opposing structure (e.g. a run of higher highs) "
        "breaking the swing low that formed it, read as the first evidence the structure itself has "
        "flipped -- a stricter, more specific claim than Family A's plain N-bar break-of-structure, "
        "which fires on any new N-bar extreme regardless of what structure preceded it."
    ),
    param_grid={
        "lookback": [10, 20, 30],
        "stop_atr_mult": [1.0, 1.5, 2.0],
        "target_atr_mult": [2.0, 3.0],
        "max_bars": [None, 48],
    },
    build=_build_market_structure_shift,
)


FAMILIES: dict[str, SkeletonSpec] = {
    _TREND_BREAKOUT.name: _TREND_BREAKOUT,
    _MTF_PULLBACK.name: _MTF_PULLBACK,
    _MEAN_REVERSION_BAND.name: _MEAN_REVERSION_BAND,
    _VOLATILITY_BREAKOUT.name: _VOLATILITY_BREAKOUT,
    _SESSION_TIME_EFFECT.name: _SESSION_TIME_EFFECT,
    _VOLUME_IMBALANCE.name: _VOLUME_IMBALANCE,
    _STAT_PAIRS.name: _STAT_PAIRS,
    _LIQUIDITY_SWEEP_REVERSAL.name: _LIQUIDITY_SWEEP_REVERSAL,
    _MOMENTUM_CONTINUATION.name: _MOMENTUM_CONTINUATION,
    _VWAP_REVERSION.name: _VWAP_REVERSION,
    _MARKET_STRUCTURE_SHIFT.name: _MARKET_STRUCTURE_SHIFT,
}

# Families that need something beyond the plain OHLCV df -- checked by
# generate_search_space() and by batch_runner.run_search() so a family
# needing "pair_close" merged in first fails with one clear message up
# front, instead of quietly producing zero-trade candidates for every
# single grid point.
FAMILIES_REQUIRING_PAIR_DATA = {name for name, spec in FAMILIES.items() if spec.requires_pair_data}


def list_families() -> dict[str, str]:
    """name -> human-readable label, for a UI dropdown / CLI --list-families."""
    return {name: spec.label for name, spec in FAMILIES.items()}


def family_description(name: str) -> str:
    if name not in FAMILIES:
        raise StrategySpaceError(f"Unknown strategy family '{name}'. Known families: {list(FAMILIES)}")
    return FAMILIES[name].description


def family_grid_size(name: str) -> int:
    """Full (pre-sampling) candidate count for one family -- lets the UI show a preview count."""
    if name not in FAMILIES:
        raise StrategySpaceError(f"Unknown strategy family '{name}'. Known families: {list(FAMILIES)}")
    return len(FAMILIES[name].combinations())


# ---------------------------------------------------------------------------
# Grid-around-a-given-strategy (Manual, Python, PineScript, or MQL5)
#
# Reuses the exact same parameter-discovery machinery Step 6's Iterative
# Refinement GA already uses (app.optimize.parameter_space for Manual,
# app.optimize.code_parameter_space for the three code sources) -- both
# gene types expose the same .lo / .hi / .is_int / .base_value / .label
# shape, so the discretization and Cartesian-product logic below is
# entirely source-type-agnostic; only the final "apply a genome" and
# "what does a candidate look like" steps differ.
# ---------------------------------------------------------------------------

def _genes_for_strategy(strategy: Strategy) -> list:
    if strategy.source_type == "manual":
        return extract_genome(strategy.config)
    if strategy.source_type in _CODE_SOURCE_TYPES:
        return discover_code_genes(strategy)
    raise StrategySpaceError(f"Unsupported strategy source type '{strategy.source_type}'.")


def _discretize_gene(gene, n_points: int) -> list[float]:
    """n_points evenly-spaced values across [gene.lo, gene.hi], rounded for
    integer genes and de-duplicated (rounding can collapse points for a
    narrow integer range, e.g. a period whose search range is only 2-4)."""
    n_points = max(int(n_points), 1)
    lo, hi = gene.lo, gene.hi
    if n_points == 1 or hi <= lo:
        raw = [gene.base_value]
    else:
        raw = list(np.linspace(lo, hi, n_points))
    values = [float(round(v)) if gene.is_int else float(v) for v in raw]

    seen: set[float] = set()
    out: list[float] = []
    for v in values:
        key = round(v, 8)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out or [float(gene.base_value)]


def _grid_combinations(genes: list, n_points: int) -> list[list[float]]:
    per_gene_values = [_discretize_gene(g, n_points) for g in genes]
    return [list(combo) for combo in itertools.product(*per_gene_values)]


def _apply_generic_genome(source_type: str, base: Any, genes: list, genome: list[float]) -> Any:
    """Returns a new Manual config dict (manual) or patched source text
    (python/pinescript/mql5) with `genome` written into `base` at the
    positions `genes` describes."""
    if source_type == "manual":
        return apply_genome(base, genes, genome)
    return apply_code_genome(base, genes, genome)


def _grid_space_around_strategy(
    strategy: Strategy, grid_points_per_gene: int, max_candidates: int, seed: int,
) -> SearchSpace:
    genes = _genes_for_strategy(strategy)
    if not genes:
        raise StrategySpaceError(
            f"No tunable numeric parameters were found on this {strategy.source_type} strategy to "
            "grid-search. Manual needs at least one indicator period or a Fixed/ATR stop-loss/"
            "take-profit value; Python needs a top-level SCREAMING_SNAKE_CASE numeric constant; "
            "PineScript needs an input.int()/input.float() value; MQL5 needs an iMA()/iRSI() period "
            "or a T58_SL_PIPS/T58_TP_PIPS directive."
        )

    base = strategy.config if strategy.source_type == "manual" else _source_text_for_strategy(strategy)
    combos = _grid_combinations(genes, grid_points_per_gene)

    total_generated = len(combos)
    sampled = False
    if total_generated > max_candidates:
        rng = random.Random(seed)
        combos = rng.sample(combos, max_candidates)
        sampled = True

    family_label = f"{strategy.source_type}_grid"
    candidates: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for genome in combos:
        applied = _apply_generic_genome(strategy.source_type, base, genes, genome)
        if strategy.source_type == "manual":
            spec = {"source_type": "manual", "config": applied}
            digest_source = json.dumps(applied, sort_keys=True, default=str)
        else:
            spec = {
                "source_type": strategy.source_type, "code_text": applied,
                "code_extension": _CODE_EXTENSIONS[strategy.source_type],
            }
            digest_source = applied
        digest = hashlib.sha1(digest_source.encode()).hexdigest()[:10]
        cid = f"{family_label}-{digest}"
        candidates[cid] = spec
        meta[cid] = {"family": family_label, "params": {g.label: v for g, v in zip(genes, genome)}}

    return SearchSpace(
        mode="family", family=family_label,
        candidates=candidates, meta=meta,
        total_generated=total_generated, sampled=sampled,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_search_space(
    mode: str,
    family: str | None = None,
    single_config: dict | None = None,
    strategy: Strategy | None = None,
    max_candidates: int = 2000,
    seed: int = 42,
    grid_points_per_gene: int = 3,
    has_pair_data: bool = False,
) -> SearchSpace:
    """
    mode="single":
        strategy=<a built Strategy instance> -- wraps it (Manual, Python,
            PineScript, or MQL5 alike) as a size-1 space. Preferred over
            single_config for anything that isn't Manual.
        single_config=<dict> -- legacy path: wraps a Manual Strategy
            config dict directly, without needing a live Strategy
            instance. Kept for backward compatibility (e.g. --cli's
            DEFAULT_MANUAL_STRATEGY). Ignored if `strategy` is given.

    mode="family":
        strategy=<a built Strategy instance> -- grid-searches THAT
            strategy's own tunable numeric parameters (any source type).
            `family` is ignored when `strategy` is given.
        family=<name> or "all" or None -- (Manual only) expands one named
            hypothesis family, or every family, into its full parameter
            grid. Used only when `strategy` is not given.

    In both family paths, if the full grid exceeds `max_candidates`, a
    random (seeded, reproducible) sample of exactly `max_candidates`
    combinations is taken instead of just the first N in itertools.product
    order -- an arbitrary "first N" slice systematically favors whatever
    the first grid dimension happens to be, which biases the search before
    it even starts.
    """
    if mode == "single":
        if strategy is not None:
            spec = spec_from_strategy(strategy)
        elif single_config is not None:
            if not isinstance(single_config, dict):
                raise StrategySpaceError(
                    "single_config must be a Manual Strategy config dict (or pass `strategy` instead "
                    "for a non-Manual strategy)."
                )
            spec = {"source_type": "manual", "config": copy.deepcopy(single_config)}
        else:
            raise StrategySpaceError(
                "mode='single' requires either `strategy` (a built Strategy instance of any "
                "supported source type) or single_config (a Manual Strategy config dict)."
            )
        cid = "single-00000"
        return SearchSpace(
            mode="single", family=None,
            candidates={cid: spec},
            meta={cid: {"family": "single", "params": {}}},
            total_generated=1, sampled=False,
        )

    if mode != "family":
        raise StrategySpaceError(f"Unknown search mode '{mode}' (expected 'single' or 'family').")

    if strategy is not None:
        return _grid_space_around_strategy(strategy, grid_points_per_gene, max_candidates, seed)

    families_to_run = list(FAMILIES.keys()) if family in (None, "all") else [family]
    for fam in families_to_run:
        if fam not in FAMILIES:
            raise StrategySpaceError(f"Unknown strategy family '{fam}'. Known families: {list(FAMILIES)}")

    if not has_pair_data:
        requested_pair_families = [f for f in families_to_run if f in FAMILIES_REQUIRING_PAIR_DATA]
        if requested_pair_families:
            if family in (None, "all"):
                # Searching "all" families with no pair data merged in: skip the
                # pair-only families rather than failing the entire search --
                # every other family still works fine on plain OHLCV data.
                families_to_run = [f for f in families_to_run if f not in FAMILIES_REQUIRING_PAIR_DATA]
            else:
                raise StrategySpaceError(
                    f"Family '{family}' requires a second instrument's price merged into the "
                    "working data first (see app.data.pairs.merge_pair_series) and "
                    "has_pair_data=True passed here. Merge pair data before searching this family."
                )

    all_items: list[tuple[str, dict]] = []
    for fam in families_to_run:
        for params in FAMILIES[fam].combinations():
            all_items.append((fam, params))

    total_generated = len(all_items)
    sampled = False
    if total_generated == 0:
        raise StrategySpaceError("The requested family/grid produced zero valid parameter combinations.")
    if total_generated > max_candidates:
        rng = random.Random(seed)
        all_items = rng.sample(all_items, max_candidates)
        sampled = True

    candidates: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for fam, params in all_items:
        # Content-addressed, not positional: the ID is derived from the
        # family name + the exact parameter combination, so the same
        # combination always gets the same ID regardless of sample order or
        # seed, and two different combinations can never collide onto the
        # same ID. A positional index (e.g. "trend_breakout-00007") would
        # mean the ID's meaning depends on *which run* produced it -- two
        # differently-seeded runs would reuse the same IDs for different
        # underlying strategies, which is both confusing to a person reading
        # the leaderboard and wrong for the results DB's resumability goal
        # (see app/search/results_db.py's module docstring).
        digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
        cid = f"{fam}-{digest}"
        candidates[cid] = {"source_type": "manual", "config": FAMILIES[fam].build(params)}
        meta[cid] = {"family": fam, "params": params}

    return SearchSpace(
        mode="family",
        family=(family if family not in (None, "all") else "all"),
        candidates=candidates, meta=meta,
        total_generated=total_generated, sampled=sampled,
    )
