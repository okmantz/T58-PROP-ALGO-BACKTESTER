"""
Strategy Space Generator (Search Lab -- Stage 0).

Feeds the batch runner's Stage 1 cheap filter. Two modes:

  "single" -- wrap ONE user-supplied Manual Strategy config as a size-1
              space, so the exact same Stage 1-5 pipeline (filter -> GA
              refine -> validation gate -> leaderboard -> promote) can be
              used to rigorously re-validate a single hand-built strategy,
              not just to search a family.

  "family" -- combinatorially expand a named strategy family (a economic
              hypothesis + a grid of parameter values) into many concrete
              Manual Strategy configs.

Deliberately NOT random indicator soup. Pure brute-force combinatorics
over arbitrary indicators mostly finds noise faster than signal, especially
at the win-rate/RR targets this app is built around (see
/areas/prop-firm-falsification-kit.md's own findings: nine instrument/signal
combinations tested by hand, one validated edge). Each family below encodes
a specific, named trading hypothesis; the grid only varies ITS parameters.
Adding a new hypothesis means adding one function + one grid here -- nothing
elsewhere in the app needs to change.

Every generated config is a plain, JSON-serializable dict in exactly the
format app.strategy.manual.ManualStrategy already accepts (the same format
the visual builder produces), so:
  - app.optimize.parameter_space.extract_genome() already works on it
  - app.optimize.refinement.run_iterative_refinement() already works on it
  - the normal report pipeline (app.reports.generator) already works on it
Nothing about ManualStrategy itself needs to change for any of this.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable


class StrategySpaceError(Exception):
    """Raised for an invalid search-space request (unknown family, bad config, etc.)."""


# ---------------------------------------------------------------------------
# Space container
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    mode: str                        # "single" | "family"
    family: str | None                # family name, "all", or None for single mode
    candidates: dict[str, dict]       # candidate_id -> Manual Strategy config dict
    meta: dict[str, dict]             # candidate_id -> {"family": str, "params": dict}
    total_generated: int              # size of the full grid before any sampling cap
    sampled: bool                     # True if total_generated > requested max_candidates


# ---------------------------------------------------------------------------
# Skeleton family definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    label: str
    description: str
    param_grid: dict[str, list]
    build: Callable[[dict], dict]
    valid: Callable[[dict], bool] = field(default=lambda params: True)

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


FAMILIES: dict[str, SkeletonSpec] = {
    _TREND_BREAKOUT.name: _TREND_BREAKOUT,
    _MTF_PULLBACK.name: _MTF_PULLBACK,
    _MEAN_REVERSION_BAND.name: _MEAN_REVERSION_BAND,
}


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
# Public entry point
# ---------------------------------------------------------------------------

def generate_search_space(
    mode: str,
    family: str | None = None,
    single_config: dict | None = None,
    max_candidates: int = 2000,
    seed: int = 42,
) -> SearchSpace:
    """
    mode="single": wraps `single_config` (a Manual Strategy config dict,
    e.g. whatever the Strategy tab / CLI already built) as a size-1 space.

    mode="family": expands one named family (or every family, if
    family is None or "all") into its full parameter grid. If the full
    grid exceeds `max_candidates`, a random (seeded, reproducible) sample
    of exactly `max_candidates` combinations is taken instead of just the
    first N in itertools.product order -- an arbitrary "first N" slice
    systematically favors whatever the first grid dimension happens to be,
    which biases the search before it even starts.
    """
    if mode == "single":
        if not single_config or not isinstance(single_config, dict):
            raise StrategySpaceError("mode='single' requires single_config (a Manual Strategy config dict).")
        cid = "single-00000"
        cfg = copy.deepcopy(single_config)
        return SearchSpace(
            mode="single", family=None,
            candidates={cid: cfg},
            meta={cid: {"family": "single", "params": {}}},
            total_generated=1, sampled=False,
        )

    if mode != "family":
        raise StrategySpaceError(f"Unknown search mode '{mode}' (expected 'single' or 'family').")

    families_to_run = list(FAMILIES.keys()) if family in (None, "all") else [family]
    for fam in families_to_run:
        if fam not in FAMILIES:
            raise StrategySpaceError(f"Unknown strategy family '{fam}'. Known families: {list(FAMILIES)}")

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
        candidates[cid] = FAMILIES[fam].build(params)
        meta[cid] = {"family": fam, "params": params}

    return SearchSpace(
        mode="family",
        family=(family if family not in (None, "all") else "all"),
        candidates=candidates, meta=meta,
        total_generated=total_generated, sampled=sampled,
    )
