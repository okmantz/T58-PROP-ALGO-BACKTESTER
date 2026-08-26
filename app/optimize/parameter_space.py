"""
Parameter space extraction for Iterative Refinement.

app.strategy.manual.ManualStrategy takes a plain, JSON-serializable nested
dict (see app/strategy/manual.py and app/ui/main_window.py::_build_strategy
for the exact shape). That happens to make it an ideal substrate for a
genetic-algorithm-style optimizer: every tunable number in a Manual
Strategy -- indicator periods, comparison thresholds, stop/target values,
ATR multiples, trailing-stop distance, break-even trigger, max bars in
trade -- is a plain numeric leaf somewhere in that dict.

This module walks a config dict, finds every leaf whose *key name* is a
known tunable parameter (see GENE_KEY_RULES below), and turns it into a
Gene: a (path, bounds, integer-or-continuous) triple. A "genome" is then
just a flat list of floats, one per gene, in the same order as the gene
list -- which is exactly the representation a GA needs for crossover and
mutation. apply_genome() writes a genome back into a full config dict.

Only Manual Strategy Builder configs are supported. Python / PineScript /
MQL5 strategies are opaque code with no declared parameter schema, so
there is nothing here to walk -- see app/optimize/refinement.py for how
that limitation is surfaced to the user.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass


class RefinementError(Exception):
    """Raised when Iterative Refinement cannot proceed (e.g. no tunable parameters)."""


# ---------------------------------------------------------------------------
# Which dict keys are treated as tunable numeric parameters, and how a
# search range is derived from their current value.
#
#   is_int      -- round mutated/random values to the nearest integer
#   lo_mult /
#   hi_mult     -- for strictly-positive values: search range is
#                  [value * lo_mult, value * hi_mult]
#   min_lo      -- floor for the low end of that range (never search below this)
#   rel_span    -- for values that can be zero/negative (comparison thresholds):
#                  search range is [value - span, value + span] where
#                  span = max(abs(value) * rel_span, abs_span)
#   abs_span    -- absolute floor for rel_span-based spans
#   abs_floor   -- fallback low end when the current value is <= 0 for a
#                  lo_mult/hi_mult-style key (e.g. an unset stop distance)
# ---------------------------------------------------------------------------
GENE_KEY_RULES: dict[str, dict] = {
    "period":            dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "lookback":          dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "stop_atr_period":   dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "target_atr_period": dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "atr_period":        dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "max_bars_in_trade": dict(is_int=True,  lo_mult=0.3, hi_mult=3.0, min_lo=1,   abs_floor=1),
    "stop_value":        dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "target_value":      dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "stop_loss_pips":    dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "take_profit_pips":  dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.1, abs_floor=0.1),
    "trigger_r":         dict(is_int=False, lo_mult=0.3, hi_mult=3.0, min_lo=0.05, abs_floor=0.05),
    # "value" covers both the trailing-stop ATR multiple AND every
    # visual-builder condition's comparison constant (e.g. the "55" in
    # "RSI(14) > 55"), which can legitimately be zero or negative
    # (e.g. "MACD Histogram > 0"), so it gets the relative/absolute-span
    # treatment rather than a pure multiplicative range.
    "value":             dict(is_int=False, rel_span=0.5, abs_span=1.0),
}

# Path-key segments that must never be descended into / mutated even if a
# nested dict happens to reuse one of the names above for something else.
# (Not currently needed by the known config shape, but kept as a guard
# rail for future config fields.)
_EXCLUDED_KEYS = {"name", "description", "author", "version", "market", "type",
                   "field", "direction", "source_type"}


@dataclass(frozen=True)
class GeneMeta:
    path: tuple                # sequence of dict keys / list indices to this value
    kind: str                  # the matched key name, e.g. "period", "stop_value"
    is_int: bool
    lo: float
    hi: float
    base_value: float
    label: str                 # human-readable dotted path, for reports/logs


def _label(path: tuple) -> str:
    parts: list[str] = []
    for key in path:
        if isinstance(key, int):
            parts[-1] = f"{parts[-1]}[{key}]"
        else:
            parts.append(str(key))
    return ".".join(parts)


def _bounds_for(key: str, value: float) -> tuple[float, float, bool]:
    rule = GENE_KEY_RULES[key]
    is_int = bool(rule["is_int"] if "is_int" in rule else False)

    if "rel_span" in rule:
        span = max(abs(value) * rule["rel_span"], rule["abs_span"])
        lo, hi = value - span, value + span
        return lo, hi, is_int

    lo_mult, hi_mult = rule["lo_mult"], rule["hi_mult"]
    min_lo, abs_floor = rule["min_lo"], rule["abs_floor"]

    if value > 0:
        lo = max(value * lo_mult, min_lo)
        hi = max(value * hi_mult, lo + min_lo)
    else:
        lo = float(abs_floor)
        hi = max(float(abs_floor) * 10.0, 10.0)

    if is_int:
        lo = float(max(1, round(lo)))
        hi = float(max(lo + 1, round(hi)))

    return lo, hi, is_int


def extract_genome(config: dict) -> list[GeneMeta]:
    """
    Walk a Manual Strategy config dict and return one GeneMeta per tunable
    numeric leaf found, in a stable (depth-first, dict-insertion) order.
    """
    genes: list[GeneMeta] = []

    def walk(node, path: tuple):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _EXCLUDED_KEYS:
                    continue
                new_path = path + (k,)
                if (
                    k in GENE_KEY_RULES
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and v is not None
                ):
                    lo, hi, is_int = _bounds_for(k, float(v))
                    genes.append(
                        GeneMeta(
                            path=new_path, kind=k, is_int=is_int,
                            lo=lo, hi=hi, base_value=float(v),
                            label=_label(new_path),
                        )
                    )
                walk(v, new_path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path + (i,))

    walk(config, ())
    return genes


def _get_by_path(obj, path: tuple):
    cur = obj
    for key in path:
        cur = cur[key]
    return cur


def _set_by_path(obj, path: tuple, value) -> None:
    cur = obj
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def apply_genome(base_config: dict, genes: list[GeneMeta], genome: list[float]) -> dict:
    """
    Deep-copies base_config and writes each genome value into the path its
    corresponding GeneMeta describes. genome must be the same length and
    order as genes (as returned by extract_genome for this same config).
    """
    if len(genome) != len(genes):
        raise RefinementError(
            f"Genome length ({len(genome)}) does not match the number of "
            f"tunable parameters ({len(genes)})."
        )
    cfg = copy.deepcopy(base_config)
    for gene, raw_value in zip(genes, genome):
        value = int(round(raw_value)) if gene.is_int else float(raw_value)
        _set_by_path(cfg, gene.path, value)
    return cfg
