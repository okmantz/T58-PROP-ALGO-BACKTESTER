"""
Strategy DNA -- represents any strategy (Manual config, Python, PineScript,
or MQL5 source) as a small set of ENTRY / EXIT / RISK "genes", so T58 can
compare thousands of strategies structurally instead of only by their
backtest numbers, and answer "what do the strategies that actually work
have in common?" -- not just "which single strategy scored highest."

    ENTRY                        EXIT                    RISK
    +-- market_structure         +-- fixed_rr            +-- fixed_pct
    +-- liquidity                +-- atr                 +-- adaptive
    +-- momentum                 +-- structure           +-- daily_governor
    +-- volatility               +-- time_exit
    +-- time_filter

Deliberately built as a lightweight, code-pattern-agnostic KEYWORD
detector over the strategy's own text (source code for Python/
PineScript/MQL5, or the JSON dump of a Manual Strategy Builder config)
rather than a real parser for four different languages -- the same
tradeoff app.strategy.lookahead_check makes for a different problem.
This means a gene can be a false negative (a strategy that documents an
idea in a way this module doesn't recognize) but essentially never a
false positive (every match is grounded in text that's actually there),
which is the right side to err on for a "what do winners have in
common" tool: missing a gene understates a pattern's true support;
inventing one would fabricate a pattern that doesn't exist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations

# ---------------------------------------------------------------------------
# Gene vocabulary
# ---------------------------------------------------------------------------

# Each keyword is matched case-insensitively as a whole "word-ish" token
# (see _matches) against the strategy's own source/config text. Order
# within a category doesn't matter; a category is "on" if ANY of its
# keywords appears anywhere in the text.
_ENTRY_KEYWORDS: dict[str, list[str]] = {
    "market_structure": [
        "market structure", "order block", "break of structure", "bos", "choch",
        "change of character", "swing high", "swing low", "supply zone", "demand zone",
        "structure_break", "higher_high", "lower_low",
    ],
    "liquidity": [
        "liquidity", "liquidity sweep", "stop hunt", "equal highs", "equal lows",
        "fair value gap", "fvg", "inducement", "sweep_high", "sweep_low",
    ],
    "momentum": [
        "rsi", "macd", "momentum", "stochastic", "rate of change", "roc", "adx",
        "cci", "williams %r",
    ],
    "volatility": [
        "atr", "bollinger", "volatility", "keltner", "std_dev", "standard deviation",
        "expansion", "squeeze", "atr_percentile",
    ],
    "time_filter": [
        "session_start", "session_end", "ny session", "london session", "asia session",
        "trading_hours", "session filter", "time filter", "kill zone", "killzone",
        "hour >=", "hour <=", "hour ==",
    ],
}

_EXIT_KEYWORDS: dict[str, list[str]] = {
    "fixed_rr": [
        "take_profit_pips", "risk_reward", "risk/reward", " rr ", "fixed target",
        "take_profit_distance", "tp_pips", "profit_target_pips",
    ],
    "atr": [
        "atr_multiple", "atr stop", "atr_stop", "atr_multiplier", "atr * ", "atr-based",
    ],
    "structure": [
        "structure exit", "opposite structure", "swing exit", "trailing_stop",
        "trail stop", "structure_based_exit",
    ],
    "time_exit": [
        "max_bars", "time exit", "time_exit", "eod exit", "session end exit",
        "close_at_session_end", "max_bars_in_trade",
    ],
}

_RISK_KEYWORDS: dict[str, list[str]] = {
    "fixed_pct": [
        "risk_value", "risk_pct", "risk_mode", "fixed risk", "% risk", "percent risk",
    ],
    "adaptive": [
        "adaptive_risk", "adaptive risk", "build_limit_aware_preset", "throttle",
        "adaptiveriskconfig", "adaptiveriskrule",
    ],
    "daily_governor": [
        "daily_loss_limit", "max_daily_loss", "daily governor", "daily circuit breaker",
        "daily_loss_limit_pct",
    ],
}

_ALL_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "entry": _ENTRY_KEYWORDS,
    "exit": _EXIT_KEYWORDS,
    "risk": _RISK_KEYWORDS,
}


# ---------------------------------------------------------------------------
# DNA representation
# ---------------------------------------------------------------------------

@dataclass
class StrategyDNA:
    entry: dict[str, bool] = field(default_factory=dict)
    exit: dict[str, bool] = field(default_factory=dict)
    risk: dict[str, bool] = field(default_factory=dict)
    # category.gene -> the actual keyword(s) that matched, for transparency
    # ("why does this strategy show liquidity=True?")
    matched_terms: dict[str, list[str]] = field(default_factory=dict)

    def active_tags(self) -> list[str]:
        """Flattened ['entry.liquidity', 'exit.atr', 'risk.adaptive', ...]
        for every gene that's ON -- this is the representation pattern
        mining operates on."""
        tags = []
        for section_name, section in (("entry", self.entry), ("exit", self.exit), ("risk", self.risk)):
            for gene, on in section.items():
                if on:
                    tags.append(f"{section_name}.{gene}")
        return tags

    def to_dict(self) -> dict:
        return {
            "entry": dict(self.entry), "exit": dict(self.exit), "risk": dict(self.risk),
            "matched_terms": {k: list(v) for k, v in self.matched_terms.items()},
            "active_tags": self.active_tags(),
        }

    def render_tree(self, name: str = "") -> str:
        """ASCII tree in the ENTRY/EXIT/RISK shape -- for a report,
        console log, or experiment-memory entry."""
        lines = [name] if name else []

        def _branch(title: str, section: dict[str, bool]):
            lines.append(title)
            keys = list(section.keys())
            for i, gene in enumerate(keys):
                connector = "└──" if i == len(keys) - 1 else "├──"
                mark = "✓" if section[gene] else " "
                lines.append(f"{connector} [{mark}] {gene.replace('_', ' ')}")

        _branch("ENTRY", self.entry)
        _branch("EXIT", self.exit)
        _branch("RISK", self.risk)
        return "\n".join(lines)


def _matches(text: str, keyword: str) -> bool:
    """A keyword matches if it appears in the text as a substring, using
    a loose word-boundary check for single-word keywords (so 'rr' inside
    'array' doesn't false-positive) and a plain substring check for
    multi-word phrases (word boundaries on a phrase add little and cost
    readability)."""
    if " " in keyword.strip():
        return keyword in text
    return re.search(r"(?<![a-z0-9_])" + re.escape(keyword.strip()) + r"(?![a-z0-9_])", text) is not None


def _extract_section(text: str, keywords: dict[str, list[str]]) -> tuple[dict[str, bool], dict[str, list[str]]]:
    section: dict[str, bool] = {}
    matched: dict[str, list[str]] = {}
    for gene, terms in keywords.items():
        hits = [t for t in terms if _matches(text, t)]
        section[gene] = bool(hits)
        if hits:
            matched[gene] = hits
    return section, matched


def extract_dna_from_text(text: str) -> StrategyDNA:
    """Core extractor: works identically on Python/PineScript/MQL5 source
    text or on the JSON-dumped config of a Manual Strategy Builder
    strategy -- see module docstring for why a single text-keyword
    approach was chosen over one parser per language."""
    text_lower = (text or "").lower()
    entry, entry_matched = _extract_section(text_lower, _ENTRY_KEYWORDS)
    exit_, exit_matched = _extract_section(text_lower, _EXIT_KEYWORDS)
    risk, risk_matched = _extract_section(text_lower, _RISK_KEYWORDS)

    matched_terms: dict[str, list[str]] = {}
    for gene, hits in entry_matched.items():
        matched_terms[f"entry.{gene}"] = hits
    for gene, hits in exit_matched.items():
        matched_terms[f"exit.{gene}"] = hits
    for gene, hits in risk_matched.items():
        matched_terms[f"risk.{gene}"] = hits

    return StrategyDNA(entry=entry, exit=exit_, risk=risk, matched_terms=matched_terms)


def extract_dna(strategy_type: str, source_or_config) -> StrategyDNA:
    """Convenience wrapper: pass the strategy's saved source text
    (python/pinescript/mql5) or its Manual Strategy Builder config dict
    -- either way, returns the same StrategyDNA shape.

    `strategy_type`: "python" | "pinescript" | "mql5" | "manual".
    `source_or_config`: a str of source code for the first three, or a
    dict (the Manual config) for "manual".
    """
    if strategy_type == "manual":
        if isinstance(source_or_config, dict):
            text = json.dumps(source_or_config, default=str)
        else:
            text = str(source_or_config)
    else:
        text = str(source_or_config)
    return extract_dna_from_text(text)


# ---------------------------------------------------------------------------
# Cross-strategy pattern mining
# ---------------------------------------------------------------------------

@dataclass
class DNAPattern:
    combo: tuple  # e.g. ("entry.liquidity", "exit.atr", "risk.adaptive")
    support_top: int          # how many top performers have every gene in combo
    support_top_pct: float    # support_top / n_top_performers, as a %
    support_rest: int         # how many non-top-performers have every gene in combo
    support_rest_pct: float
    lift: float                # support_top_pct / support_rest_pct (inf-safe, see below)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["combo"] = list(self.combo)
        return d

    def render_line(self) -> str:
        genes = " + ".join(g.split(".", 1)[1].replace("_", " ") for g in self.combo)
        return (
            f"{genes}  -- in {self.support_top}/{self.support_top_pct:.0f}% of top performers "
            f"vs {self.support_rest}/{self.support_rest_pct:.0f}% of the rest (lift {self.lift:.2f}x)"
        )


def find_common_patterns(
    entries: list[tuple[str, StrategyDNA, bool]],
    min_combo_size: int = 2,
    max_combo_size: int = 4,
    min_support_top: int = 2,
    min_lift: float = 1.2,
    max_results: int = 15,
) -> list[DNAPattern]:
    """The 'compare thousands of strategies and discover what the
    highest-performing ones consistently contain' feature: a simple,
    dependency-free frequent-itemset count (no need to pull in mlxtend
    for a vocabulary this small -- there are at most 5+4+3=12 genes, so
    even combo_size=4 is only a few hundred combinations to check).

    `entries`: (name, dna, is_top_performer) for every strategy being
    compared -- `is_top_performer` is the caller's own definition of
    "worked" (e.g. passed evaluation, or in the top quartile by net
    profit); this module doesn't decide what counts as a winner.

    Returns patterns ranked by lift (how much more common the combo is
    among top performers than among everyone else), filtered to combos
    that clear `min_support_top` top-performer occurrences and
    `min_lift` -- a combo held by only 1 strategy, or exactly as common
    among losers as winners, isn't a discovery.
    """
    top = [dna for _, dna, is_top in entries if is_top]
    rest = [dna for _, dna, is_top in entries if not is_top]
    n_top, n_rest = len(top), len(rest)
    if n_top == 0:
        return []

    all_tags = sorted({tag for _, dna, _ in entries for tag in dna.active_tags()})

    results: list[DNAPattern] = []
    for size in range(min_combo_size, max_combo_size + 1):
        for combo in combinations(all_tags, size):
            combo_set = set(combo)
            support_top = sum(1 for dna in top if combo_set.issubset(dna.active_tags()))
            if support_top < min_support_top:
                continue
            support_rest = sum(1 for dna in rest if combo_set.issubset(dna.active_tags()))
            support_top_pct = support_top / n_top * 100.0
            support_rest_pct = (support_rest / n_rest * 100.0) if n_rest else 0.0
            # Lift compares top-performer prevalence to everyone-else
            # prevalence; when the combo NEVER appears outside the top
            # group, treat that as a strong (but finite) signal rather
            # than a divide-by-zero infinity that would always sort
            # first regardless of how small support_top actually is.
            lift = (support_top_pct / support_rest_pct) if support_rest_pct > 0 else (support_top_pct / 1.0)
            if lift < min_lift:
                continue
            results.append(DNAPattern(
                combo=combo, support_top=support_top, support_top_pct=support_top_pct,
                support_rest=support_rest, support_rest_pct=support_rest_pct, lift=lift,
            ))

    results.sort(key=lambda p: (p.lift, p.support_top), reverse=True)
    return results[:max_results]
