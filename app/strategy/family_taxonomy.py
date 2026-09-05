"""
Strategy Family Taxonomy.

app.search.strategy_space already lets the Search Lab search ONE named
hypothesis family at a time, or "all" of them -- the missing piece is a
canonical vocabulary that (a) covers the full research universe Owen
asked for and the T58 Quant Trading Masterclass material itself
recommends (its Lesson 3 family table), and (b) can classify ANY
strategy this app can produce -- not just a app.search.strategy_space
grid candidate, but also an Ollama-generated Python strategy
(app.ai.research_loop), a GA-evolved genome (app.evolution.engine), or a
hand-uploaded Manual/Python/PineScript/MQL5 file -- into one of these
canonical groups, so results from every one of those sources can be
compared and ranked on the same axis. See app.search.family_diversity
for what actually gets BUILT on top of this classification (per-family
ranking, "don't let the optimizer generate 10,000 variants of the same
family" diversity caps).

Two classification paths, tried in order:

  1. Skeleton name -- exact and authoritative. Every named hypothesis
     family in app.search.strategy_space.FAMILIES encodes exactly ONE
     economic hypothesis (see that module's own docstring), so its
     skeleton name maps 1:1 onto a canonical group with total
     confidence -- no inference needed.

  2. Heuristic vote -- for anything that ISN'T a known skeleton (an AI-
     generated code strategy, a "single" mode candidate, a
     "<source>_grid" refinement of a user-supplied strategy, or a
     hand-uploaded file). Combines app.strategy.dna's already-proven,
     false-positive-resistant gene detector (weighted higher) with a
     small additional keyword table (weighted lower) covering the
     canonical groups app.strategy.dna doesn't tag at all -- it has no
     "trend", "mean_reversion", "breakout", "pullback", "vwap", or
     "volatility_contraction" gene, since its vocabulary was built for a
     different question (entry/exit/risk mechanics, not hypothesis
     family). The group with the highest combined score wins; ties break
     by a fixed priority order; no matches at all returns
     "uncategorized" rather than guessing.
"""
from __future__ import annotations

from app.strategy.dna import StrategyDNA, extract_dna

# ---------------------------------------------------------------------------
# Canonical taxonomy -- Owen's 10-family list, merged with the T58 Quant
# Trading Masterclass material's own Lesson 3 family table (which adds
# volatility_contraction, statistical_arbitrage, relative_strength, and
# regime_switching as their own distinct research families).
# ---------------------------------------------------------------------------

FAMILY_GROUPS: dict[str, str] = {
    "trend_following": "Ride a persistent directional move.",
    "momentum": "Recent movement is expected to continue (time-series momentum).",
    "breakout": "Price clears a defined range or level.",
    "mean_reversion": "An extreme move is expected to revert toward a mean/level.",
    "volatility_expansion": "A volatility regime is changing from calm to active.",
    "volatility_contraction": "Trades the move that follows a compression/squeeze.",
    "market_structure": "Structural price transitions (break of structure, CHoCH).",
    "liquidity_sweep": "A stop/liquidity event is expected to produce a move.",
    "opening_range": "A specific clock-time window behaves differently (session/opening range).",
    "vwap": "Reversion to, or continuation from, a volume-weighted anchor price.",
    "pullback": "A temporary counter-trend move inside a larger trend.",
    "statistical_arbitrage": "A relationship between two or more instruments.",
    "relative_strength": "One instrument is expected to outperform another.",
    # Not a single-strategy hypothesis like the others -- populated by
    # COMPOSING several families behind a live regime classifier (see
    # app.validation.regime_matrix, whose whole point is deciding which
    # regime is active so a different family can be allowed to trade in
    # each one), never by a single app.search.strategy_space grid family.
    "regime_switching": "A different strategy/family trades depending on the active market regime.",
    "uncategorized": "Doesn't clearly match any of the above.",
}

# Path 1: exact skeleton-name -> canonical group (authoritative).
_SKELETON_TO_GROUP: dict[str, str] = {
    "trend_breakout": "breakout",
    "mtf_pullback": "pullback",
    "mean_reversion_band": "mean_reversion",
    "volatility_breakout": "volatility_expansion",
    "session_time_effect": "opening_range",
    "volume_imbalance": "momentum",
    "stat_pairs": "statistical_arbitrage",
    "liquidity_sweep_reversal": "liquidity_sweep",
    "momentum_continuation": "momentum",
    "vwap_reversion": "vwap",
    "market_structure_shift": "market_structure",
    "prev_day_range_breakout": "breakout",
    "macd_cross_trend": "trend_following",
    "swing_structure_fade": "mean_reversion",
    "fvg_imbalance_continuation": "liquidity_sweep",
    "order_block_reaction": "market_structure",
    "volatility_contraction_squeeze": "volatility_contraction",
    "overnight_gap_fade": "mean_reversion",
    "wma_ribbon_trend": "trend_following",
    "pct_change_momentum_burst": "momentum",
    "rsi_extreme_reversion": "mean_reversion",
    # Prop-eval-shaped scalp families (app.search.strategy_space Families U-Z)
    "liquidity_sweep_quick_reclaim": "liquidity_sweep",
    "range_midpoint_fade": "mean_reversion",
    "opening_range_retest_confirmation": "opening_range",
    "micro_pullback_continuation": "pullback",
    "change_of_character_reversal_scalp": "market_structure",
    "fvg_quick_fill_fade": "liquidity_sweep",
    # Expansion round 2 (app.search.strategy_space Families added Sep 2026)
    "relative_strength_momentum": "relative_strength",
    "volume_climax_reversal": "mean_reversion",
    "vwap_trend_continuation": "vwap",
    "bollinger_band_walk_continuation": "trend_following",
    "gap_and_go_continuation": "momentum",
    "volume_confirmed_breakout": "breakout",
    "wma_sma_divergence_trend": "trend_following",
    "higher_low_structure_continuation": "pullback",
}

# Path 2a: app.strategy.dna active_tags() -> canonical group. Checked in
# THIS priority order (first match wins) when more than one gene is on --
# liquidity and structure are the most specific/rare signals, so they
# outrank the broader momentum/volatility/time_filter genes.
_DNA_TAG_PRIORITY: list[tuple[str, str]] = [
    ("entry.liquidity", "liquidity_sweep"),
    ("entry.market_structure", "market_structure"),
    ("entry.momentum", "momentum"),
    ("entry.time_filter", "opening_range"),
    ("entry.volatility", "volatility_expansion"),
]
_DNA_WEIGHT = 2  # dna.py's keyword detector essentially never false-positives; trust it more

# Path 2b: a small, independent keyword table for the groups app.strategy.dna
# doesn't tag at all. Same "substring, case-insensitive" convention as
# app.strategy.dna._matches, kept separate rather than added to dna.py so
# this module's own (looser, hypothesis-guessing) vocabulary can never
# affect dna.py's existing entry/exit/risk gene detection or its tests.
_KEYWORD_GROUPS: dict[str, list[str]] = {
    "trend_following": ["ema_fast", "ema_slow", "trend filter", "moving average", "ema50", "ema200", "trend-following", "trend following"],
    "mean_reversion": ["bollinger", "z-score", "zscore", "mean reversion", "mean-reversion", "fade the", "reversion to"],
    "breakout": ["breakout", "donchian", "range break", "n-bar high", "n-bar low"],
    "pullback": ["pullback", "retracement", "dip buy", "buy the dip"],
    "vwap": ["vwap"],
    "volatility_contraction": ["compression", "squeeze", "narrow range", "nr7", "inside bar", "contraction"],
    "statistical_arbitrage": ["pair_zscore", "pairs trade", "stat arb", "relative value", "cointegrat"],
    "relative_strength": ["relative strength", "outperform", "underperform", "ratio spread"],
}
_KEYWORD_WEIGHT = 1


def _score_text(text: str) -> dict[str, int]:
    text_lower = (text or "").lower()
    scores: dict[str, int] = {}
    for group, keywords in _KEYWORD_GROUPS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits:
            scores[group] = scores.get(group, 0) + hits * _KEYWORD_WEIGHT
    return scores


def _score_dna(dna: StrategyDNA) -> dict[str, int]:
    active = set(dna.active_tags())
    scores: dict[str, int] = {}
    for tag, group in _DNA_TAG_PRIORITY:
        if tag in active:
            scores[group] = scores.get(group, 0) + _DNA_WEIGHT
    return scores


def classify_family(
    skeleton_family: str | None = None,
    dna: StrategyDNA | None = None,
    raw_text: str | None = None,
) -> str:
    """Returns one canonical group name from FAMILY_GROUPS. `skeleton_family`
    (if it's a recognized app.search.strategy_space.FAMILIES name) always
    wins outright. Otherwise combines `dna` (an already-extracted
    app.strategy.dna.StrategyDNA, if available) and `raw_text` (source
    code or a Manual config's JSON dump) into a heuristic vote."""
    if skeleton_family in _SKELETON_TO_GROUP:
        return _SKELETON_TO_GROUP[skeleton_family]

    scores: dict[str, int] = {}
    if dna is not None:
        for group, s in _score_dna(dna).items():
            scores[group] = scores.get(group, 0) + s
    if raw_text:
        for group, s in _score_text(raw_text).items():
            scores[group] = scores.get(group, 0) + s

    if not scores:
        return "uncategorized"
    best_score = max(scores.values())
    # Fixed tie-break order -- same priority list dna tags already use,
    # extended with the keyword-only groups after it, so ties resolve the
    # same way every time rather than depending on dict iteration order.
    tie_break_order = [g for _, g in _DNA_TAG_PRIORITY] + list(_KEYWORD_GROUPS.keys())
    for group in tie_break_order:
        if scores.get(group) == best_score:
            return group
    return max(scores, key=scores.get)


def classify_record(record: dict) -> str:
    """Convenience wrapper for a Search Lab stage record (the dict shape
    app.search.batch_runner's _stageN_task functions produce): tries the
    record's own `family` field (a skeleton name) first, then falls back
    to extracting DNA + raw text from whichever of `config` / `code_text`
    the record's source_type populated."""
    skeleton_family = record.get("family")
    if skeleton_family in _SKELETON_TO_GROUP:
        return _SKELETON_TO_GROUP[skeleton_family]

    source_type = record.get("source_type", "manual")
    if source_type == "regime_router":
        # app.strategy.regime_router.RegimeRouterStrategy -- a composed
        # meta-strategy, not a single hypothesis; always this group.
        return "regime_switching"
    if source_type == "manual":
        config = record.get("config") or {}
        dna = extract_dna("manual", config)
        raw_text = None
        try:
            import json
            raw_text = json.dumps(config, default=str)
        except Exception:
            raw_text = str(config)
    else:
        raw_text = record.get("code_text") or ""
        dna = extract_dna(source_type, raw_text)

    return classify_family(skeleton_family=skeleton_family, dna=dna, raw_text=raw_text)


def family_label(group: str) -> str:
    """Human-readable Title Case label for a canonical group name."""
    return group.replace("_", " ").title()
