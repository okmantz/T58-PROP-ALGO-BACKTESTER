"""Tests for app.strategy.dna."""
from __future__ import annotations

from app.strategy.dna import (
    StrategyDNA, extract_dna, extract_dna_from_text, find_common_patterns,
)

_LIQUIDITY_STRATEGY = """
def generate_signals(df):
    # liquidity sweep of prior session lows, then NY session entry with an HTF trend filter
    df['sweep_low'] = df['low'] < df['prior_low']
    df['ny_session'] = (df.index.hour >= 13) & (df.index.hour < 20)
    atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    df['atr_stop'] = atr * 1.5
    signals.attrs['take_profit_distance'] = atr * 2.0
    adaptive_risk = True
    return signals
"""

_PLAIN_SMA_STRATEGY = """
def generate_signals(df):
    df['sma_fast'] = df['close'].rolling(20).mean()
    df['sma_slow'] = df['close'].rolling(50).mean()
    long_entry = df['sma_fast'] > df['sma_slow']
    take_profit_pips = 40
    stop_loss_pips = 20
    return signals
"""


def test_extract_dna_detects_liquidity_volatility_time_filter_and_adaptive_risk():
    dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    assert dna.entry["liquidity"] is True
    assert dna.entry["volatility"] is True
    assert dna.entry["time_filter"] is True
    assert dna.exit["atr"] is True
    assert dna.risk["adaptive"] is True
    assert dna.entry["market_structure"] is False


def test_extract_dna_plain_sma_has_minimal_genes():
    dna = extract_dna("python", _PLAIN_SMA_STRATEGY)
    assert dna.exit["fixed_rr"] is True
    assert dna.entry["liquidity"] is False
    assert dna.risk["adaptive"] is False


def test_word_boundary_avoids_false_positive_on_substring():
    # "rr" as a bare substring shouldn't match inside unrelated words like "array"
    dna = extract_dna_from_text("result = array.mean()")
    assert dna.exit["fixed_rr"] is False


def test_manual_config_dict_is_supported():
    cfg = {
        "indicators": [
            {"type": "atr", "period": 14, "as": "atr_val"},
            {"type": "rsi", "period": 14, "as": "rsi_val"},
        ],
        "trailing_stop": {"atr_period": 14, "atr_multiple": 1.5},
        "session_start": "13:00", "session_end": "20:00",
    }
    dna = extract_dna("manual", cfg)
    assert dna.entry["momentum"] is True   # rsi
    assert dna.entry["volatility"] is True  # atr
    assert dna.entry["time_filter"] is True  # session_start/session_end
    assert dna.exit["structure"] is True    # trailing_stop


def test_active_tags_and_matched_terms_are_consistent():
    dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    tags = dna.active_tags()
    assert "entry.liquidity" in tags
    assert "risk.adaptive" in tags
    assert "entry.liquidity" in dna.matched_terms
    assert "sweep_low" in dna.matched_terms["entry.liquidity"] or any(
        "sweep" in term for term in dna.matched_terms["entry.liquidity"]
    )


def test_render_tree_contains_all_three_sections():
    dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    tree = dna.render_tree("My Strategy")
    assert "My Strategy" in tree
    assert "ENTRY" in tree
    assert "EXIT" in tree
    assert "RISK" in tree
    assert "[✓]" in tree
    assert "[ ]" in tree


def test_to_dict_round_trips_through_json():
    import json
    dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    d = dna.to_dict()
    json.loads(json.dumps(d))  # must not raise
    assert d["entry"]["liquidity"] is True
    assert "entry.liquidity" in d["active_tags"]


def test_find_common_patterns_surfaces_winner_only_combo():
    winner_dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    loser_dna = extract_dna("python", _PLAIN_SMA_STRATEGY)
    entries = [
        ("winner_1", winner_dna, True),
        ("winner_2", winner_dna, True),
        ("loser_1", loser_dna, False),
        ("loser_2", loser_dna, False),
        ("loser_3", loser_dna, False),
    ]
    patterns = find_common_patterns(entries, min_combo_size=2, max_combo_size=3, min_support_top=2, min_lift=1.0)
    assert patterns  # something was found
    top_pattern = patterns[0]
    assert top_pattern.support_top == 2
    assert top_pattern.support_rest == 0
    assert all(tag in winner_dna.active_tags() for tag in top_pattern.combo)


def test_find_common_patterns_returns_empty_with_no_top_performers():
    dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    entries = [("a", dna, False), ("b", dna, False)]
    assert find_common_patterns(entries) == []


def test_find_common_patterns_respects_min_support_top():
    winner_dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    entries = [("only_one_winner", winner_dna, True)]
    patterns = find_common_patterns(entries, min_support_top=2)
    assert patterns == []


def test_dna_pattern_render_line_is_readable():
    winner_dna = extract_dna("python", _LIQUIDITY_STRATEGY)
    loser_dna = extract_dna("python", _PLAIN_SMA_STRATEGY)
    entries = [("w1", winner_dna, True), ("w2", winner_dna, True), ("l1", loser_dna, False)]
    patterns = find_common_patterns(entries, min_support_top=2, min_lift=1.0)
    assert patterns
    line = patterns[0].render_line()
    assert "top performers" in line
    assert "lift" in line
