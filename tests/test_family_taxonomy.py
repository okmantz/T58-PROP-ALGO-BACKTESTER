"""Tests for app.strategy.family_taxonomy."""
from __future__ import annotations

from app.search.strategy_space import FAMILIES
from app.strategy.dna import extract_dna
from app.strategy.family_taxonomy import FAMILY_GROUPS, classify_family, classify_record, family_label


def test_every_skeleton_family_maps_to_a_known_canonical_group():
    from app.strategy.family_taxonomy import _SKELETON_TO_GROUP
    assert set(FAMILIES.keys()) == set(_SKELETON_TO_GROUP.keys())
    for group in _SKELETON_TO_GROUP.values():
        assert group in FAMILY_GROUPS


def test_skeleton_name_is_authoritative_even_with_conflicting_text():
    result = classify_family(skeleton_family="mean_reversion_band", raw_text="ema trend momentum breakout")
    assert result == "mean_reversion"


def test_dna_liquidity_gene_outranks_a_single_momentum_keyword():
    # "sweep_low" and "rsi" both fire (liquidity + momentum genes), but the
    # DNA-tag priority list ranks liquidity above momentum -- see
    # app.strategy.family_taxonomy._DNA_TAG_PRIORITY.
    dna = extract_dna("python", "if sweep_low and rsi < 30: enter_long()")
    result = classify_family(dna=dna)
    assert result == "liquidity_sweep"


def test_keyword_fallback_detects_vwap():
    result = classify_family(raw_text="enter long when close < vwap, exit at vwap")
    assert result == "vwap"


def test_no_signal_at_all_returns_uncategorized():
    assert classify_family() == "uncategorized"
    assert classify_family(raw_text="the quick brown fox") == "uncategorized"


def test_classify_record_uses_family_field_first():
    record = {"family": "trend_breakout", "source_type": "manual", "config": {}}
    assert classify_record(record) == "breakout"


def test_classify_record_falls_back_to_config_text_for_single_mode():
    record = {
        "family": "single", "source_type": "manual",
        "config": {"entry_conditions": {"long": [{"type": "liquidity_sweep"}]}},
    }
    assert classify_record(record) == "liquidity_sweep"


def test_classify_record_handles_code_based_candidates():
    record = {"family": "python_grid", "source_type": "python", "code_text": "vwap_distance = close - vwap"}
    assert classify_record(record) == "vwap"


def test_family_label_is_title_case():
    assert family_label("liquidity_sweep") == "Liquidity Sweep"
    assert family_label("vwap") == "Vwap"
