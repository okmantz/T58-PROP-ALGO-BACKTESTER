from dataclasses import dataclass

import pytest

from app.ai.ollama_client import _build_prompt, _parse_suggestions
from app.ai.ollama_settings import DEFAULT_HOST, DEFAULT_MODEL, OllamaSettings, load_settings, save_settings


@dataclass
class _FakeGene:
    label: str
    is_int: bool
    lo: float
    hi: float
    base_value: float


def _genes():
    return [
        _FakeGene("emaFast", True, 6.0, 60.0, 20.0),
        _FakeGene("T58_SL_PIPS", False, 7.5, 75.0, 25.0),
    ]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def test_build_prompt_includes_every_gene_and_its_bounds():
    prompt = _build_prompt(
        "Test Strategy", "pinescript", _genes(),
        {"net_profit": -100.0, "win_rate": 40.0},
        {"account_size": 50000.0, "max_drawdown_pct": 4.0},
        n_suggestions=3,
    )
    assert "emaFast" in prompt
    assert "6.0" in prompt and "60.0" in prompt
    assert "T58_SL_PIPS" in prompt
    assert "net_profit" in prompt
    assert "max_drawdown_pct" in prompt
    assert "JSON array" in prompt


def test_build_prompt_never_asks_the_model_to_write_code():
    prompt = _build_prompt("Test", "python", _genes(), {}, {}, 2)
    assert "NOT being asked to" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_parses_clean_json_array_of_genomes():
    raw = "[[25, 30.0], [18, 40.0]]"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[25, 30.0], [18, 40.0]]


def test_parses_json_wrapped_in_markdown_fence_and_prose():
    raw = "Sure, here are some ideas:\n```json\n[[25, 30.0], [15, 45.0]]\n```\nHope that helps!"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[25, 30.0], [15, 45.0]]


def test_tolerates_a_single_flat_genome_instead_of_a_list_of_them():
    raw = "[25, 30.0]"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[25, 30.0]]


def test_clamps_out_of_range_values_to_gene_bounds():
    raw = "[[9999, -50]]"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[60, 7.5]]  # clamped to gene hi/lo respectively


def test_rounds_integer_genes_but_not_decimal_genes():
    raw = "[[22.6, 31.2]]"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[23, 31.2]]


def test_drops_candidates_with_wrong_length():
    raw = "[[25, 30.0, 999], [18, 40.0]]"
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[18, 40.0]]


def test_drops_candidates_with_non_numeric_values():
    raw = '[[25, "fast"], [18, 40.0]]'
    genomes = _parse_suggestions(raw, _genes())
    assert genomes == [[18, 40.0]]


def test_returns_empty_list_for_garbage_or_missing_json():
    assert _parse_suggestions("I don't know, maybe try tweaking it?", _genes()) == []
    assert _parse_suggestions("", _genes()) == []
    assert _parse_suggestions("[not valid json", _genes()) == []


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def test_settings_default_to_disabled_with_sane_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.ollama_settings.get_app_base_dir", lambda: tmp_path)
    settings = load_settings()
    assert settings.enabled is False
    assert settings.host == DEFAULT_HOST
    assert settings.model == DEFAULT_MODEL
    assert not settings.is_usable


def test_save_and_load_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ai.ollama_settings.get_app_base_dir", lambda: tmp_path)
    save_settings(OllamaSettings(enabled=True, host="http://localhost:11434", model="mistral", api_key="secret123"))
    loaded = load_settings()
    assert loaded.enabled is True
    assert loaded.model == "mistral"
    assert loaded.is_usable
    # api_key round-trips via either keyring or the obfuscated fallback file
    assert loaded.api_key == "secret123"
