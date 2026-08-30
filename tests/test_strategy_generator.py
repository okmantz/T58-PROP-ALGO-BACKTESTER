from app.ai.ollama_settings import OllamaSettings
from app.ai.strategy_generator import (
    _build_prompt,
    _extract_code,
    _slugify,
    generate_strategy,
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def test_build_prompt_includes_idea_and_language_contract():
    prompt = _build_prompt("python", "EMA cross with an RSI filter on gold")
    assert "EMA cross with an RSI filter on gold" in prompt
    assert "generate_signals" in prompt
    assert "lookahead" in prompt.lower()


def test_build_prompt_pinescript_lists_supported_subset_only():
    prompt = _build_prompt("pinescript", "simple crossover strategy")
    assert "ta.crossover" in prompt
    assert "strategy.entry" in prompt
    # Should explicitly warn against unsupported constructs.
    assert "security()" in prompt or "multi-timeframe" in prompt


def test_build_prompt_mql5_lists_supported_subset_only():
    prompt = _build_prompt("mql5", "MA crossover EA")
    assert "iMA" in prompt
    assert "trade.Buy" in prompt


def test_build_prompt_includes_research_excerpts_when_given():
    excerpts = [{"source": "paper.pdf", "text": "Momentum strategies show positive autocorrelation.", "score": 1.0}]
    prompt = _build_prompt("python", "momentum idea", research_excerpts=excerpts)
    assert "paper.pdf" in prompt
    assert "positive autocorrelation" in prompt


def test_build_prompt_includes_prior_examples_when_given():
    examples = [{"name": "my_strategy.py", "status": "validated", "excerpt": "def generate_signals(df): ..."}]
    prompt = _build_prompt("python", "another idea", prior_examples=examples)
    assert "my_strategy.py" in prompt
    assert "validated" in prompt


def test_build_prompt_omits_sections_when_nothing_retrieved():
    prompt = _build_prompt("python", "plain idea", research_excerpts=[], prior_examples=[])
    assert "research library" not in prompt.lower()
    assert "existing python strategies" not in prompt.lower()


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def test_extract_code_from_fenced_block():
    raw = "Here is the strategy:\n```python\ndef generate_signals(df):\n    return df['close'] * 0\n```\nEnjoy!"
    code = _extract_code(raw)
    assert code == "def generate_signals(df):\n    return df['close'] * 0"


def test_extract_code_from_unfenced_but_code_like_response():
    raw = "def generate_signals(df):\n    return df['close'] * 0\n"
    code = _extract_code(raw)
    assert code is not None
    assert "generate_signals" in code


def test_extract_code_returns_none_for_pure_prose():
    raw = "I'm not able to help with that request, sorry."
    assert _extract_code(raw) is None


def test_extract_code_returns_none_for_empty_response():
    assert _extract_code("") is None
    assert _extract_code(None) is None


def test_extract_code_picks_first_fence_when_multiple():
    raw = "```python\nAAA\n```\nsome text\n```python\nBBB\n```"
    assert _extract_code(raw) == "AAA"


# ---------------------------------------------------------------------------
# Slugify
# ---------------------------------------------------------------------------

def test_slugify_produces_filesystem_safe_name():
    assert _slugify("EMA Cross + RSI Filter!!") == "ema_cross_rsi_filter"


def test_slugify_falls_back_when_nothing_alphanumeric():
    assert _slugify("???") == "generated_strategy"


# ---------------------------------------------------------------------------
# generate_strategy -- guard clauses (no live Ollama needed)
# ---------------------------------------------------------------------------

def test_generate_strategy_rejects_unknown_language():
    result = generate_strategy(OllamaSettings(enabled=True, host="http://x"), "cobol", "an idea")
    assert result.code is None
    assert "Unknown language" in result.error


def test_generate_strategy_rejects_empty_idea():
    result = generate_strategy(OllamaSettings(enabled=True, host="http://x"), "python", "   ")
    assert result.code is None
    assert "idea" in result.error.lower()


def test_generate_strategy_rejects_disabled_settings():
    result = generate_strategy(OllamaSettings(enabled=False), "python", "an idea")
    assert result.code is None
    assert "ollama" in result.error.lower()
