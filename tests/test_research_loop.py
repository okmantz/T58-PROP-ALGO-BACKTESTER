"""Tests for app.ai.research_loop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ai.ollama_settings import OllamaSettings
from app.ai.strategy_generator import GenerationResult
from app.ai import research_loop as rl
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules


@pytest.fixture(autouse=True)
def _isolated_experiment_db(tmp_path, monkeypatch):
    """Every run_research_loop() call records into app.ai.experiment_memory
    (both the SQL row and, since research_loop's dedup check consults
    is_dna_tagset_previously_discarded, PAST records) -- isolate each
    test's database so one test's DISCARD verdicts can never leak into
    another test's dedup check."""
    from app.ai import experiment_memory
    monkeypatch.setattr(experiment_memory, "_db_path", lambda: tmp_path / "experiments.db")
    yield


def _df(n=2000, seed=1, trending=True):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    if trending:
        drift = np.linspace(0, 60, n)
        noise = np.cumsum(rng.normal(0, 0.5, n))
    else:
        drift = np.zeros(n)
        noise = rng.normal(0, 0.5, n)
    price = 1900 + drift + noise
    high = price + np.abs(rng.normal(0.3, 0.15, n))
    low = price - np.abs(rng.normal(0.3, 0.15, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": high, "low": low, "close": price, "volume": 100.0,
    })


_SMA_CROSS_CODE = """
import numpy as np

STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df['close'].rolling(10).mean()
    slow = df['close'].rolling(30).mean()
    signal = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return df['close'].__class__(signal, index=df.index)
"""

_NO_SIGNAL_CODE = """
def generate_signals(df):
    return df['close'] * 0
"""


def _settings(usable=True):
    return OllamaSettings(enabled=usable, host="http://localhost:11434", model="llama3.1")


def _rules():
    return PropRules(account_size=50_000, evaluation_profit_target_pct=8, daily_loss_limit_pct=5, max_drawdown_pct=10)


def _risk():
    return RiskConfig(initial_balance=50_000, risk_mode="percent", risk_value=1.0)


# ---------------------------------------------------------------------------
# diagnose_failure
# ---------------------------------------------------------------------------

class _FakeTrade:
    def __init__(self, entry_time, pnl):
        self.entry_time = entry_time
        self.pnl = pnl


def test_diagnose_failure_flags_low_volatility_concentration():
    df = _df(500, trending=False)
    # Make the early half of the series artificially low-volatility by
    # flattening high/low there, then concentrate losing trades in it.
    df.loc[: len(df) // 2, "high"] = df.loc[: len(df) // 2, "close"] + 0.01
    df.loc[: len(df) // 2, "low"] = df.loc[: len(df) // 2, "close"] - 0.01
    early_times = df["timestamp"].iloc[: len(df) // 2 : 5]
    trades = [_FakeTrade(t, -50.0) for t in early_times]
    diag = rl.diagnose_failure(df, trades)
    assert diag["regime"] == "low_vol"
    assert diag["low_vol_loss_pct"] >= 65.0
    assert "low-volatility" in diag["suggestion"]


def test_diagnose_failure_returns_none_suggestion_with_too_few_losers():
    df = _df(200)
    trades = [_FakeTrade(df["timestamp"].iloc[10], -50.0)]
    diag = rl.diagnose_failure(df, trades)
    assert diag["suggestion"] is None


def test_diagnose_failure_handles_no_losers_gracefully():
    df = _df(200)
    trades = [_FakeTrade(df["timestamp"].iloc[i], 50.0) for i in range(10)]
    diag = rl.diagnose_failure(df, trades)
    assert diag["suggestion"] is None


# ---------------------------------------------------------------------------
# _ask_ollama_next_hypothesis (fallback path -- no network)
# ---------------------------------------------------------------------------

def test_ask_ollama_falls_back_when_not_usable():
    settings = _settings(usable=False)
    diag = {"suggestion": "Test adding an ATR-percentile entry filter."}
    idea, from_ollama = rl._ask_ollama_next_hypothesis(settings, "Original idea.", diag)
    assert from_ollama is False
    assert "Original idea." in idea
    assert "ATR-percentile" in idea


def test_ask_ollama_falls_back_to_prior_idea_when_no_suggestion():
    settings = _settings(usable=False)
    idea, from_ollama = rl._ask_ollama_next_hypothesis(settings, "Original idea.", {"suggestion": None})
    assert idea == "Original idea."
    assert from_ollama is False


# ---------------------------------------------------------------------------
# run_research_loop -- generate_strategy mocked, no real Ollama needed
# ---------------------------------------------------------------------------

def test_run_research_loop_requires_usable_ollama():
    result = rl.run_research_loop(_df(), _risk(), _rules(), _settings(usable=False))
    assert result.stopped_reason == "ollama_unavailable"
    assert result.iterations == []


def test_run_research_loop_records_keep_verdict_for_a_working_strategy(monkeypatch):
    monkeypatch.setattr(rl, "generate_strategy", lambda *a, **k: GenerationResult(code=_SMA_CROSS_CODE))
    monkeypatch.setattr(rl, "_ask_ollama_next_hypothesis", lambda *a, **k: ("next idea", False))
    cfg = rl.ResearchLoopConfig(n_iterations=1, mc_sims=50, survival_sims=100, keep_score_threshold=-1.0)
    result = rl.run_research_loop(_df(2000), _risk(), _rules(), _settings(), cfg)
    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.verdict in ("KEEP", "DISCARD")  # depends on synthetic data, but must have run
    assert it.trades > 0
    assert it.prop_survival_score is not None
    assert result.best_iteration is it


def test_run_research_loop_marks_no_trades_as_discard(monkeypatch):
    monkeypatch.setattr(rl, "generate_strategy", lambda *a, **k: GenerationResult(code=_NO_SIGNAL_CODE))
    cfg = rl.ResearchLoopConfig(n_iterations=1)
    result = rl.run_research_loop(_df(500), _risk(), _rules(), _settings(), cfg)
    assert len(result.iterations) == 1
    assert result.iterations[0].verdict == "NO_TRADES"
    assert result.iterations[0].trades == 0


def test_run_research_loop_stops_on_generation_failure(monkeypatch):
    monkeypatch.setattr(rl, "generate_strategy", lambda *a, **k: GenerationResult(error="Ollama unreachable"))
    cfg = rl.ResearchLoopConfig(n_iterations=3)
    result = rl.run_research_loop(_df(500), _risk(), _rules(), _settings(), cfg)
    assert len(result.iterations) == 1  # breaks immediately, doesn't retry a dead connection 3 times
    assert result.iterations[0].verdict == "GENERATION_FAILED"
    assert result.stopped_reason == "GENERATION_FAILED"


def test_run_research_loop_skips_repeated_dna_signature(monkeypatch):
    # Every iteration generates the exact same code -> exact same DNA
    # signature -> after the first DISCARD, subsequent iterations must
    # be skipped as duplicates rather than re-run.
    monkeypatch.setattr(rl, "generate_strategy", lambda *a, **k: GenerationResult(code=_SMA_CROSS_CODE))
    monkeypatch.setattr(rl, "_ask_ollama_next_hypothesis", lambda *a, **k: ("same idea again", False))
    cfg = rl.ResearchLoopConfig(n_iterations=3, mc_sims=50, survival_sims=100, keep_score_threshold=999.0)  # force DISCARD
    result = rl.run_research_loop(_df(2000), _risk(), _rules(), _settings(), cfg)
    assert len(result.iterations) == 3
    verdicts = [it.verdict for it in result.iterations]
    assert verdicts[0] == "DISCARD"
    assert "SKIPPED_DUPLICATE" in verdicts[1:]


def test_research_loop_iteration_to_dict_is_plain_data():
    it = rl.ResearchLoopIteration(iteration=1, idea="x", strategy_name="s", verdict="KEEP")
    d = it.to_dict()
    assert d["iteration"] == 1
    assert d["verdict"] == "KEEP"
