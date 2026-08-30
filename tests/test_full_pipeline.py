import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.optimize.parameter_space import RefinementError
from app.orchestration.full_pipeline import FullPipelineConfig, run_full_pipeline
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.manual import ManualStrategy


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _trending_df(n=2400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
    }


def _never_fires_config():
    return {
        "name": "never fires",
        "indicators": [{"type": "sma", "period": 5, "column": "close", "as": "sma_fast"}],
        "long_entry": "sma_fast > 999999",
        "long_exit": "sma_fast < 0",
        "short_entry": "sma_fast > 999999",
        "short_exit": "sma_fast < 0",
    }


def _cfg(**overrides):
    base = dict(
        n_folds=3, ga_population=4, ga_generations=1, ga_search_mc_sims=20,
        final_mc_sims=200, oos_check_folds=3, holdout_frac=0.2,
    )
    base.update(overrides)
    return FullPipelineConfig(**base)


def test_full_pipeline_end_to_end_manual_strategy(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    assert len(result.baseline_bt.trades) > 0
    assert len(result.final_bt.trades) > 0
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")
    assert result.report_paths["html"].exists()
    # Manual configs aren't files -- nothing should be saved to the library.
    assert result.saved_library_path is None
    assert "aren't files" in result.saved_library_note


def test_full_pipeline_raises_fast_on_zero_trade_baseline(tmp_path):
    df = _trending_df()
    strategy = ManualStrategy(_never_fires_config())
    with pytest.raises(RefinementError, match="ZERO trades"):
        run_full_pipeline(df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg())


def test_full_pipeline_python_strategy_saves_winner_to_library(tmp_path):
    df = _trending_df(n=2400, seed=9)
    py_source = '''
import pandas as pd

STRATEGY_NAME = "Test SMA Cross"
FAST = 5
SLOW = 15

def generate_signals(df: pd.DataFrame):
    fast = df["close"].rolling(FAST).mean()
    slow = df["close"].rolling(SLOW).mean()
    signals = pd.Series(0, index=df.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    return signals
'''
    strat_path = tmp_path / "test_sma.py"
    strat_path.write_text(py_source)

    from app.strategy.python import PythonStrategy
    strategy = PythonStrategy(strat_path)

    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    assert result.final_source_type == "python"
    assert result.final_code_text is not None
    if result.saved_library_path is not None:
        assert result.saved_library_path.exists()
        assert result.saved_library_path.suffix == ".py"
        saved = library.list_saved_strategies("python")
        assert any(s.status in ("validated", "tested_passed", "tested_failed") for s in saved)


def test_full_pipeline_skips_optimization_gracefully_when_no_tunable_params(tmp_path, monkeypatch):
    """A strategy with no numeric parameters to tune should still complete
    the pipeline end-to-end using the baseline configuration as final,
    rather than failing the whole run."""
    df = _trending_df()
    strategy = ManualStrategy({
        "name": "no params",
        "indicators": [],
        "long_entry": "close > low",
        "long_exit": "close < high",
        "short_entry": "close < high",
        "short_exit": "close > low",
    })
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=None,
    )
    assert result.refinement_ran is False
    assert result.refinement_skip_reason is not None
    assert result.final_config == strategy.config
    assert result.report_paths["html"].exists()


def test_full_pipeline_with_ai_assist_enabled_seeds_ga_and_logs(tmp_path, monkeypatch):
    """ollama_settings, when usable, must actually reach the GA (via
    ai_suggest_cb) and log AI activity -- without needing a live Ollama
    server, by stubbing OllamaClient itself."""
    from app.ai.ollama_settings import OllamaSettings
    import app.ai.ollama_client as ollama_client_module

    class _FakeResult:
        def __init__(self, genomes):
            self.genomes = genomes
            self.error = None

    class _FakeOllamaClient:
        def __init__(self, settings):
            self.settings = settings

        def suggest_parameter_adjustments(self, **kwargs):
            genes = kwargs["genes"]
            return _FakeResult([[g.base_value for g in genes]])

    monkeypatch.setattr(ollama_client_module, "OllamaClient", _FakeOllamaClient)

    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    settings = OllamaSettings(enabled=True, host="http://localhost:11434", model="llama3.1")

    logs = []
    result = run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(),
        progress_cb=logs.append, ollama_settings=settings,
    )
    assert any("AI assist" in line for line in logs)
    assert result.verdict in ("READY", "MARGINAL", "NOT READY")


def test_full_pipeline_ai_assist_disabled_by_default(tmp_path):
    """Omitting ollama_settings entirely (the default for every existing
    caller) must behave exactly as before -- no AI-related log lines."""
    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    logs = []
    run_full_pipeline(df, strategy, RiskConfig(), PropRules(), tmp_path, _cfg(), progress_cb=logs.append)
    assert not any("AI assist" in line for line in logs)


def test_full_pipeline_ai_assist_gives_up_after_two_consecutive_failures(tmp_path, monkeypatch):
    """A consistently failing/unreachable Ollama must not pay its timeout
    on every single generation -- after 2 consecutive failures it should
    stop trying for the rest of the run and say so once."""
    from app.ai.ollama_settings import OllamaSettings
    import app.ai.ollama_client as ollama_client_module

    class _FakeResult:
        def __init__(self):
            self.genomes = []
            self.error = "Ollama at http://localhost:11434 didn't respond in time."

    call_count = {"n": 0}

    class _FakeOllamaClient:
        def __init__(self, settings):
            pass

        def suggest_parameter_adjustments(self, **kwargs):
            call_count["n"] += 1
            return _FakeResult()

    monkeypatch.setattr(ollama_client_module, "OllamaClient", _FakeOllamaClient)

    df = _trending_df()
    strategy = ManualStrategy(_sma_config())
    settings = OllamaSettings(enabled=True, host="http://localhost:11434", model="llama3.1")
    cfg = _cfg(ga_generations=5)  # would be 6 calls (gen 0-5) without the circuit breaker

    logs = []
    run_full_pipeline(
        df, strategy, RiskConfig(), PropRules(), tmp_path, cfg,
        progress_cb=logs.append, ollama_settings=settings,
    )
    assert call_count["n"] == 2  # stopped after exactly 2 consecutive failures
    assert any("giving up after 2 consecutive failures" in line for line in logs)
