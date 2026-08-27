"""Tests for app.search.strategy_space."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.search.strategy_space import (
    FAMILIES, StrategySpaceError, build_strategy_from_spec, family_description,
    family_grid_size, generate_search_space, list_families, spec_from_strategy,
)
from app.strategy.manual import ManualStrategy
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.python import PythonStrategy


def _synthetic_df(n=2500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.02, 0.5, n))
    close = 100 + drift + np.sin(np.arange(n) / 50) * 2
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close + rng.normal(0, 0.1, n)
    vol = rng.random(n) * 1000
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": vol})


_PYTHON_SRC = '''STRATEGY_NAME = "Test EMA Cross"
EMA_FAST = 5
EMA_SLOW = 15
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    sig = (fast > slow).astype(int) - (fast < slow).astype(int)
    return sig
'''

_PINESCRIPT_SRC = '''//@version=5
strategy("Test", overlay=true)
fastLen = input.int(5, "Fast Length")
slowLen = input.int(15, "Slow Length")
fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
longCondition = ta.crossover(fast, slow)
shortCondition = ta.crossunder(fast, slow)
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
// T58_SL_PIPS=20
// T58_TP_PIPS=40
'''

_MQL5_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 15, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) {
      trade.Buy(0.1, _Symbol);
   }
   if (fastMA < slowMA) {
      trade.Sell(0.1, _Symbol);
   }
   // T58_SL_PIPS=20
   // T58_TP_PIPS=40
}
'''


def _write_python_strategy(tmp_path):
    path = tmp_path / "strat.py"
    path.write_text(_PYTHON_SRC, encoding="utf-8")
    return PythonStrategy(path)


# ---------------------------------------------------------------------------
# Named hypothesis families (Manual only)
# ---------------------------------------------------------------------------

def test_list_families_matches_registry():
    families = list_families()
    assert set(families) == set(FAMILIES)
    for name in families:
        assert family_description(name)
        assert family_grid_size(name) > 0


def test_unknown_family_raises():
    with pytest.raises(StrategySpaceError):
        generate_search_space("family", family="not_a_real_family")


def test_single_mode_wraps_exactly_one_config():
    cfg = {
        "name": "SMA Cross",
        "indicators": [{"type": "sma", "period": 10, "column": "close", "as": "s"}],
        "long_entry": "s > close", "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    space = generate_search_space("single", single_config=cfg)
    assert space.mode == "single"
    assert len(space.candidates) == 1
    (cid, spec), = space.candidates.items()
    assert spec["source_type"] == "manual"
    assert spec["config"]["name"] == "SMA Cross"
    # Must be a deep copy, not the same object, so later mutation (e.g. Stage 2's
    # GA writing tuned parameters back) never mutates the caller's original config.
    spec["config"]["name"] = "mutated"
    assert cfg["name"] == "SMA Cross"


def test_single_mode_requires_a_config_or_strategy():
    with pytest.raises(StrategySpaceError):
        generate_search_space("single", single_config=None)


def test_family_mode_generates_full_grid_when_under_cap():
    space = generate_search_space("family", family="mean_reversion_band", max_candidates=10_000, seed=1)
    assert space.mode == "family"
    assert space.sampled is False
    assert len(space.candidates) == space.total_generated == family_grid_size("mean_reversion_band")


def test_family_mode_samples_reproducibly_when_over_cap():
    space_a = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=7)
    space_b = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=7)
    assert space_a.sampled is True
    assert len(space_a.candidates) == 20
    assert set(space_a.candidates.keys()) == set(space_b.candidates.keys())
    # IDs are content-addressed (hash of family + params), not positional --
    # so same seed must also reproduce the exact same underlying configs,
    # not just the same-shaped ID set.
    assert space_a.candidates == space_b.candidates
    # different seed -> (almost certainly) a different sample -> different IDs,
    # since IDs are derived from the params themselves.
    space_c = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=99)
    assert set(space_a.candidates.keys()) != set(space_c.candidates.keys())


def test_family_all_combines_every_family():
    space = generate_search_space("family", family="all", max_candidates=100_000, seed=1)
    expected_total = sum(family_grid_size(name) for name in FAMILIES)
    assert space.total_generated == expected_total
    seen_families = {meta["family"] for meta in space.meta.values()}
    assert seen_families == set(FAMILIES)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_every_family_produces_valid_runnable_configs(family_name):
    """
    Every generated candidate must build into a real Strategy and produce a
    finite result on real data -- this is also a regression guard for the
    "close > highest_high(window including current bar)" class of bug,
    which is structurally always-false and would otherwise show up as
    "zero trades" silently rather than as an explicit test failure.
    """
    df = _synthetic_df()
    risk = RiskConfig()
    space = generate_search_space("family", family=family_name, max_candidates=6, seed=3)
    assert len(space.candidates) > 0
    any_trades = False
    for cid, spec in space.candidates.items():
        assert spec["source_type"] == "manual"
        strategy = build_strategy_from_spec(spec)
        result = run_backtest(df, strategy, risk)
        assert result.statistics is not None
        if result.trades:
            any_trades = True
    assert any_trades, f"family '{family_name}' produced zero trades across every sampled candidate"


# ---------------------------------------------------------------------------
# build_strategy_from_spec / spec_from_strategy round-trips, all 4 source types
# ---------------------------------------------------------------------------

def test_spec_round_trip_manual():
    strategy = ManualStrategy({"name": "x", "long_entry": "close > 0", "stop_loss_pips": 10, "take_profit_pips": 20})
    spec = spec_from_strategy(strategy)
    assert spec["source_type"] == "manual"
    rebuilt = build_strategy_from_spec(spec)
    assert isinstance(rebuilt, ManualStrategy)
    assert rebuilt.config["name"] == "x"


def test_spec_round_trip_python(tmp_path):
    strategy = _write_python_strategy(tmp_path)
    spec = spec_from_strategy(strategy)
    assert spec["source_type"] == "python"
    assert spec["code_extension"] == ".py"
    assert "EMA_FAST" in spec["code_text"]
    rebuilt = build_strategy_from_spec(spec, tmp_dir=tmp_path / "scratch")
    assert isinstance(rebuilt, PythonStrategy)
    df = _synthetic_df(n=200)
    rebuilt.generate(df)  # must not raise


def test_spec_round_trip_pinescript():
    strategy = PineScriptStrategy(_PINESCRIPT_SRC)
    spec = spec_from_strategy(strategy)
    assert spec["source_type"] == "pinescript"
    assert spec["code_extension"] == ".pine"
    rebuilt = build_strategy_from_spec(spec)
    assert isinstance(rebuilt, PineScriptStrategy)
    rebuilt.generate(_synthetic_df(n=200))  # must not raise


def test_spec_round_trip_mql5():
    strategy = MQL5Strategy(_MQL5_SRC)
    spec = spec_from_strategy(strategy)
    assert spec["source_type"] == "mql5"
    assert spec["code_extension"] == ".mq5"
    rebuilt = build_strategy_from_spec(spec)
    assert isinstance(rebuilt, MQL5Strategy)
    rebuilt.generate(_synthetic_df(n=200))  # must not raise


def test_build_strategy_from_spec_python_requires_tmp_dir():
    strategy = ManualStrategy({"name": "x", "long_entry": "close>0"})
    spec = spec_from_strategy(strategy)
    spec["source_type"] = "python"
    spec["code_text"] = _PYTHON_SRC
    with pytest.raises(StrategySpaceError):
        build_strategy_from_spec(spec, tmp_dir=None)


# ---------------------------------------------------------------------------
# Single mode with a `strategy=` instance, all 4 source types
# ---------------------------------------------------------------------------

def test_single_mode_with_manual_strategy_instance():
    strategy = ManualStrategy({"name": "x", "long_entry": "close > 0", "stop_loss_pips": 10, "take_profit_pips": 20})
    space = generate_search_space("single", strategy=strategy)
    assert len(space.candidates) == 1
    (cid, spec), = space.candidates.items()
    assert spec["source_type"] == "manual"
    assert spec["config"]["name"] == "x"


def test_single_mode_with_python_strategy_instance(tmp_path):
    strategy = _write_python_strategy(tmp_path)
    space = generate_search_space("single", strategy=strategy)
    assert len(space.candidates) == 1
    (cid, spec), = space.candidates.items()
    assert spec["source_type"] == "python"
    assert spec["code_extension"] == ".py"


def test_single_mode_with_pinescript_strategy_instance():
    strategy = PineScriptStrategy(_PINESCRIPT_SRC)
    space = generate_search_space("single", strategy=strategy)
    (cid, spec), = space.candidates.items()
    assert spec["source_type"] == "pinescript"


def test_single_mode_with_mql5_strategy_instance():
    strategy = MQL5Strategy(_MQL5_SRC)
    space = generate_search_space("single", strategy=strategy)
    (cid, spec), = space.candidates.items()
    assert spec["source_type"] == "mql5"


# ---------------------------------------------------------------------------
# Family mode: grid around a given strategy (`strategy=`), all 4 source types
# ---------------------------------------------------------------------------

def test_grid_family_around_manual_strategy():
    strategy = ManualStrategy({
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 20, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "stop_loss_pips": 20, "take_profit_pips": 40,
    })
    space = generate_search_space("family", strategy=strategy, grid_points_per_gene=2, max_candidates=100, seed=1)
    assert space.mode == "family"
    assert space.family == "manual_grid"
    assert len(space.candidates) > 1
    for cid, spec in space.candidates.items():
        assert spec["source_type"] == "manual"
        build_strategy_from_spec(spec)  # must construct without error


def test_grid_family_around_python_strategy(tmp_path):
    strategy = _write_python_strategy(tmp_path)
    space = generate_search_space("family", strategy=strategy, grid_points_per_gene=2, max_candidates=50, seed=1)
    assert space.family == "python_grid"
    assert len(space.candidates) > 1
    scratch = tmp_path / "scratch"
    seen_code = set()
    for cid, spec in space.candidates.items():
        assert spec["source_type"] == "python"
        seen_code.add(spec["code_text"])
        rebuilt = build_strategy_from_spec(spec, tmp_dir=scratch)
        rebuilt.generate(_synthetic_df(n=100))  # must not raise
    assert len(seen_code) > 1  # actually varies the parameters, not just re-emitting the same file


def test_grid_family_around_pinescript_strategy():
    strategy = PineScriptStrategy(_PINESCRIPT_SRC)
    space = generate_search_space("family", strategy=strategy, grid_points_per_gene=2, max_candidates=50, seed=1)
    assert space.family == "pinescript_grid"
    assert len(space.candidates) > 1
    for cid, spec in space.candidates.items():
        assert spec["source_type"] == "pinescript"
        build_strategy_from_spec(spec).generate(_synthetic_df(n=100))


def test_grid_family_around_mql5_strategy():
    strategy = MQL5Strategy(_MQL5_SRC)
    space = generate_search_space("family", strategy=strategy, grid_points_per_gene=2, max_candidates=50, seed=1)
    assert space.family == "mql5_grid"
    assert len(space.candidates) > 1
    for cid, spec in space.candidates.items():
        assert spec["source_type"] == "mql5"
        build_strategy_from_spec(spec).generate(_synthetic_df(n=100))


def test_grid_family_candidate_ids_are_content_addressed_and_reproducible():
    strategy = MQL5Strategy(_MQL5_SRC)
    space_a = generate_search_space("family", strategy=strategy, grid_points_per_gene=3, max_candidates=3, seed=5)
    space_b = generate_search_space("family", strategy=strategy, grid_points_per_gene=3, max_candidates=3, seed=5)
    assert set(space_a.candidates.keys()) == set(space_b.candidates.keys())
    assert space_a.candidates == space_b.candidates


def test_grid_family_raises_when_strategy_has_no_tunable_parameters():
    strategy = ManualStrategy({"name": "no params", "long_entry": "close > 0"})
    with pytest.raises(StrategySpaceError):
        generate_search_space("family", strategy=strategy)


def test_grid_family_strategy_argument_ignores_family_argument():
    """When `strategy` is given, the named-family registry path must not
    be used, even if a `family` name is also (redundantly) passed."""
    strategy = MQL5Strategy(_MQL5_SRC)
    space = generate_search_space("family", family="trend_breakout", strategy=strategy, max_candidates=10, seed=1)
    assert space.family == "mql5_grid"
    for spec in space.candidates.values():
        assert spec["source_type"] == "mql5"
