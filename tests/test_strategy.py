import pandas as pd
import pytest

from app.strategy.base import StrategyError
from app.strategy.manual import ManualStrategy
from app.strategy.pinescript import PineScriptStrategy
from app.strategy.mql5 import MQL5Strategy


def _sample_df(n=100):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = pd.Series(range(n), dtype=float) + 1.10
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": 100.0,
    })


def test_manual_strategy_sma_cross_generates_signals():
    df = _sample_df(100)
    cfg = {
        "name": "test sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 10, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }
    result = ManualStrategy(cfg).generate(df)
    assert len(result.signals) == len(df)
    assert set(result.signals.unique()).issubset({-1, 0, 1})
    # monotonically increasing close means fast SMA > slow SMA eventually -> long signal appears
    assert (result.signals == 1).any()


def test_manual_strategy_requires_entry_rule():
    df = _sample_df(20)
    with pytest.raises(StrategyError):
        ManualStrategy({"name": "bad"}).generate(df)


def test_manual_strategy_rejects_unsafe_expression():
    df = _sample_df(20)
    cfg = {"name": "bad", "long_entry": "__import__('os').system('echo hi')"}
    with pytest.raises(StrategyError):
        ManualStrategy(cfg).generate(df)


def test_pinescript_strategy_raises_clear_error():
    strat = PineScriptStrategy("strategy(\"test\")\nplot(close)")
    with pytest.raises(StrategyError):
        strat.generate(_sample_df(10))


def test_mql5_strategy_raises_clear_error():
    strat = MQL5Strategy("void OnTick() {}")
    with pytest.raises(StrategyError):
        strat.generate(_sample_df(10))
