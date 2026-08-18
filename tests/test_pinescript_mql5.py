import numpy as np
import pandas as pd
import pytest

from app.strategy.base import StrategyError
from app.strategy.mql5 import MQL5Strategy
from app.strategy.pinescript import PineScriptStrategy


def _oscillating_df(n=300, seed=5):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0005, "low": price - 0.0005,
        "close": price, "volume": 100.0,
    })


PINE_SMA_CROSS = """
//@version=5
strategy("SMA Cross", overlay=true)
fastLen = input.int(5, title="Fast")
slowLen = input.int(15, title="Slow")
fastMA = ta.sma(close, fastLen)
slowMA = ta.sma(close, slowLen)
longCondition = ta.crossover(fastMA, slowMA)
shortCondition = ta.crossunder(fastMA, slowMA)
// T58_SL_PIPS=20
// T58_TP_PIPS=40
if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)
"""

MQL5_SMA_CROSS = """
#include <Trade/Trade.mqh>
CTrade trade;

// T58_SL_PIPS=20
// T58_TP_PIPS=40

void OnTick()
{
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_SMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 15, 0, MODE_SMA, PRICE_CLOSE);

   if (fastMA > slowMA)
   {
      trade.Buy(0.1);
   }
   if (fastMA < slowMA)
   {
      trade.Sell(0.1);
      trade.PositionClose(_Symbol);
   }
}
"""


def test_pinescript_sma_cross_produces_signals_and_sl_tp():
    df = _oscillating_df()
    result = PineScriptStrategy(PINE_SMA_CROSS).generate(df)
    assert set(result.signals.unique()).issubset({-1, 0, 1})
    assert (result.signals != 0).any()
    assert result.stop_loss_pips == 20.0
    assert result.take_profit_pips == 40.0


def test_pinescript_compound_condition_with_rsi():
    df = _oscillating_df(n=200, seed=2)
    pine = """
    strategy("x")
    rsiVal = ta.rsi(close, 14)
    fastMA = ta.sma(close, 5)
    slowMA = ta.sma(close, 15)
    longCondition = ta.crossover(fastMA, slowMA) and rsiVal < 70
    if longCondition
        strategy.entry("Long", strategy.long, when=longCondition)
    """
    result = PineScriptStrategy(pine).generate(df)
    assert len(result.signals) == len(df)


def test_pinescript_unsupported_function_raises_clear_error():
    df = _oscillating_df(n=50)
    pine = """
    strategy("x")
    val = ta.macd(close, 12, 26, 9)
    if val > 0
        strategy.entry("Long", strategy.long)
    """
    with pytest.raises(StrategyError):
        PineScriptStrategy(pine).generate(df)


def test_pinescript_no_entry_call_raises():
    with pytest.raises(StrategyError):
        PineScriptStrategy("strategy(\"test\")\nplot(close)").generate(_oscillating_df(n=20))


def test_mql5_sma_cross_produces_signals_and_sl_tp():
    df = _oscillating_df()
    result = MQL5Strategy(MQL5_SMA_CROSS).generate(df)
    assert set(result.signals.unique()).issubset({-1, 0, 1})
    assert (result.signals != 0).any()
    assert result.stop_loss_pips == 20.0
    assert result.take_profit_pips == 40.0


def test_mql5_kr_brace_style_and_inline_if():
    df = _oscillating_df(seed=9)
    kr = """
    void OnTick() {
      double fastMA = iMA(_Symbol, PERIOD_CURRENT, 5, 0, MODE_SMA, PRICE_CLOSE);
      double slowMA = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
      if (fastMA > slowMA) trade.Buy(0.1);
      if (fastMA < slowMA) trade.Sell(0.1);
    }
    """
    result = MQL5Strategy(kr).generate(df)
    assert (result.signals != 0).any()


def test_mql5_copybuffer_pattern_raises_clear_error():
    df = _oscillating_df(n=50)
    bad = """
    void OnTick() {
      int h = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE);
      double buf[];
      CopyBuffer(h, 0, 0, 1, buf);
      if (buf[0] > 1.0) { trade.Buy(0.1); }
    }
    """
    with pytest.raises(StrategyError):
        MQL5Strategy(bad).generate(df)


def test_mql5_no_buy_sell_raises():
    with pytest.raises(StrategyError):
        MQL5Strategy("void OnTick() {}").generate(_oscillating_df(n=20))
