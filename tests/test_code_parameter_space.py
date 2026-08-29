import pandas as pd
import numpy as np

from app.optimize.code_parameter_space import (
    apply_code_genome,
    discover_mql5_parameters,
    discover_pinescript_parameters,
)
from app.strategy.pinescript import PineScriptStrategy

_PINE_LITERAL_LENGTHS = """\
//@version=5
strategy("test", overlay=true)

// T58_SL_PIPS=25
// T58_TP_PIPS=50

emaFast = ta.ema(close, 20)
emaSlow = ta.ema(close, 50)
rsiVal = ta.rsi(close, 14)

uptrend = emaFast > emaSlow
downtrend = emaFast < emaSlow

longCondition = uptrend and rsiVal < 40
shortCondition = downtrend and rsiVal > 60
exitLongCondition = rsiVal > 60
exitShortCondition = rsiVal < 40

if longCondition
    strategy.entry("Long", strategy.long)
if shortCondition
    strategy.entry("Short", strategy.short)

strategy.close("Long", when=exitLongCondition)
strategy.close("Short", when=exitShortCondition)
"""

_PINE_INPUT_LENGTHS = """\
//@version=5
strategy("test", overlay=true)

fastLen = input.int(20, "Fast length")
emaFast = ta.ema(close, fastLen)

longCondition = emaFast > close

if longCondition
    strategy.entry("Long", strategy.long)
"""


def _flat_df(n=200):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 0.0002, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0005,
        "low": price - 0.0005, "close": price, "volume": 100.0,
    })


def test_discovers_bare_numeric_ta_lengths_as_genes():
    """A ta.ema/rsi call whose length is a plain number (the overwhelmingly
    common way strategies are actually written) should be discovered as a
    tunable parameter, not just input.int()-wrapped lengths."""
    genes = discover_pinescript_parameters(_PINE_LITERAL_LENGTHS)
    names = {g.name for g in genes}
    assert {"emaFast", "emaSlow", "rsiVal", "T58_SL_PIPS", "T58_TP_PIPS"} <= names
    ema_fast_gene = next(g for g in genes if g.name == "emaFast")
    assert ema_fast_gene.base_value == 20.0
    assert ema_fast_gene.is_int


def test_input_wrapped_length_is_not_double_counted():
    """When a length is already input.int()-wrapped, it must be discovered
    exactly once (via the existing input.* path) and not a second time by
    the new bare-literal ta.* pattern, which only matches numeric literals."""
    genes = discover_pinescript_parameters(_PINE_INPUT_LENGTHS)
    assert len(genes) == 1
    assert genes[0].name == "fastLen"
    assert genes[0].base_value == 20.0


def test_patched_ta_length_genome_still_parses_and_runs():
    """Applying a mutated genome to the bare-numeric-length genes must
    produce source that still parses under PineScriptStrategy and runs
    through the engine without error."""
    genes = discover_pinescript_parameters(_PINE_LITERAL_LENGTHS)
    # genome order matches discovery order: SL, TP, emaFast, emaSlow, rsiVal
    genome = [30.0, 60.0, 12.0, 26.0, 21.0]
    patched = apply_code_genome(_PINE_LITERAL_LENGTHS, genes, genome)

    assert "ta.ema(close, 12)" in patched
    assert "ta.ema(close, 26)" in patched
    assert "ta.rsi(close, 21)" in patched

    strategy = PineScriptStrategy(patched)
    result = strategy.generate(_flat_df())
    assert result.stop_loss_pips == 30.0
    assert result.take_profit_pips == 60.0


def test_mql5_discovery_unaffected_by_pinescript_change():
    """Sanity check that the MQL5 discovery path (unchanged) still works
    the same as before after editing the shared module."""
    code = (
        "// T58_SL_PIPS=20\n// T58_TP_PIPS=40\n"
        "double fast = iMA(NULL, 0, 10, 0, MODE_EMA, PRICE_CLOSE, 0);\n"
        "double rsi = iRSI(NULL, 0, 14, PRICE_CLOSE, 0);\n"
    )
    genes = discover_mql5_parameters(code)
    kinds = {g.kind for g in genes}
    assert {"mql5_ma_period", "mql5_rsi_period", "mql5_sl_pips", "mql5_tp_pips"} <= kinds
