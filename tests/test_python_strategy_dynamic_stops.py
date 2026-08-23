"""
Regression tests for the Python strategy adapter's dynamic stop/target
passthrough (signals.attrs contract). Before this, a Python strategy had
no way to tell the engine what stop or target it actually intended for a
given trade -- generate_signals() could only return a bare -1/0/1 series,
so any per-trade risk logic the strategy computed was silently discarded
and every trade fell back to the engine's generic default stop with no
take-profit at all.
"""
import numpy as np
import pandas as pd
import pytest

from app.strategy.base import StrategyError
from app.strategy.python import PythonStrategy


def _sample_df(n=50):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(np.linspace(1900, 1910, n))
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 100.0,
    })


def _write_strategy(tmp_path, body: str) -> str:
    path = tmp_path / "strat.py"
    path.write_text(body)
    return str(path)


def test_python_strategy_without_attrs_has_no_dynamic_stops(tmp_path):
    """Backward compatibility: old-style strategies with no .attrs still work,
    and simply carry no dynamic stop/target (the pre-existing behavior)."""
    src = """
import pandas as pd
def generate_signals(df, config=None):
    s = pd.Series(0, index=df.index, dtype="int8")
    s.iloc[10] = 1
    return s
"""
    strat = PythonStrategy(_write_strategy(tmp_path, src))
    result = strat.generate(_sample_df())
    assert result.stop_loss_distance is None
    assert result.take_profit_distance is None
    assert result.signals.iloc[10] == 1


def test_python_strategy_attrs_stop_and_take_profit_passthrough(tmp_path):
    """A strategy that attaches stop_loss_distance/take_profit_distance via
    .attrs should have those values reach the StrategyResult intact, aligned
    to df's index, so the execution engine actually uses them."""
    src = """
import numpy as np
import pandas as pd
def generate_signals(df, config=None):
    s = pd.Series(0, index=df.index, dtype="int8")
    s.iloc[10] = 1
    s.iloc[20] = -1
    stop = pd.Series(np.nan, index=df.index)
    take = pd.Series(np.nan, index=df.index)
    stop.iloc[10] = 2.5
    take.iloc[10] = 6.25
    stop.iloc[20] = 3.0
    take.iloc[20] = 7.5
    s.attrs["stop_loss_distance"] = stop
    s.attrs["take_profit_distance"] = take
    s.attrs["breakeven_trigger_r"] = 1.0
    return s
"""
    strat = PythonStrategy(_write_strategy(tmp_path, src))
    df = _sample_df()
    result = strat.generate(df)

    assert result.stop_loss_distance is not None
    assert result.take_profit_distance is not None
    assert result.stop_loss_distance.iloc[10] == pytest.approx(2.5)
    assert result.take_profit_distance.iloc[10] == pytest.approx(6.25)
    assert result.stop_loss_distance.iloc[20] == pytest.approx(3.0)
    assert result.breakeven_trigger_r == pytest.approx(1.0)
    # non-signal bars stay NaN -- only entry bars carry a distance
    assert pd.isna(result.stop_loss_distance.iloc[0])


def test_python_strategy_attrs_wrong_length_raises(tmp_path):
    """A malformed distance series (wrong length) must fail loudly rather
    than silently misalign and corrupt every trade's risk sizing."""
    src = """
import pandas as pd
def generate_signals(df, config=None):
    s = pd.Series(0, index=df.index, dtype="int8")
    s.iloc[10] = 1
    s.attrs["stop_loss_distance"] = pd.Series([1.0, 2.0, 3.0])  # wrong length
    return s
"""
    strat = PythonStrategy(_write_strategy(tmp_path, src))
    with pytest.raises(StrategyError):
        strat.generate(_sample_df())


def test_python_strategy_all_nan_distance_treated_as_absent(tmp_path):
    """An all-NaN distance series (e.g. a strategy that never actually filled
    it in) should behave exactly like not providing one -- fall back to
    STOP_LOSS_PIPS / the engine default rather than being passed through as
    a series of nothing."""
    src = """
import numpy as np
import pandas as pd
def generate_signals(df, config=None):
    s = pd.Series(0, index=df.index, dtype="int8")
    s.iloc[10] = 1
    s.attrs["stop_loss_distance"] = pd.Series(np.nan, index=df.index)
    return s
"""
    strat = PythonStrategy(_write_strategy(tmp_path, src))
    result = strat.generate(_sample_df())
    assert result.stop_loss_distance is None
