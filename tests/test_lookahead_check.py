"""
Regression tests for app.strategy.lookahead_check.

Uses small synthetic Python strategies (one deliberately leaky, one clean)
rather than a real uploaded strategy file, so the test is fast and doesn't
depend on external data.
"""
import numpy as np
import pandas as pd

from app.strategy.python import PythonStrategy
from app.strategy.lookahead_check import check_for_lookahead


def _sample_df(n=600):
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(7)
    close = 1900 + np.cumsum(rng.normal(0, 0.5, n))
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


LEAKY_STRATEGY = """
import pandas as pd

def generate_signals(df, config=None):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    h1 = x.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    out = pd.Series(0, index=x.index, dtype="int8")
    for i in range(60, len(x)):
        ts = x["timestamp"].iloc[i]
        # BUG: includes the still-forming current-hour bar.
        h1c = h1[h1.index < ts]
        if len(h1c) < 2:
            continue
        if h1c["close"].iloc[-1] > h1c["close"].iloc[-2]:
            out.iloc[i] = 1
    return out
"""

CLEAN_STRATEGY = """
import pandas as pd

def generate_signals(df, config=None):
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    h1 = x.set_index("timestamp").resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    out = pd.Series(0, index=x.index, dtype="int8")
    for i in range(60, len(x)):
        ts = x["timestamp"].iloc[i]
        # Fixed: only fully-closed HTF bars.
        h1c = h1[h1.index + pd.Timedelta(hours=1) <= ts]
        if len(h1c) < 2:
            continue
        if h1c["close"].iloc[-1] > h1c["close"].iloc[-2]:
            out.iloc[i] = 1
    return out
"""

NO_SIGNAL_STRATEGY = """
import pandas as pd

def generate_signals(df, config=None):
    return pd.Series(0, index=df.index, dtype="int8")
"""


def test_detects_known_leaky_htf_pattern(tmp_path):
    path = _write_strategy(tmp_path, LEAKY_STRATEGY)
    strat = PythonStrategy(path)
    df = _sample_df()
    result = check_for_lookahead(strat, df, max_signal_checkpoints=30)
    assert result.checked
    assert result.bug_detected
    assert result.first_divergence_index is not None


def test_clean_completed_bars_strategy_passes(tmp_path):
    path = _write_strategy(tmp_path, CLEAN_STRATEGY)
    strat = PythonStrategy(path)
    df = _sample_df()
    result = check_for_lookahead(strat, df, max_signal_checkpoints=30)
    assert result.checked
    assert not result.bug_detected


def test_zero_signal_strategy_falls_back_to_fractional_checkpoints(tmp_path):
    path = _write_strategy(tmp_path, NO_SIGNAL_STRATEGY)
    strat = PythonStrategy(path)
    df = _sample_df()
    result = check_for_lookahead(strat, df)
    assert result.checked
    assert not result.bug_detected
    assert result.checkpoint_source == "fractional-fallback"


def test_skips_when_too_little_data(tmp_path):
    path = _write_strategy(tmp_path, CLEAN_STRATEGY)
    strat = PythonStrategy(path)
    df = _sample_df(n=50)
    result = check_for_lookahead(strat, df)
    assert not result.checked
    assert result.skip_reason is not None
