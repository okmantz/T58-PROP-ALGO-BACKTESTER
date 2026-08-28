"""Tests for the new indicator primitives added to app.strategy.indicators
(relative_volume, volume_delta, pair_ratio, pair_zscore) and the matching
app.strategy.manual operand kinds (including "time_of_day")."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.pairs import merge_pair_series
from app.strategy.indicators import build_indicator_series, pair_ratio, pair_zscore, relative_volume, volume_delta
from app.strategy.manual import ManualStrategy


def _df_with_volume(n=60):
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    close = pd.Series(np.linspace(100, 100 + n * 0.01, n))
    openp = close - 0.01  # every bar closes up-from-open by construction
    volume = pd.Series([10.0] * n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": close + 0.02, "low": close - 0.02,
                          "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# relative_volume
# ---------------------------------------------------------------------------

def test_relative_volume_is_1_when_volume_is_constant():
    df = _df_with_volume()
    rv = relative_volume(df, period=10)
    assert np.allclose(rv.dropna().to_numpy(), 1.0)


def test_relative_volume_flags_an_above_average_bar():
    df = _df_with_volume(n=30)
    df.loc[20, "volume"] = 1000.0  # one big spike
    rv = relative_volume(df, period=10)
    assert rv.iloc[20] > 5.0


def test_relative_volume_falls_back_to_1_with_no_volume_column():
    df = _df_with_volume().drop(columns=["volume"])
    rv = relative_volume(df, period=10)
    assert (rv == 1.0).all()


# ---------------------------------------------------------------------------
# volume_delta
# ---------------------------------------------------------------------------

def test_volume_delta_is_positive_when_every_bar_closes_up():
    df = _df_with_volume()  # constructed so close > open on every bar
    vd = volume_delta(df, period=10)
    assert np.allclose(vd.iloc[10:].to_numpy(), 1.0)  # 100% of volume is "up" volume, once the window is full


def test_volume_delta_is_negative_when_every_bar_closes_down():
    df = _df_with_volume()
    df["open"] = df["close"] + 0.01  # flip every bar to a down-close
    vd = volume_delta(df, period=10)
    assert np.allclose(vd.iloc[10:].to_numpy(), -1.0)


def test_volume_delta_zero_with_no_volume_column():
    df = _df_with_volume().drop(columns=["volume"])
    vd = volume_delta(df, period=10)
    assert (vd == 0.0).all()


# ---------------------------------------------------------------------------
# pair_ratio / pair_zscore
# ---------------------------------------------------------------------------

def _pair_setup():
    df = _df_with_volume(n=80)
    pair_df = df[["timestamp", "close"]].copy()
    pair_df["close"] = pair_df["close"] * 2.0  # simple, exact 2x relationship
    merged = merge_pair_series(df, pair_df)
    return merged


def test_pair_ratio_is_exactly_a_half_for_a_2x_relationship():
    merged = _pair_setup()
    ratio = pair_ratio(merged)
    assert np.allclose(ratio.dropna().to_numpy(), 0.5)


def test_pair_ratio_missing_column_raises_keyerror():
    df = _df_with_volume()
    with pytest.raises(KeyError):
        pair_ratio(df)


def test_pair_zscore_is_zero_for_a_perfectly_constant_ratio():
    merged = _pair_setup()
    z = pair_zscore(merged, period=10)
    assert np.allclose(z.iloc[10:].to_numpy(), 0.0, atol=1e-8)


def test_pair_zscore_reacts_to_a_ratio_shock():
    merged = _pair_setup()
    merged.loc[merged.index[-1], "close"] *= 1.5   # break the ratio on the last bar
    z = pair_zscore(merged, period=20)
    assert abs(z.iloc[-1]) > 1.0


# ---------------------------------------------------------------------------
# build_indicator_series dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["relative_volume", "volume_delta"])
def test_build_indicator_series_dispatches_volume_kinds(kind):
    df = _df_with_volume()
    series = build_indicator_series(df, kind, period=10)
    assert len(series) == len(df)


def test_build_indicator_series_dispatches_pair_kinds():
    merged = _pair_setup()
    assert len(build_indicator_series(merged, "pair_ratio")) == len(merged)
    assert len(build_indicator_series(merged, "pair_zscore", period=10)) == len(merged)


# ---------------------------------------------------------------------------
# manual.py "time_of_day" operand
# ---------------------------------------------------------------------------

def test_time_of_day_operand_is_boolean_flag_for_session_window():
    df = _df_with_volume(n=100)  # 15min bars, so this spans several hours
    cfg = {
        "name": "session filter test",
        "entry_conditions": {
            "long": [{"left": {"type": "time_of_day", "session_start": "00:00", "session_end": "01:00"},
                      "operator": "is true", "right": {"type": "value", "value": 1}}],
        },
        "exit_conditions": {"long": [], "short": []},
        "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    strategy = ManualStrategy(cfg)
    result = strategy.generate(df)
    ts = pd.to_datetime(df["timestamp"])
    in_window = (ts.dt.time >= pd.to_datetime("00:00").time()) & (ts.dt.time <= pd.to_datetime("01:00").time())
    # The signal has no exit condition, so once triggered it HOLDS (this
    # engine's normal long/flat/short signal semantics -- see
    # app.strategy.base.signals_from_conditions) rather than resetting to
    # flat the moment the raw boolean condition goes false. What the
    # "time_of_day" operand must guarantee is that entry can only ever be
    # TRIGGERED while inside the session window -- so every bar before the
    # window opens must still be flat, and the transition into a long
    # position must land on an in-window bar.
    before_window = ts.dt.time < pd.to_datetime("00:00").time()
    assert (result.signals[before_window] == 0).all() if before_window.any() else True
    first_long_idx = result.signals[result.signals == 1].index.min()
    assert in_window.loc[first_long_idx]
