"""Technical indicators and deterministic market-derived series for T58."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _period(period: int) -> int:
    return max(int(period), 1)


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=_period(period), min_periods=_period(period)).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    p = _period(period)
    return series.ewm(span=p, adjust=False, min_periods=p).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    p = _period(period)
    weights = np.arange(1, p + 1, dtype=float)
    return series.rolling(p, min_periods=p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    p = _period(period)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    avg_loss = loss.ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss.ne(0), 100)
    return result.fillna(50)


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - prev_close).abs(),
        (frame["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    p = _period(period)
    return true_range(frame).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()


def vwap(frame: pd.DataFrame) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
    ts = pd.to_datetime(frame["timestamp"])
    day = ts.dt.normalize()
    pv = typical * volume
    return pv.groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0, np.nan)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    line = fast_ema - slow_ema
    signal_line = line.ewm(span=_period(signal), adjust=False, min_periods=_period(signal)).mean()
    histogram = line - signal_line
    return line, signal_line, histogram


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    p = _period(period)
    mid = sma(series, p)
    std = series.rolling(p, min_periods=p).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def highest_high(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(_period(period), min_periods=_period(period)).max()


def lowest_low(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(_period(period), min_periods=_period(period)).min()


def average_volume(series: pd.Series, period: int = 20) -> pd.Series:
    return sma(series, period)


def candle_range(frame: pd.DataFrame) -> pd.Series:
    return frame["high"] - frame["low"]


def percentage_change(series: pd.Series, period: int = 1) -> pd.Series:
    return series.pct_change(_period(period)) * 100.0


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def build_indicator_series(frame: pd.DataFrame, kind: str, period: int = 14, column: str = "close", lookback: int | None = None) -> pd.Series:
    kind = kind.lower()
    p = _period(period)
    source = frame[column] if column in frame.columns else frame["close"]

    if kind == "sma":
        return sma(source, p)
    if kind == "ema":
        return ema(source, p)
    if kind == "wma":
        return wma(source, p)
    if kind == "rsi":
        return rsi(source, p)
    if kind == "vwap":
        return vwap(frame)
    if kind == "atr":
        return atr(frame, p)
    if kind == "macd":
        return macd(source)[0]
    if kind == "macd_signal":
        return macd(source)[1]
    if kind == "macd_histogram":
        return macd(source)[2]
    if kind == "bollinger_mid":
        return bollinger(source, p)[0]
    if kind == "bollinger_upper":
        return bollinger(source, p)[1]
    if kind == "bollinger_lower":
        return bollinger(source, p)[2]
    if kind == "highest_high":
        return highest_high(frame["high"], p)
    if kind == "lowest_low":
        return lowest_low(frame["low"], p)
    if kind == "average_volume":
        volume = frame["volume"] if "volume" in frame.columns else pd.Series(1.0, index=frame.index)
        return average_volume(volume, p)
    if kind == "candle_range":
        return candle_range(frame)
    if kind == "percentage_change":
        return percentage_change(source, p)
    raise KeyError(kind)


# Legacy PineScript/MQL5 adapters expect this mapping to contain callables
# accepting (series, period). Keep those functions intact and add the new
# generic visual-builder series separately.
INDICATOR_FUNCS = {
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "rsi": rsi,
}
def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where `a` crosses above `b`."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where `a` crosses below `b`."""
    return (a < b) & (a.shift(1) >= b.shift(1))


INDICATOR_FUNCS = {"sma": sma, "ema": ema, "wma": wma, "rsi": rsi}
