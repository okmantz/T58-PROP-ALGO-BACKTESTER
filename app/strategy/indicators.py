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


def relative_volume(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current bar's volume divided by its own trailing average -- >1 means
    this bar traded on above-average participation. Falls back to a
    constant 1.0 series when the data has no volume column (e.g. some FX
    feeds), so a strategy that references it simply never fires rather
    than crashing."""
    if "volume" not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    volume = frame["volume"]
    avg = average_volume(volume, period)
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


def volume_delta(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling sum of signed volume (candle-direction-signed: up-close bars
    contribute +volume, down-close bars contribute -volume) over `period`
    bars, normalized by the rolling sum of total volume so the result is a
    dimensionless imbalance ratio in roughly [-1, 1] regardless of the
    instrument's absolute volume scale. A simple, transparent proxy for
    order-flow / buying-vs-selling pressure imbalance -- not a true
    tick-level bid/ask delta (this app only has OHLCV bars, not trade-by-
    trade prints), but directionally meaningful and, crucially, computed
    only from already-closed bars (no lookahead)."""
    if "volume" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    signed = frame["volume"].where(frame["close"] >= frame["open"], -frame["volume"])
    p = _period(period)
    signed_sum = signed.rolling(p, min_periods=p).sum()
    total_sum = frame["volume"].rolling(p, min_periods=p).sum()
    return (signed_sum / total_sum.replace(0, np.nan)).fillna(0.0)


def pair_ratio(frame: pd.DataFrame, pair_column: str = "pair_close") -> pd.Series:
    """Price ratio of this instrument's close to a second instrument's close
    that has already been merged into `frame` as `pair_column` (see
    app.data.pairs.merge_pair_series). Requires the merge step to have
    happened first -- raises KeyError otherwise so a mis-set-up pairs
    strategy fails loudly instead of silently trading on garbage."""
    if pair_column not in frame.columns:
        raise KeyError(
            f"'{pair_column}' not found in market data -- a pairs/relative-value "
            "strategy requires the second instrument's close to be merged in first "
            "via app.data.pairs.merge_pair_series()."
        )
    return frame["close"] / frame[pair_column].replace(0, np.nan)


def pair_zscore(frame: pd.DataFrame, period: int = 50, pair_column: str = "pair_close") -> pd.Series:
    """Rolling z-score of the two-instrument price ratio -- the standard
    statistical-arbitrage signal: how many standard deviations the current
    ratio sits from its own trailing mean. A large positive/negative
    z-score is the classic 'spread has stretched, bet on reversion' entry
    trigger for a pairs strategy."""
    ratio = pair_ratio(frame, pair_column)
    p = _period(period)
    mean = ratio.rolling(p, min_periods=p).mean()
    std = ratio.rolling(p, min_periods=p).std(ddof=0)
    return ((ratio - mean) / std.replace(0, np.nan)).fillna(0.0)


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def build_indicator_series(frame: pd.DataFrame, kind: str, period: int = 14, column: str = "close", lookback: int | None = None) -> pd.Series:
    """Thin caching wrapper -- see app.strategy.indicator_cache for why.
    The actual per-kind math is unchanged, in _build_indicator_series_uncached
    below; every existing caller and behavior is identical, just memoized
    per (frame, kind, period, column, lookback) within this process."""
    from app.strategy import indicator_cache

    return indicator_cache.get_or_compute(
        frame, kind, period, column, lookback,
        compute_fn=lambda: _build_indicator_series_uncached(frame, kind, period, column, lookback),
    )


def _build_indicator_series_uncached(frame: pd.DataFrame, kind: str, period: int = 14, column: str = "close", lookback: int | None = None) -> pd.Series:
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
    if kind == "relative_volume":
        return relative_volume(frame, p)
    if kind == "volume_delta":
        return volume_delta(frame, p)
    if kind == "pair_ratio":
        return pair_ratio(frame)
    if kind == "pair_zscore":
        return pair_zscore(frame, p)
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
