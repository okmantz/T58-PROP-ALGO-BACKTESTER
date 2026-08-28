"""Tests for app.data.pairs.merge_pair_series."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.pairs import PairDataError, merge_pair_series


def _df(timestamps, closes):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_merge_is_backward_looking_and_forward_fills():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10", "2023-01-01 00:15"], [1, 2, 3, 4])
    pair_df = _df(["2023-01-01 00:00", "2023-01-01 00:10"], [100, 200])
    merged = merge_pair_series(df, pair_df)
    assert list(merged["pair_close"]) == [100, 100, 200, 200]


def test_bars_before_pair_data_starts_get_nan():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"], [1, 2, 3])
    pair_df = _df(["2023-01-01 00:05"], [50])
    merged = merge_pair_series(df, pair_df)
    assert pd.isna(merged["pair_close"].iloc[0])
    assert merged["pair_close"].iloc[1] == 50
    assert merged["pair_close"].iloc[2] == 50


def test_original_row_order_and_columns_preserved():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05", "2023-01-01 00:10"], [1, 2, 3])
    pair_df = _df(["2023-01-01 00:00"], [10])
    merged = merge_pair_series(df, pair_df)
    assert list(merged["close"]) == [1, 2, 3]  # original data untouched, in original order
    assert list(merged.index) == list(df.index)


def test_custom_column_name():
    df = _df(["2023-01-01 00:00"], [1])
    pair_df = _df(["2023-01-01 00:00"], [99])
    merged = merge_pair_series(df, pair_df, column_name="dxy_close")
    assert "dxy_close" in merged.columns
    assert merged["dxy_close"].iloc[0] == 99


def test_missing_timestamp_column_raises():
    df = pd.DataFrame({"close": [1, 2, 3]})
    pair_df = _df(["2023-01-01 00:00"], [1])
    with pytest.raises(PairDataError):
        merge_pair_series(df, pair_df)


def test_missing_close_column_on_pair_df_raises():
    df = _df(["2023-01-01 00:00"], [1])
    pair_df = pd.DataFrame({"timestamp": pd.to_datetime(["2023-01-01 00:00"])})
    with pytest.raises(PairDataError):
        merge_pair_series(df, pair_df)


def test_original_df_not_mutated():
    df = _df(["2023-01-01 00:00", "2023-01-01 00:05"], [1, 2])
    pair_df = _df(["2023-01-01 00:00"], [10])
    merge_pair_series(df, pair_df)
    assert "pair_close" not in df.columns
