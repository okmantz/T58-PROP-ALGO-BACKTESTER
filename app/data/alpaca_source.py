"""Fetch OHLCV market data from Alpaca for use as backtest input.

This mirrors the data-fetch pattern from Owen's falsification-kit-tester
(inplay_screener/fetch_5m.py, honest_fill_sim/fetch_cache.py): alpaca-py's
StockHistoricalDataClient/CryptoHistoricalDataClient + *BarsRequest +
TimeFrame + Adjustment/DataFeed. The kit's scripts are one-shot CLI tools
that read ALPACA_API_KEY/ALPACA_SECRET_KEY from the environment and raise
SystemExit if they're missing; this module instead takes keys as plain
arguments (supplied interactively via the UI, see app/data/alpaca_credentials.py
for how they're saved) and returns a DataFrame in this app's standard
OHLCV schema (timestamp, open, high, low, close, volume) -- ready to feed
straight into app.data.importer / app.data.storage exactly like any
manually-imported CSV.

Requires the optional `alpaca-py` package (see config/requirements.txt). The
import is deferred into each function so the rest of the app still works
if it isn't installed; callers should catch AlpacaImportError and show it
as a plain message rather than a stack trace.

Note: Alpaca's market data API covers US equities and crypto. It does not
provide forex, CFD, or futures data (the instruments most of Owen's other
prop-firm work trades) -- for those, the existing "import CSV" / raw-file
path in app/data/storage.py remains the way to get data in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.storage import get_raw_data_dir

# (label, amount, TimeFrameUnit name) -- label is what the UI shows/stores.
TIMEFRAME_CHOICES = [
    ("1Min", 1, "Minute"),
    ("5Min", 5, "Minute"),
    ("15Min", 15, "Minute"),
    ("30Min", 30, "Minute"),
    ("1Hour", 1, "Hour"),
    ("1Day", 1, "Day"),
]
TIMEFRAME_LABELS = [t[0] for t in TIMEFRAME_CHOICES]

ASSET_CLASSES = ["Stock", "Crypto"]
FEED_CHOICES = ["iex", "sip"]  # iex = free plan; sip = paid market-data subscription
ADJUSTMENT_CHOICES = ["raw", "split", "dividend", "all"]


class AlpacaImportError(RuntimeError):
    """Raised when the optional alpaca-py dependency isn't installed."""


class AlpacaFetchError(RuntimeError):
    """Raised when a request to Alpaca fails (bad keys, bad symbol, no
    data returned for the range, network error, etc.)."""


def _require_alpaca() -> None:
    try:
        import alpaca  # noqa: F401
    except ImportError as exc:
        raise AlpacaImportError(
            "The 'alpaca-py' package isn't installed. Run "
            "`pip install alpaca-py` (or rebuild the .exe with it bundled) "
            "to enable fetching data from Alpaca."
        ) from exc


def _timeframe_object(label: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    for name, amount, unit in TIMEFRAME_CHOICES:
        if name == label:
            return TimeFrame(amount, getattr(TimeFrameUnit, unit))
    raise AlpacaFetchError(
        f"Unknown timeframe '{label}'. Choose one of: {', '.join(TIMEFRAME_LABELS)}."
    )


def test_connection(api_key: str, secret_key: str) -> str:
    """Verifies the keys work by hitting the (paper) trading account
    endpoint -- the same account a free Alpaca paper account provides, per
    the falsification kit's own setup instructions. Raises
    AlpacaImportError/AlpacaFetchError on failure; returns a short
    human-readable success message otherwise."""
    _require_alpaca()
    from alpaca.trading.client import TradingClient

    if not api_key or not secret_key:
        raise AlpacaFetchError("Enter both an API key and a secret key first.")
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        account = client.get_account()
        return f"Connected. Account status: {account.status}."
    except Exception as exc:
        raise AlpacaFetchError(f"Could not connect to Alpaca: {exc}") from exc


def _normalize_bars(raw: "pd.DataFrame", symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise AlpacaFetchError(
            f"Alpaca returned no bars for {symbol} in that date range/timeframe -- "
            "try a wider date range, a coarser timeframe, or double-check the symbol."
        )
    df = raw.reset_index()
    if "symbol" in df.columns:
        # A single-symbol request still comes back with a symbol column in
        # alpaca-py's multi-index frame; filter defensively in case a
        # comma-separated symbol string ever slips through.
        df = df[df["symbol"] == symbol]
    missing = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise AlpacaFetchError(f"Unexpected response shape from Alpaca (missing columns: {missing}).")
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise AlpacaFetchError(f"Alpaca returned no bars for {symbol} in that date range.")
    return out


def fetch_stock_bars(
    api_key: str,
    secret_key: str,
    symbol: str,
    timeframe_label: str,
    start: str,
    end: str,
    feed: str = "iex",
    adjustment: str = "raw",
) -> pd.DataFrame:
    """Fetches historical stock bars for one symbol. start/end are
    'YYYY-MM-DD' strings (or anything pandas.Timestamp can parse)."""
    _require_alpaca()
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    tf = _timeframe_object(timeframe_label)
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except Exception as exc:
        raise AlpacaFetchError(f"Could not parse start/end date: {exc}") from exc

    client = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_ts,
        end=end_ts,
        feed=DataFeed(feed),
        adjustment=Adjustment(adjustment),
    )
    try:
        raw = client.get_stock_bars(req).df
    except Exception as exc:
        raise AlpacaFetchError(f"Alpaca request failed for {symbol}: {exc}") from exc
    return _normalize_bars(raw, symbol)


def fetch_crypto_bars(
    api_key: str,
    secret_key: str,
    symbol: str,
    timeframe_label: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetches historical crypto bars for one symbol (e.g. 'BTC/USD').
    Crypto data on Alpaca has no feed/adjustment concept, unlike stocks."""
    _require_alpaca()
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest

    tf = _timeframe_object(timeframe_label)
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except Exception as exc:
        raise AlpacaFetchError(f"Could not parse start/end date: {exc}") from exc

    client = CryptoHistoricalDataClient(api_key, secret_key)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start_ts, end=end_ts)
    try:
        raw = client.get_crypto_bars(req).df
    except Exception as exc:
        raise AlpacaFetchError(f"Alpaca request failed for {symbol}: {exc}") from exc
    return _normalize_bars(raw, symbol)


def fetch_bars(
    api_key: str,
    secret_key: str,
    symbol: str,
    asset_class: str,
    timeframe_label: str,
    start: str,
    end: str,
    feed: str = "iex",
    adjustment: str = "raw",
) -> pd.DataFrame:
    """Dispatches to fetch_stock_bars or fetch_crypto_bars by asset_class
    ('Stock' or 'Crypto') -- the single entry point the UI calls."""
    if asset_class.lower().startswith("crypto"):
        return fetch_crypto_bars(api_key, secret_key, symbol, timeframe_label, start, end)
    return fetch_stock_bars(api_key, secret_key, symbol, timeframe_label, start, end, feed, adjustment)


def save_bars_as_csv(df: pd.DataFrame, symbol: str, timeframe_label: str) -> Path:
    """Writes fetched bars into data/raw/<SYMBOL>/, alongside any manually
    imported CSVs for the same instrument -- so it shows up for free in the
    existing "Market Data Library" grouping (app.data.storage) and dataset
    list/multi-select, with no changes needed to that selection UI."""
    safe_symbol = "".join(c for c in symbol.upper() if c.isalnum() or c in ("-", "_")) or "SYMBOL"
    folder = get_raw_data_dir() / safe_symbol
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{safe_symbol}_{timeframe_label}_alpaca.csv"
    df.to_csv(dest, index=False)
    return dest
