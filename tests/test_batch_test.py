import shutil

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.orchestration.batch_test import BatchTestItem, run_batch_test
from app.prop.simulator import PropRules
from app.strategy import library
from app.strategy.manual import ManualStrategy


@pytest.fixture(autouse=True)
def clean_library_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "get_app_base_dir", lambda: tmp_path)
    base_dir = library.get_strategy_library_dir()
    yield
    shutil.rmtree(base_dir, ignore_errors=True)


def _trending_df(n=1200, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift * (1 if (i // 40) % 2 == 0 else -1) + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(name="sma cross"):
    return {
        "name": name,
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def _flat_config():
    return {"name": "no trades", "long_entry": "close > 999999", "long_exit": "close < 0",
            "short_entry": "close < -999999", "short_exit": "close > 0"}


def test_run_batch_test_produces_one_report_per_strategy(tmp_path):
    df = _trending_df()
    risk = RiskConfig()
    rules = PropRules()
    items = [
        BatchTestItem(label="strategy_a", strategy=ManualStrategy(_sma_config("a"))),
        BatchTestItem(label="strategy_b", strategy=ManualStrategy(_sma_config("b"))),
    ]

    summary = run_batch_test(df, items, risk, rules, tmp_path, mc_sims=50)

    assert len(summary.outcomes) == 2
    assert len(summary.succeeded) == 2
    for outcome in summary.outcomes:
        assert outcome.report_html is not None
        assert outcome.report_html.exists()


def test_run_batch_test_keeps_going_after_a_zero_trade_strategy(tmp_path):
    df = _trending_df()
    risk = RiskConfig()
    rules = PropRules()
    items = [
        BatchTestItem(label="dead_strategy", strategy=ManualStrategy(_flat_config())),
        BatchTestItem(label="live_strategy", strategy=ManualStrategy(_sma_config("live"))),
    ]

    summary = run_batch_test(df, items, risk, rules, tmp_path, mc_sims=50)

    assert len(summary.outcomes) == 2
    assert len(summary.failed) == 1
    assert summary.failed[0].label == "dead_strategy"
    assert len(summary.succeeded) == 1
    assert summary.succeeded[0].label == "live_strategy"


def test_run_batch_test_records_result_onto_library_metadata(tmp_path):
    from app.strategy.python import PythonStrategy

    src = (
        "STRATEGY_NAME = \"Test EMA Cross\"\n"
        "EMA_FAST = 5\nEMA_SLOW = 15\nSTOP_LOSS_PIPS = 20\nTAKE_PROFIT_PIPS = 40\n\n"
        "def generate_signals(df):\n"
        "    fast = df[\"close\"].ewm(span=EMA_FAST, adjust=False).mean()\n"
        "    slow = df[\"close\"].ewm(span=EMA_SLOW, adjust=False).mean()\n"
        "    return (fast > slow).astype(int) - (fast < slow).astype(int)\n"
    )
    saved_path = library.save_strategy_text(src, "batch_test_strat.py", "python")
    df = _trending_df(n=1500, seed=4)
    risk = RiskConfig()
    rules = PropRules()
    items = [BatchTestItem(
        label="batch_test_strat.py", strategy=PythonStrategy(saved_path),
        library_ref=("python", "batch_test_strat.py"),
    )]

    summary = run_batch_test(df, items, risk, rules, tmp_path, mc_sims=50)

    assert len(summary.succeeded) == 1
    meta = library.load_strategy_metadata("python", "batch_test_strat.py")
    assert "last_run" in meta
    assert meta["last_run"]["trades"] == summary.succeeded[0].trades
