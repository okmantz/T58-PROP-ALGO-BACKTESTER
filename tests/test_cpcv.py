import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy
from app.validation.cpcv import CPCVError, compute_pbo, run_cpcv


def _trending_df(n=3000, seed=3, drift=0.00012):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00006)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00003))
        l = min(o, c) - abs(rng.normal(0, 0.00003))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _sma_config(fast=5, slow=15):
    return {
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": fast, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": slow, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 20,
        "take_profit_pips": 40,
    }


def test_run_cpcv_basic():
    df = _trending_df()
    result = run_cpcv(
        df, lambda: ManualStrategy(_sma_config()), RiskConfig(),
        n_groups=6, n_test_groups=2, metric="profit_factor", max_paths=10,
    )
    assert result.n_paths > 0
    assert result.n_paths <= 10
    assert isinstance(result.mean_oos_metric, float)
    assert 0.0 <= result.pct_paths_oos_negative <= 100.0


def test_run_cpcv_rejects_bad_grouping():
    df = _trending_df(n=500)
    with pytest.raises(CPCVError):
        run_cpcv(df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_groups=2, n_test_groups=2)


def test_run_cpcv_rejects_insufficient_data():
    df = _trending_df(n=50)
    with pytest.raises(CPCVError):
        run_cpcv(df, lambda: ManualStrategy(_sma_config()), RiskConfig(), n_groups=6, n_test_groups=2)


def test_compute_pbo_multi_candidate():
    df = _trending_df(n=3000, seed=9)
    specs = [
        {"source_type": "manual", "config": _sma_config(5, 15)},
        {"source_type": "manual", "config": _sma_config(8, 21)},
        {"source_type": "manual", "config": _sma_config(3, 50)},
    ]
    result = compute_pbo(df, specs, RiskConfig(), n_groups=6, n_test_groups=2, metric="sharpe_ratio", max_paths=10)
    assert result.n_candidates == 3
    assert 0.0 <= result.pbo <= 1.0
    assert len(result.mean_is_by_candidate) == 3
    assert len(result.mean_oos_by_candidate) == 3
    assert 0 <= result.overall_best_candidate_index < 3


def test_compute_pbo_single_candidate_is_degenerate_but_runs():
    df = _trending_df(n=2500, seed=11)
    specs = [{"source_type": "manual", "config": _sma_config()}]
    result = compute_pbo(df, specs, RiskConfig(), n_groups=6, n_test_groups=2, metric="profit_factor", max_paths=8)
    assert result.n_candidates == 1
    assert "degenerate" in result.note
