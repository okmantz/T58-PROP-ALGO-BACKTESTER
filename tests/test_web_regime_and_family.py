"""Tests for the Regime Survival Matrix and Family Diversity web routes."""
from __future__ import annotations

from pathlib import Path

from app.search.batch_runner import SearchStageConfig, run_search
from app.search.strategy_space import generate_search_space
from app.backtest.risk import RiskConfig as _RiskConfig
from app.prop.simulator import PropRules as _PropRules
from app.web.server import REGIME_DIR, SEARCH_DIR, app

_SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"


def test_regime_matrix_form_loads():
    client = app.test_client()
    r = client.get("/regime-matrix")
    assert r.status_code == 200
    assert b"Regime survival matrix" in r.data


def test_regime_matrix_run_end_to_end_with_manual_strategy():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "strategy_mode": "manual",
            "sma_fast": "10", "sma_slow": "30", "sl_pips": "20", "tp_pips": "40",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "pip_size": "0.0001", "dimension_a": "volatility", "dimension_b": "environment",
        }
        r = client.post("/regime-matrix/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert ("Verdict" in body) or ("Not enough bars" in body)


def test_regime_matrix_run_rejects_identical_dimensions():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "strategy_mode": "manual",
            "sma_fast": "10", "sma_slow": "30", "sl_pips": "20", "tp_pips": "40",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "pip_size": "0.0001", "dimension_a": "volatility", "dimension_b": "volatility",
        }
        r = client.post("/regime-matrix/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"DIFFERENT dimensions" in r.data


def test_family_diversity_form_loads_with_no_runs():
    client = app.test_client()
    r = client.get("/family-diversity")
    assert r.status_code == 200
    assert b"Strategy family diversity" in r.data


def test_family_diversity_shows_a_real_run(tmp_path):
    """End-to-end: run a tiny real Search Lab search (writing its db into
    the app's actual SEARCH_DIR, exactly like run_search's own web job
    does), then confirm /family-diversity can find and render it."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(9)
    n = 2500
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = 0.00015 + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    space = generate_search_space(mode="family", family="trend_breakout", max_candidates=8, seed=1)
    cfg = SearchStageConfig(
        min_trades=3, min_profit_factor=0.5, max_drawdown_buffer_mult=5.0,
        stage1_top_n=6, ga_population=4, ga_generations=1, ga_search_sims=30,
        stage2_top_n=3, full_mc_sims=50, walk_forward_folds=0, robustness_neighbors=0,
        workers=1, random_seed=42,
    )
    db_path = SEARCH_DIR / "search_test_family_diversity.db"
    summary = run_search(
        df, _RiskConfig(), _PropRules(), space, cfg,
        db_path=str(db_path), instrument="TEST", timeframe="5m",
    )
    assert summary.stage1_survivors > 0  # this data/family combo must produce real survivors

    client = app.test_client()
    r = client.get(f"/family-diversity?db_path={db_path}&run_id={summary.run_id}&stage=stage1")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Best strategy family for this market/data" in body

    db_path.unlink(missing_ok=True)
