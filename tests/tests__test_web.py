from pathlib import Path

from app.web.server import app, REPORTS_DIR
from app.data import storage


def test_index_loads():
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"T58" in r.data


def test_manifest_served():
    client = app.test_client()
    r = client.get("/manifest.json")
    assert r.status_code == 200
    assert r.content_type == "application/manifest+json"


def test_full_pipeline_via_manual_strategy(tmp_path):
    client = app.test_client()
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

    with open(sample_csv, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "strategy_mode": "manual",
            "sma_fast": "20", "sma_slow": "50", "sl_pips": "20", "tp_pips": "40",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "100", "mc_method": "bootstrap",
        }
        r = client.post("/run", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Eval Pass Probability" in body

    # clean up any report files this test produced
    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


def test_missing_csv_returns_error():
    client = app.test_client()
    r = client.post("/run", data={"strategy_mode": "manual"}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"CSV" in r.data or b"dataset" in r.data


_BIGGER_CSV = None


def _make_bigger_csv(seed: int) -> bytes:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0005, "low": price - 0.0005,
        "close": price, "volume": 100.0,
    })
    return df.to_csv(index=False).encode()


def test_multi_file_upload_stores_all_and_uses_last_as_active():
    import shutil
    raw_dir = storage.get_raw_data_dir()
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        import io
        client = app.test_client()
        csv_a = _make_bigger_csv(1)
        csv_b = _make_bigger_csv(2)
        data = {
            "csv_file": [(io.BytesIO(csv_a), "setA.csv"), (io.BytesIO(csv_b), "setB.csv")],
            "strategy_mode": "manual", "sma_fast": "5", "sma_slow": "15", "sl_pips": "20", "tp_pips": "40",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "50", "mc_method": "bootstrap",
        }
        r = client.post("/run", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        stored_names = sorted(p.name for p in raw_dir.glob("*.csv"))
        assert stored_names == ["setA.csv", "setB.csv"]
        body = r.get_data(as_text=True)
        assert "setB.csv" in body  # most recently uploaded file becomes active
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        for f in REPORTS_DIR.glob("report_*"):
            f.unlink()


def test_run_against_existing_stored_dataset_without_new_upload():
    import shutil
    raw_dir = storage.get_raw_data_dir()
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        storage.store_csv_bytes(_make_bigger_csv(3), "stored.csv")
        client = app.test_client()
        data = {
            "existing_dataset": "stored.csv",
            "strategy_mode": "manual", "sma_fast": "5", "sma_slow": "15", "sl_pips": "20", "tp_pips": "40",
            "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
            "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
            "payout_threshold": "0", "buffer": "0", "payout_cap": "",
            "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
            "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
            "spread_pips": "1.0", "pip_size": "0.0001",
            "n_sims": "50", "mc_method": "bootstrap",
        }
        r = client.post("/run", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        assert "stored.csv" in r.get_data(as_text=True)
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)
        for f in REPORTS_DIR.glob("report_*"):
            f.unlink()
