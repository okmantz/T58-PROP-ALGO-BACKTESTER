"""
End-to-end tests for the Search Lab web routes (app.web.server).

These actually POST to /search/start, poll the real background job through
/search/job/<id>/status.json until it finishes, and exercise the promote
endpoint -- not mocks of the job system, the real threading.Thread +
run_search() path a browser would drive. Kept small-scale (few candidates,
low Monte Carlo sim counts, walk-forward/robustness disabled) so the whole
file finishes in a reasonable time under CI.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import app.web.server as server_module
from app.web.server import _SEARCH_JOBS, app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_PYTHON_SRC = '''STRATEGY_NAME = "Web Test EMA Cross"
EMA_FAST = 6
EMA_SLOW = 18
STOP_LOSS_PIPS = 18
TAKE_PROFIT_PIPS = 36

def generate_signals(df):
    fast = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    slow = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return (fast > slow).astype(int) - (fast < slow).astype(int)
'''

_MQL5_SRC = '''void OnTick() {
   double fastMA = iMA(_Symbol, PERIOD_CURRENT, 6, 0, MODE_EMA, PRICE_CLOSE);
   double slowMA = iMA(_Symbol, PERIOD_CURRENT, 18, 0, MODE_EMA, PRICE_CLOSE);
   if (fastMA > slowMA) { trade.Buy(0.1, _Symbol); }
   if (fastMA < slowMA) { trade.Sell(0.1, _Symbol); }
   // T58_SL_PIPS=18
   // T58_TP_PIPS=36
}
'''

_LOOSE_STAGE_FIELDS = {
    "max_candidates": "8", "seed": "1", "workers": "2",
    "min_trades": "1", "min_profit_factor": "0.0", "stage1_top_n": "5",
    "ga_population": "4", "ga_generations": "1", "stage2_top_n": "3",
    "full_mc_sims": "60", "walk_forward_folds": "0", "robustness_neighbors": "0",
    "fitness_metric": "composite_prop_score",
}


@pytest.fixture(autouse=True)
def _cleanup_search_artifacts(tmp_path, monkeypatch):
    """
    Isolate every test in this file from the real reports/search/
    directory. SEARCH_DIR resolves to <repo_root>/reports/search in normal
    (non-frozen) runs -- that's where a real user's past search-run
    databases and leaderboard reports live. Redirecting SEARCH_DIR to a
    pytest tmp_path for the duration of each test (rather than rmtree-ing
    the real one, which was this file's original approach) means these
    tests can freely create and destroy search artifacts with zero risk to
    a real user's search history.
    """
    isolated_search_dir = tmp_path / "search"
    isolated_search_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server_module, "SEARCH_DIR", isolated_search_dir)
    yield
    _SEARCH_JOBS.clear()


def _poll_until_done(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/search/job/{job_id}/status.json")
        data = r.get_json()
        assert data["found"] is True
        if data["done"]:
            return data
        time.sleep(0.3)
    raise AssertionError(f"search job {job_id} did not finish within {timeout}s")


def test_search_form_loads():
    client = app.test_client()
    r = client.get("/search")
    assert r.status_code == 200
    assert b"Search Lab" in r.data


def test_search_job_status_404_for_unknown_job():
    client = app.test_client()
    r = client.get("/search/job/does-not-exist/status.json")
    assert r.status_code == 404
    assert r.get_json()["found"] is False


def test_search_job_page_404_for_unknown_job():
    client = app.test_client()
    r = client.get("/search/job/does-not-exist")
    assert r.status_code == 404


def test_search_named_family_manual_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302  # redirected to the job page
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    job_page = client.get(f"/search/job/{job_id}")
    assert job_page.status_code == 200
    assert b"Search running" in job_page.data or b"Search complete" in job_page.data

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["mode"] == "family"
    assert status["summary"]["family"] == "trend_breakout"
    assert status["summary"]["total_candidates"] == 8
    assert isinstance(status["leaderboard"], list)

    if status["summary"]["champion_candidate_id"]:
        candidate_id = status["summary"]["champion_candidate_id"]
        promo = client.post(f"/search/job/{job_id}/promote", data={"candidate_id": candidate_id})
        assert promo.status_code == 200
        promo_data = promo.get_json()
        assert promo_data["ok"] is True
        report_url = promo_data["report_html"]
        report_resp = client.get(report_url)
        assert report_resp.status_code == 200


def test_search_family_grid_python_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_grid", "strategy_mode": "python",
            "strategy_code": _PYTHON_SRC, "grid_points": "2",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["family"] == "python_grid"
    for row in status["leaderboard"]:
        assert row["source_type"] == "python"


def test_search_single_mode_mql5_end_to_end():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "single", "strategy_mode": "mql5",
            "strategy_code": _MQL5_SRC,
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]

    status = _poll_until_done(client, job_id)
    assert status["error"] is None
    assert status["summary"]["mode"] == "single"
    assert status["summary"]["total_candidates"] == 1


def test_search_start_with_no_dataset_shows_error():
    client = app.test_client()
    r = client.post(
        "/search/start",
        data={"search_mode": "family_named", "family": "trend_breakout", **_LOOSE_STAGE_FIELDS},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    assert b"Please upload at least one valid CSV" in r.data


def test_search_promote_requires_candidate_id():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = {
            "csv_file": (f, "EURUSD_5M_sample.csv"),
            "search_mode": "family_named", "family": "trend_breakout",
            **_LOOSE_STAGE_FIELDS,
        }
        r = client.post("/search/start", data=data, content_type="multipart/form-data")
    job_id = r.headers["Location"].rstrip("/").split("/")[-1]
    _poll_until_done(client, job_id)

    promo = client.post(f"/search/job/{job_id}/promote", data={})
    assert promo.status_code == 400
    assert promo.get_json()["ok"] is False
