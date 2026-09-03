"""Tests for the Stage 3 early-kill floor added to app.search.batch_runner:
a candidate that fails a basic trades/profit-factor/drawdown floor on its
plain backtest should be rejected BEFORE Monte Carlo/walk-forward/
robustness ever run, not after.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from app.search.batch_runner import _init_worker, _stage3_task


def _trending_df(n=400, seed=3, drift=0.00015):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.1000
    rows = []
    for i in range(n):
        step = drift + rng.normal(0, 0.00003)
        o = price
        c = o + step
        h = max(o, c) + abs(rng.normal(0, 0.00002))
        l = min(o, c) - abs(rng.normal(0, 0.00002))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _base_cfg(**overrides):
    cfg = {
        "full_mc_sims": 20, "random_seed": 1, "fitness_metric": "composite_prop_score",
        "walk_forward_folds": 0, "walk_forward_metric": "profit_factor",
        "walk_forward_min_efficiency": 0.4, "robustness_neighbors": 0,
        "robustness_perturbation_frac": 0.15, "robustness_min_stability": 0.4,
        "stage3_min_trades": 20, "stage3_min_profit_factor": 1.05,
        "stage3_max_drawdown_buffer_mult": 1.5,
    }
    cfg.update(overrides)
    return cfg


def _run_stage3(spec, cfg, df):
    with tempfile.TemporaryDirectory() as td:
        df_path = f"{td}/data.pkl"
        df.to_pickle(df_path)
        _init_worker(df_path, {}, {}, td)
        return _stage3_task("cand-1", spec, cfg)


# A strategy that trades constantly but has no real edge (buys every bar) --
# will clear "produces trades" but should fail the min-trades-vs-profit-factor
# floor cleanly rather than reaching Monte Carlo.
_WEAK_SPEC = {
    "source_type": "manual",
    "config": {
        "name": "always-long",
        "long_entry": "close > 0",
        "risk_management": {
            "stop_type": "fixed", "stop_value": 5, "target_type": "fixed", "target_value": 2,
        },
    },
}


def test_candidate_failing_min_trades_is_rejected_before_monte_carlo():
    df = _trending_df(n=60)  # too short to generate 20+ trades for most configs
    cfg = _base_cfg(stage3_min_trades=10_000)  # impossible floor -> guaranteed early-kill
    result = _run_stage3(_WEAK_SPEC, cfg, df)
    assert result["passed_stage3_gate"] is False
    assert "early-kill floor" in (result.get("error") or "")
    # The expensive stages must never have run -- their summary keys are absent entirely
    # (a real Stage 3 pass populates mc_summary/prop_summary/lookahead).
    assert "mc_summary" not in result
    assert "lookahead" not in result


def test_candidate_clearing_the_floor_still_runs_the_full_gate():
    df = _trending_df(n=1500)
    cfg = _base_cfg(stage3_min_trades=1, stage3_min_profit_factor=0.0, stage3_max_drawdown_buffer_mult=100.0)
    result = _run_stage3(_WEAK_SPEC, cfg, df)
    # With a near-zero floor, the candidate should proceed into the real gate
    # (mc_summary/lookahead keys present), whatever the eventual pass/fail verdict is.
    if result.get("statistics", {}).get("total_trades"):
        assert "mc_summary" in result
        assert "lookahead" in result


def test_early_kill_defaults_apply_when_keys_are_missing_from_cfg():
    """Backward compatibility: older cfg dicts (or direct callers) that don't
    set the new stage3_min_trades/stage3_min_profit_factor/
    stage3_max_drawdown_buffer_mult keys must still work, using the
    documented defaults (20 / 1.05 / 1.5x) rather than crashing on a KeyError."""
    df = _trending_df(n=60)
    cfg = {
        "full_mc_sims": 20, "random_seed": 1, "fitness_metric": "composite_prop_score",
        "walk_forward_folds": 0, "walk_forward_metric": "profit_factor",
        "walk_forward_min_efficiency": 0.4, "robustness_neighbors": 0,
        "robustness_perturbation_frac": 0.15, "robustness_min_stability": 0.4,
    }
    result = _run_stage3(_WEAK_SPEC, cfg, df)
    assert "passed_stage3_gate" in result  # must not raise KeyError
