"""Tests for app.search.strategy_space."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.search.strategy_space import (
    FAMILIES, StrategySpaceError, family_description, family_grid_size,
    generate_search_space, list_families,
)
from app.strategy.manual import ManualStrategy


def _synthetic_df(n=2500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.02, 0.5, n))
    close = 100 + drift + np.sin(np.arange(n) / 50) * 2
    high = close + rng.random(n) * 0.5
    low = close - rng.random(n) * 0.5
    openp = close + rng.normal(0, 0.1, n)
    vol = rng.random(n) * 1000
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close, "volume": vol})


def test_list_families_matches_registry():
    families = list_families()
    assert set(families) == set(FAMILIES)
    for name in families:
        assert family_description(name)
        assert family_grid_size(name) > 0


def test_unknown_family_raises():
    with pytest.raises(StrategySpaceError):
        generate_search_space("family", family="not_a_real_family")


def test_single_mode_wraps_exactly_one_config():
    cfg = {
        "name": "SMA Cross",
        "indicators": [{"type": "sma", "period": 10, "column": "close", "as": "s"}],
        "long_entry": "s > close", "stop_loss_pips": 20, "take_profit_pips": 40,
    }
    space = generate_search_space("single", single_config=cfg)
    assert space.mode == "single"
    assert len(space.candidates) == 1
    (cid, out_cfg), = space.candidates.items()
    assert out_cfg["name"] == "SMA Cross"
    # Must be a deep copy, not the same object, so later mutation (e.g. Stage 2's
    # GA writing tuned parameters back) never mutates the caller's original config.
    out_cfg["name"] = "mutated"
    assert cfg["name"] == "SMA Cross"


def test_single_mode_requires_a_config():
    with pytest.raises(StrategySpaceError):
        generate_search_space("single", single_config=None)


def test_family_mode_generates_full_grid_when_under_cap():
    space = generate_search_space("family", family="mean_reversion_band", max_candidates=10_000, seed=1)
    assert space.mode == "family"
    assert space.sampled is False
    assert len(space.candidates) == space.total_generated == family_grid_size("mean_reversion_band")


def test_family_mode_samples_reproducibly_when_over_cap():
    space_a = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=7)
    space_b = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=7)
    assert space_a.sampled is True
    assert len(space_a.candidates) == 20
    assert set(space_a.candidates.keys()) == set(space_b.candidates.keys())
    # IDs are content-addressed (hash of family + params), not positional --
    # so same seed must also reproduce the exact same underlying configs,
    # not just the same-shaped ID set.
    assert space_a.candidates == space_b.candidates
    # different seed -> (almost certainly) a different sample -> different IDs,
    # since IDs are derived from the params themselves.
    space_c = generate_search_space("family", family="trend_breakout", max_candidates=20, seed=99)
    assert set(space_a.candidates.keys()) != set(space_c.candidates.keys())


def test_family_all_combines_every_family():
    space = generate_search_space("family", family="all", max_candidates=100_000, seed=1)
    expected_total = sum(family_grid_size(name) for name in FAMILIES)
    assert space.total_generated == expected_total
    seen_families = {meta["family"] for meta in space.meta.values()}
    assert seen_families == set(FAMILIES)


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_every_family_produces_valid_runnable_configs(family_name):
    """
    Every generated config must be accepted by ManualStrategy and produce a
    finite result on real data -- this is also a regression guard for the
    "close > highest_high(window including current bar)" class of bug,
    which is structurally always-false and would otherwise show up as
    "zero trades" silently rather than as an explicit test failure.
    """
    df = _synthetic_df()
    risk = RiskConfig()
    space = generate_search_space("family", family=family_name, max_candidates=6, seed=3)
    assert len(space.candidates) > 0
    any_trades = False
    for cid, cfg in space.candidates.items():
        result = run_backtest(df, ManualStrategy(cfg), risk)
        assert result.statistics is not None
        if result.trades:
            any_trades = True
    assert any_trades, f"family '{family_name}' produced zero trades across every sampled candidate"
