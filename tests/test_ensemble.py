"""Tests for app.ensemble.ensemble (blend and vote modes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.ensemble.ensemble import (
    EnsembleError, EnsembleVoteConfig, build_ensemble_legs, run_ensemble_blend, run_ensemble_vote,
)
from app.portfolio.portfolio import PortfolioConfig
from app.strategy.base import Strategy, StrategyResult
from app.strategy.manual import ManualStrategy


def _synthetic_df(n=1000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    drift = np.cumsum(rng.normal(0.02, 0.4, n))
    close = 100 + drift
    high = close + rng.random(n) * 0.3
    low = close - rng.random(n) * 0.3
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


class _ConstantSignalStrategy(Strategy):
    """Test double: ignores price data entirely and always emits the same
    signal -- makes vote-mode combination logic fully deterministic to test."""

    source_type = "manual"

    def __init__(self, signal_value: int, name: str = "constant"):
        self.signal_value = signal_value
        self.config = {"name": name}

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        return StrategyResult(
            name=self.config["name"],
            source_type=self.source_type,
            signals=pd.Series(self.signal_value, index=df.index),
            stop_loss_pips=20.0,
            take_profit_pips=40.0,
        )


def _manual_ema_cross(fast: int, slow: int, name: str) -> ManualStrategy:
    return ManualStrategy({
        "name": name,
        "indicators": [
            {"type": "ema", "period": fast, "column": "close", "as": "f"},
            {"type": "ema", "period": slow, "column": "close", "as": "s"},
        ],
        "long_entry": "f > s", "long_exit": "f < s",
        "short_entry": "f < s", "short_exit": "f > s",
        "stop_loss_pips": 200, "take_profit_pips": 400,
    })


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_requires_at_least_two_legs():
    df = _synthetic_df()
    with pytest.raises(EnsembleError):
        build_ensemble_legs(df, [_manual_ema_cross(5, 20, "only_one")], RiskConfig())


def test_duplicate_leg_names_rejected():
    df = _synthetic_df()
    strategies = [_manual_ema_cross(5, 20, "same"), _manual_ema_cross(10, 40, "same")]
    with pytest.raises(EnsembleError):
        build_ensemble_legs(df, strategies, RiskConfig())


def test_mismatched_names_length_rejected():
    df = _synthetic_df()
    strategies = [_manual_ema_cross(5, 20, "a"), _manual_ema_cross(10, 40, "b")]
    with pytest.raises(EnsembleError):
        build_ensemble_legs(df, strategies, RiskConfig(), names=["only_one_name"])


def test_default_leg_names_are_unique_and_derived_from_strategy():
    df = _synthetic_df()
    strategies = [_manual_ema_cross(5, 20, "Fast Cross"), _manual_ema_cross(10, 40, "Slow Cross")]
    legs = build_ensemble_legs(df, strategies, RiskConfig())
    assert [leg.name for leg in legs] == ["Fast Cross", "Slow Cross"]


# ---------------------------------------------------------------------------
# Blend mode
# ---------------------------------------------------------------------------

def test_blend_mode_combines_two_different_strategies_on_same_instrument():
    df = _synthetic_df()
    strategies = [_manual_ema_cross(5, 20, "fast"), _manual_ema_cross(15, 60, "slow")]
    result = run_ensemble_blend(df, strategies, RiskConfig(), config=PortfolioConfig(initial_balance=100_000.0))
    assert len(result.legs) == 2
    assert {leg.name for leg in result.legs} == {"fast", "slow"}
    assert result.combined_statistics is not None
    # the equity curve rebuild is chronological by exit time, regardless of
    # per-leg trade-list order (see app.portfolio.portfolio._rebuild_equity_curve)
    equity_times = pd.to_datetime(result.combined_equity_curve["timestamp"])
    assert list(equity_times) == sorted(equity_times)


def test_blend_mode_every_leg_points_at_the_identical_df():
    """The whole point of 'blend' mode vs. the existing multi-asset
    Portfolio feature is that every leg shares ONE instrument."""
    df = _synthetic_df()
    strategies = [_manual_ema_cross(5, 20, "fast"), _manual_ema_cross(15, 60, "slow")]
    legs = build_ensemble_legs(df, strategies, RiskConfig())
    assert legs[0].df is df
    assert legs[1].df is df


# ---------------------------------------------------------------------------
# Vote mode
# ---------------------------------------------------------------------------

def test_vote_mode_requires_min_agreement_before_entering():
    df = _synthetic_df(n=50)
    always_long = _ConstantSignalStrategy(1, "always_long")
    always_short = _ConstantSignalStrategy(-1, "always_short")
    # 2 legs disagreeing on every bar, min_agreement=2 -> can never agree -> zero trades
    result = run_ensemble_vote(df, [always_long, always_short], RiskConfig(), vote_config=EnsembleVoteConfig(min_agreement=2))
    assert len(result.trades) == 0


def test_vote_mode_enters_once_enough_legs_agree():
    df = _synthetic_df(n=50)
    legs = [_ConstantSignalStrategy(1, "long_a"), _ConstantSignalStrategy(1, "long_b"), _ConstantSignalStrategy(-1, "short_c")]
    # 2-of-3 agree long every single bar -> should hold one continuous long position
    result = run_ensemble_vote(df, legs, RiskConfig(), vote_config=EnsembleVoteConfig(min_agreement=2))
    assert len(result.trades) >= 1
    assert all(t.direction == 1 for t in result.trades)


def test_vote_mode_conflict_stays_flat_even_at_min_agreement_1():
    df = _synthetic_df(n=50)
    always_long = _ConstantSignalStrategy(1, "always_long")
    always_short = _ConstantSignalStrategy(-1, "always_short")
    # min_agreement=1: both directions clear the threshold on every bar --
    # that is a conflict, not a consensus, so it must stay flat rather than
    # arbitrarily picking a side.
    result = run_ensemble_vote(df, [always_long, always_short], RiskConfig(), vote_config=EnsembleVoteConfig(min_agreement=1))
    assert len(result.trades) == 0


def test_vote_mode_inherits_risk_management_from_first_leg():
    df = _synthetic_df(n=50)
    first = _ConstantSignalStrategy(1, "first")
    first.__class__  # no-op, just documenting which leg's risk wins
    second = _ConstantSignalStrategy(1, "second")
    result = run_ensemble_vote(df, [first, second], RiskConfig(), vote_config=EnsembleVoteConfig(min_agreement=1))
    assert len(result.trades) >= 1
    # first leg's stop_loss_pips=20 -> initial_risk should reflect a 20-pip stop
    assert result.trades[0].initial_risk == pytest.approx(20 * RiskConfig().pip_size)


def test_min_agreement_cannot_exceed_leg_count():
    df = _synthetic_df(n=50)
    legs = [_ConstantSignalStrategy(1, "a"), _ConstantSignalStrategy(1, "b")]
    with pytest.raises(EnsembleError):
        run_ensemble_vote(df, legs, RiskConfig(), vote_config=EnsembleVoteConfig(min_agreement=3))
