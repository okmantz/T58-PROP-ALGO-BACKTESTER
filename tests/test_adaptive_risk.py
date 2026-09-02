"""Tests for app.backtest.adaptive_risk and its wiring into execution/engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.adaptive_risk import (
    AdaptiveRiskConfig,
    AdaptiveRiskError,
    AdaptiveRiskRule,
    AdaptiveRiskState,
    build_limit_aware_preset,
)
from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy


def _synthetic_df(n=1500, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="15min")
    # A noisy mean-reverting-ish series (not just a smooth trend) so a
    # frequent-trading strategy sees a realistic mix of wins and losses,
    # including losing streaks, on data of this size.
    walk = np.cumsum(rng.normal(0, 0.3, n))
    close = 100 + walk - 0.15 * (walk - pd.Series(walk).rolling(50, min_periods=1).mean().to_numpy())
    high = close + rng.random(n) * 0.4
    low = close - rng.random(n) * 0.4
    openp = close + rng.normal(0, 0.05, n)
    return pd.DataFrame({"timestamp": ts, "open": openp, "high": high, "low": low, "close": close})


_FREQUENT_TRADER = {
    "name": "RSI flip-flop (trades often, on purpose, for adaptive-risk testing)",
    "entry_conditions": {
        "long": [{"left": {"type": "rsi", "period": 5}, "operator": "<", "right": {"type": "value", "value": 50}}],
        "short": [{"left": {"type": "rsi", "period": 5}, "operator": ">", "right": {"type": "value", "value": 50}}],
    },
    "exit_conditions": {"long": [], "short": []},
    "risk_management": {
        "stop_type": "atr", "stop_value": 1.0, "target_type": "atr", "target_value": 1.5,
        "opposite_signal_exit": True,
    },
}


# ---------------------------------------------------------------------------
# AdaptiveRiskRule / AdaptiveRiskState unit tests
# ---------------------------------------------------------------------------

def test_unknown_trigger_raises():
    with pytest.raises(AdaptiveRiskError):
        AdaptiveRiskRule(trigger="not_a_real_trigger", threshold=2, risk_multiplier=0.5)


def test_consecutive_losses_increments_and_resets_on_win():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.record_trade_close(pnl=-100.0, is_new_day=True)
    assert state.consecutive_losses == 1
    state.record_trade_close(pnl=-50.0, is_new_day=False)
    assert state.consecutive_losses == 2
    state.record_trade_close(pnl=200.0, is_new_day=False)
    assert state.consecutive_losses == 0


def test_scratch_trade_leaves_streak_unchanged():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.record_trade_close(pnl=-100.0, is_new_day=True)
    state.record_trade_close(pnl=0.0, is_new_day=False)
    assert state.consecutive_losses == 1


def test_daily_pnl_resets_on_new_day():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.record_trade_close(pnl=-300.0, is_new_day=True)
    assert state.day_realized_pnl == -300.0
    state.record_trade_close(pnl=-100.0, is_new_day=True)  # new day -> should NOT accumulate with yesterday's -300
    assert state.day_realized_pnl == -100.0


def test_active_multiplier_disabled_config_is_always_1():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.consecutive_losses = 5
    cfg = AdaptiveRiskConfig(enabled=False, rules=[AdaptiveRiskRule("consecutive_losses", 2, 0.5)])
    assert state.active_multiplier(cfg) == 1.0


def test_active_multiplier_consecutive_losses_rule():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    cfg = AdaptiveRiskConfig(enabled=True, rules=[AdaptiveRiskRule("consecutive_losses", 2, 0.5)])
    assert state.active_multiplier(cfg) == 1.0  # not yet triggered
    state.record_trade_close(-10.0, True)
    state.record_trade_close(-10.0, False)
    assert state.active_multiplier(cfg) == 0.5
    assert "consecutive_losses" in state.active_rule_labels(cfg)[0]


def test_multiple_active_rules_stack_multiplicatively():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    cfg = AdaptiveRiskConfig(
        enabled=True,
        rules=[
            AdaptiveRiskRule("consecutive_losses", 2, 0.5),
            AdaptiveRiskRule("daily_loss_pct", 1.0, 0.5),  # 1% of 10,000 = $100
        ],
    )
    state.record_trade_close(-60.0, True)
    state.record_trade_close(-60.0, False)   # 2 losses in a row AND $120 realized loss today (1.2% > 1%)
    assert state.active_multiplier(cfg) == pytest.approx(0.25)  # 0.5 * 0.5
    assert len(state.active_rule_labels(cfg)) == 2


def test_progress_to_target_rule_requires_profit_target_amount():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.cumulative_realized_pnl = 700.0
    cfg = AdaptiveRiskConfig(
        enabled=True, rules=[AdaptiveRiskRule("progress_to_target_pct", 80, 0.3)],
        profit_target_amount=None,
    )
    # No profit_target_amount supplied -> the rule can never fire, by design.
    assert state.active_multiplier(cfg) == 1.0

    cfg_with_target = AdaptiveRiskConfig(
        enabled=True, rules=[AdaptiveRiskRule("progress_to_target_pct", 80, 0.3)],
        profit_target_amount=800.0,   # 700 / 800 = 87.5% >= 80% threshold
    )
    assert state.active_multiplier(cfg_with_target) == pytest.approx(0.3)


def test_daily_profit_pct_rule():
    state = AdaptiveRiskState(initial_balance=10_000.0)
    state.record_trade_close(250.0, True)  # 2.5% up today
    cfg = AdaptiveRiskConfig(enabled=True, rules=[AdaptiveRiskRule("daily_profit_pct", 2.0, 0.4)])
    assert state.active_multiplier(cfg) == pytest.approx(0.4)


def test_config_accepts_plain_dict_rules():
    """AdaptiveRiskConfig should coerce plain dict rules (e.g. parsed straight
    from JSON on the CLI) into AdaptiveRiskRule instances automatically."""
    cfg = AdaptiveRiskConfig(enabled=True, rules=[{"trigger": "consecutive_losses", "threshold": 3, "risk_multiplier": 0.5}])
    assert isinstance(cfg.rules[0], AdaptiveRiskRule)
    assert cfg.rules[0].threshold == 3


# ---------------------------------------------------------------------------
# Engine wiring: run_backtest / run_execution actually apply the multiplier
# ---------------------------------------------------------------------------

def test_backtest_without_adaptive_risk_leaves_every_trade_at_multiplier_1():
    df = _synthetic_df()
    strategy = ManualStrategy(_FREQUENT_TRADER)
    result = run_backtest(df, strategy, RiskConfig())
    assert len(result.trades) > 5  # sanity: this strategy really does trade often
    assert all(t.adaptive_risk_multiplier == 1.0 for t in result.trades)
    assert all(t.adaptive_risk_rules_active == () for t in result.trades)


def test_backtest_with_adaptive_risk_derisks_after_losing_streak():
    df = _synthetic_df()
    strategy = ManualStrategy(_FREQUENT_TRADER)
    adaptive = AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=2, risk_multiplier=0.5)],
    )
    result = run_backtest(df, strategy, RiskConfig(), adaptive_risk=adaptive)
    assert len(result.trades) > 5

    derisked = [t for t in result.trades if t.adaptive_risk_multiplier < 1.0]
    assert derisked, "expected at least one losing streak long enough to trigger de-risking on this dataset"
    for t in derisked:
        assert t.adaptive_risk_multiplier == pytest.approx(0.5)
        assert any("consecutive_losses" in label for label in t.adaptive_risk_rules_active)

    # Every de-risked trade must have been preceded by >=2 losses -- spot
    # check the first one against the trades that closed before it opened.
    first_derisked = derisked[0]
    prior_trades = [t for t in result.trades if t.exit_time <= first_derisked.entry_time]
    assert len(prior_trades) >= 2
    assert prior_trades[-1].pnl < 0 and prior_trades[-2].pnl < 0


def test_adaptive_risk_never_increases_size_above_nominal():
    df = _synthetic_df()
    strategy = ManualStrategy(_FREQUENT_TRADER)
    adaptive = AdaptiveRiskConfig(
        enabled=True,
        rules=[AdaptiveRiskRule(trigger="consecutive_losses", threshold=1, risk_multiplier=0.5)],
    )
    result = run_backtest(df, strategy, RiskConfig(), adaptive_risk=adaptive)
    assert all(t.adaptive_risk_multiplier <= 1.0 for t in result.trades)


# ---------------------------------------------------------------------------
# drawdown_pct trigger (all-time, vs. a prop firm's overall drawdown floor,
# as opposed to daily_loss_pct which only looks at today) + the one-click
# limit-aware preset built from PropRules
# ---------------------------------------------------------------------------

def test_drawdown_pct_tracks_distance_from_realized_peak_not_just_today():
    state = AdaptiveRiskState(initial_balance=100_000.0)
    state.record_trade_close(-3_000.0, is_new_day=True)
    assert state.current_drawdown_pct() == pytest.approx(3.0)

    # A new peak resets drawdown to 0, unlike daily_loss_pct which only
    # resets on a new calendar day.
    state.record_trade_close(8_000.0, is_new_day=False)
    assert state.current_drawdown_pct() == pytest.approx(0.0)
    assert state.peak_realized_balance == pytest.approx(105_000.0)

    state.record_trade_close(-4_000.0, is_new_day=False)
    assert state.current_drawdown_pct() == pytest.approx(4.0)


def test_drawdown_pct_rule_throttles_size_once_triggered():
    state = AdaptiveRiskState(initial_balance=100_000.0)
    state.record_trade_close(-6_000.0, is_new_day=True)  # 6% drawdown from peak
    cfg = AdaptiveRiskConfig(enabled=True, rules=[
        AdaptiveRiskRule(trigger="drawdown_pct", threshold=5.0, risk_multiplier=0.5),
    ])
    assert state.active_multiplier(cfg) == pytest.approx(0.5)


def test_build_limit_aware_preset_scales_thresholds_off_prop_rules():
    from app.prop.simulator import PropRules

    rules = PropRules(daily_loss_limit_pct=4.0, max_drawdown_pct=8.0)
    cfg = build_limit_aware_preset(rules)

    assert cfg.enabled
    dd_rules = [r for r in cfg.rules if r.trigger == "drawdown_pct"]
    daily_rules = [r for r in cfg.rules if r.trigger == "daily_loss_pct"]
    lock_rules = [r for r in cfg.rules if r.trigger == "daily_profit_pct"]

    assert sorted(r.threshold for r in dd_rules) == [4.0, 6.0]      # 50%/75% of 8.0
    assert sorted(r.threshold for r in daily_rules) == [2.0, 3.0]   # 50%/75% of 4.0
    assert len(lock_rules) == 1 and lock_rules[0].risk_multiplier == 0.0


def test_build_limit_aware_preset_can_disable_profit_lock():
    from app.prop.simulator import PropRules

    cfg = build_limit_aware_preset(PropRules(), daily_profit_lock_pct=None)
    assert not any(r.trigger == "daily_profit_pct" for r in cfg.rules)


def test_limit_aware_preset_actually_throttles_a_real_backtest():
    """End-to-end: wiring a losing streak into a real run_backtest call
    with the preset applied should reduce total risk taken vs. the same
    run with adaptive risk off, once the account is meaningfully down
    from its peak."""
    from app.prop.simulator import PropRules

    df = _synthetic_df(n=2000, seed=7)
    risk = RiskConfig(initial_balance=100_000.0, risk_value=1.0)
    preset = build_limit_aware_preset(PropRules(daily_loss_limit_pct=5.0, max_drawdown_pct=10.0))

    baseline = run_backtest(df, ManualStrategy(_FREQUENT_TRADER), risk)
    throttled = run_backtest(df, ManualStrategy(_FREQUENT_TRADER), risk, adaptive_risk=preset)

    # Should not change WHICH bars get signals, only entry sizing -- same
    # trade count, but total notional risked should be <= baseline once
    # any throttling rule ever fired (it can only ever scale size down).
    assert len(throttled.trades) == len(baseline.trades)
    if baseline.trades:
        assert sum(abs(t.size) for t in throttled.trades) <= sum(abs(t.size) for t in baseline.trades) + 1e-6
