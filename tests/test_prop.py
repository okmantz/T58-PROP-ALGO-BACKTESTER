import pandas as pd

from app.prop.simulator import PropRules, simulate_account


def test_evaluation_passes_on_sufficient_profit():
    rules = PropRules(
        account_size=10000,
        evaluation_profit_target_pct=5,
        daily_loss_limit_pct=100,
        max_drawdown_pct=100,
        consistency_rule_pct=None,
        min_trading_days=2,
        payout_frequency_days=0,
        payout_threshold_pct=0,
    )
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [300, 300, 0]  # 600 / 10000 = 6% > 5% target, over 3 trading days
    result = simulate_account(pnls, dates, rules)
    assert result.passed_evaluation
    assert result.days_to_pass is not None


def test_daily_loss_limit_triggers_failure():
    rules = PropRules(account_size=10000, daily_loss_limit_pct=5, max_drawdown_pct=100)
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")]
    pnls = [-300, -300]  # -600 in one day = -6% > 5% limit
    result = simulate_account(pnls, dates, rules)
    assert result.failed
    assert result.failure_reason == "daily_loss_limit"


def test_max_drawdown_triggers_failure():
    rules = PropRules(account_size=10000, daily_loss_limit_pct=100, max_drawdown_pct=5, drawdown_type="static")
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [-200, -200, -200]  # cumulative -600 = -6% > 5% max dd, spread across days to avoid daily limit
    result = simulate_account(pnls, dates, rules)
    assert result.failed
    assert "max_drawdown" in result.failure_reason


def test_funded_account_reaches_payout():
    rules = PropRules(
        account_size=10000,
        evaluation_profit_target_pct=5,
        daily_loss_limit_pct=100,
        max_drawdown_pct=100,
        consistency_rule_pct=None,
        min_trading_days=1,
        payout_frequency_days=1,
        payout_threshold_pct=1,
        required_buffer_pct=0,
    )
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    pnls = [600, 200, 200]  # passes eval on day 1, then accrues funded profit for payout
    result = simulate_account(pnls, dates, rules)
    assert result.passed_evaluation
    assert result.reached_first_payout


def test_no_trades_returns_safe_default():
    result = simulate_account([], [], PropRules())
    assert not result.passed_evaluation
    assert not result.failed
