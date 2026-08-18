import pandas as pd
import pytest

from app.backtest.execution import Trade
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules


def _mock_trades(n=60, seed_pnls=None):
    trades = []
    base = pd.Timestamp("2024-01-01")
    pnls = seed_pnls or ([150, -80] * (n // 2))
    for i, pnl in enumerate(pnls[:n]):
        t = base + pd.Timedelta(days=i // 3)
        trades.append(Trade(
            entry_time=t, exit_time=t, direction=1, entry_price=1.1, exit_price=1.1,
            size=1000, pnl=pnl, pnl_pct=0.1, exit_reason="signal", commission=0, equity_after=0,
        ))
    return trades


def test_monte_carlo_runs_and_produces_probabilities():
    trades = _mock_trades(90)
    rules = PropRules(account_size=10000, evaluation_profit_target_pct=5, daily_loss_limit_pct=50,
                       max_drawdown_pct=50, min_trading_days=1, consistency_rule_pct=None)
    cfg = MonteCarloConfig(n_simulations=200, method="bootstrap", random_seed=1)
    result = run_monte_carlo(trades, rules, cfg)
    assert 0 <= result.evaluation_pass_probability <= 100
    assert 0 <= result.first_payout_probability <= 100
    assert result.n_simulations == 200


def test_monte_carlo_shuffle_method():
    trades = _mock_trades(60)
    rules = PropRules()
    cfg = MonteCarloConfig(n_simulations=100, method="shuffle", random_seed=2)
    result = run_monte_carlo(trades, rules, cfg)
    assert result.n_simulations == 100


def test_monte_carlo_empty_trades_raises():
    with pytest.raises(ValueError):
        run_monte_carlo([], PropRules(), MonteCarloConfig(n_simulations=10))
