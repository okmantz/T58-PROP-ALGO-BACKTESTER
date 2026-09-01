"""
Tests for the account-survivability hard caps added to RiskConfig /
run_execution: RiskConfig.max_trade_loss (per-trade ceiling) and
RiskConfig.max_account_drawdown_pct (account-blown circuit breaker), plus
with_prop_safety_defaults wiring PropRules.max_drawdown_pct into a
RiskConfig automatically.

These exist because a pip_size/instrument-scale mismatch (a strategy's
fixed-pips stop calibrated for FX but tested against a stock/index/gold
instrument) combined with an honest gap-through fill could previously
report a single simulated trade losing far more than the entire account
-- real observed cases from a batch Full Pipeline run: a single-run net
loss of -$50,000 and -$2,534,176 on nominal $50,000 accounts. That cannot
happen to a real funded account (negative-balance protection, and the
firm terminates an account once it breaches the drawdown floor rather
than letting it keep trading into deeper negative equity), so the engine
must not be able to report it either.
"""
import pandas as pd
import pytest

from app.backtest.execution import run_execution
from app.backtest.risk import RiskConfig, with_prop_safety_defaults
from app.prop.simulator import PropRules


def test_max_trade_loss_caps_a_catastrophic_gap_through_loss():
    """A pip_size/instrument mismatch (FX pip_size against a ~150-priced
    instrument) sizes the position off a nearly-zero stop distance, so an
    ordinary-looking overnight gap becomes a massive dollar loss. The
    engine must cap that single trade's loss instead of reporting it in
    full."""
    ts = pd.date_range("2024-01-01 09:00", periods=2, freq="1d")
    rows = [
        (ts[0], 150.0, 150.2, 149.8, 150.0, 1000.0),
        (ts[1], 130.0, 130.5, 125.0, 128.0, 1000.0),  # huge overnight gap down
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 1])

    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=1.0,
        pip_size=0.0001,  # FX-scale pip_size left at default against a stock-scale price
    )
    trades, _equity_df = run_execution(
        df, signals, risk, stop_loss_pips=15, take_profit_pips=None,
    )
    gap_trade = next(t for t in trades if t.exit_reason == "stop_loss")
    intended_risk = risk.risk_amount(risk.initial_balance)  # $100
    cap = risk.max_trade_loss(risk.initial_balance)  # 3x intended risk = $300
    assert cap == pytest.approx(intended_risk * 3.0)
    # Without the cap this trade would lose well over $1M (huge size x
    # $20 gap). With it, the loss must never exceed the configured
    # ceiling.
    assert -gap_trade.pnl <= cap + 1e-6
    assert -gap_trade.pnl == pytest.approx(cap, rel=1e-6)


def test_winning_trades_are_never_clamped():
    """The loss cap must only ever touch losing trades -- a winner must
    come through completely unaffected by max_trade_loss."""
    ts = pd.date_range("2024-01-01 09:00", periods=2, freq="1d")
    rows = [
        (ts[0], 100.0, 100.2, 99.8, 100.0, 1000.0),
        (ts[1], 100.0, 106.0, 99.9, 105.0, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0])

    risk = RiskConfig(initial_balance=10_000.0, risk_mode="percent", risk_value=1.0, pip_size=1.0)
    trades, _equity_df = run_execution(df, signals, risk, stop_loss_pips=2, take_profit_pips=None)
    assert len(trades) == 1
    assert trades[0].pnl > 0


def test_account_blown_halts_new_trades():
    """Once equity crosses the configured max_account_drawdown_pct floor,
    no further NEW trades may open -- mirrors a real prop account being
    terminated rather than continuing to trade into deeper negative
    equity."""
    ts = pd.date_range("2024-01-01 09:00", periods=6, freq="1d")
    rows = [
        (ts[0], 100.0, 100.2, 99.8, 100.0, 1000.0),
        (ts[1], 100.0, 100.2, 80.0, 82.0, 1000.0),   # blows the account
        (ts[2], 82.0, 90.0, 81.0, 88.0, 1000.0),     # would-be new signal, must be blocked
        (ts[3], 88.0, 92.0, 87.0, 91.0, 1000.0),
        (ts[4], 91.0, 95.0, 90.0, 94.0, 1000.0),
        (ts[5], 94.0, 98.0, 93.0, 97.0, 1000.0),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    signals = pd.Series([1, 0, 1, 1, 1, 1])  # flat then tries to re-enter long repeatedly

    risk = RiskConfig(
        initial_balance=10_000.0, risk_mode="percent", risk_value=50.0,  # deliberately reckless sizing
        pip_size=1.0, max_account_drawdown_pct=10.0,  # blown once equity <= $9,000
    )
    with pytest.warns(RuntimeWarning, match="Account BLOWN"):
        trades, equity_df = run_execution(df, signals, risk, stop_loss_pips=25, take_profit_pips=None)

    # Only the first trade (the one that blew the account) should exist --
    # every later signal must have been blocked by the circuit breaker.
    assert len(trades) == 1
    assert equity_df["equity"].iloc[-1] == pytest.approx(equity_df["equity"].iloc[1], abs=1e-6) \
        or equity_df["equity"].iloc[-1] <= risk.initial_balance


def test_with_prop_safety_defaults_wires_max_drawdown_from_prop_rules():
    risk = RiskConfig(initial_balance=50_000.0)
    rules = PropRules(account_size=50_000.0, max_drawdown_pct=10.0)
    safe_risk = with_prop_safety_defaults(risk, rules)
    assert safe_risk.max_account_drawdown_pct == 10.0
    # Original RiskConfig instance must be untouched (returns a copy).
    assert risk.max_account_drawdown_pct is None


def test_with_prop_safety_defaults_never_overrides_explicit_value():
    risk = RiskConfig(initial_balance=50_000.0, max_account_drawdown_pct=4.0)
    rules = PropRules(account_size=50_000.0, max_drawdown_pct=10.0)
    safe_risk = with_prop_safety_defaults(risk, rules)
    assert safe_risk.max_account_drawdown_pct == 4.0
