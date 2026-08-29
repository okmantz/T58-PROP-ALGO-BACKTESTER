import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.manual import ManualStrategy


def _sma_strategy():
    return ManualStrategy({
        "name": "sma cross",
        "indicators": [
            {"type": "sma", "period": 5, "column": "close", "as": "sma_fast"},
            {"type": "sma", "period": 15, "column": "close", "as": "sma_slow"},
        ],
        "long_entry": "sma_fast > sma_slow",
        "long_exit": "sma_fast < sma_slow",
        "short_entry": "sma_fast < sma_slow",
        "short_exit": "sma_fast > sma_slow",
        "stop_loss_pips": 15,
        "take_profit_pips": 30,
    })


def _gold_scale_df(n=300, seed=3):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1900.0
    rows = []
    for i in range(n):
        drift = rng.normal(0, 0.6)
        o = price
        c = o + drift
        h = max(o, c) + abs(rng.normal(0, 0.4))
        l = min(o, c) - abs(rng.normal(0, 0.4))
        rows.append((ts[i], o, h, l, c, 100.0))
        price = c
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def _flat_fx_df(n=200, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    price = 1.10 + np.cumsum(rng.normal(0, 0.00005, n))
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price + 0.0003,
        "low": price - 0.0003, "close": price, "volume": 100.0,
    })


def _generate(df, risk, tmp_path):
    strategy = _sma_strategy()
    bt = run_backtest(df, strategy, risk)
    rules = PropRules()
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(pnls, dates, rules)
    mc = run_monte_carlo(bt.trades, rules, MonteCarloConfig(n_simulations=100, random_seed=1))
    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    paths = generate_full_report(
        output_dir=tmp_path, strategy_name="Test", strategy_source_type="manual",
        instrument="TEST.csv", timeframe="5m", backtest_period=period,
        backtest_result=bt, prop_rules=rules, prop_single_run=single_run,
        monte_carlo_result=mc, basename="test_report", risk_config=risk,
    )
    return paths["html"].read_text(encoding="utf-8")


def test_pip_size_mismatch_warning_appears_in_saved_html_report(tmp_path):
    """A warning that only ever reached a live run console was silently lost
    the moment someone read the saved report later -- it must be baked into
    the HTML file itself."""
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="percent", risk_value=2.0,
        pip_size=0.0001,  # wrong for ~1900-scale prices
        spread_pips=1.0, slippage_pips=0.5, commission_per_trade=5.0,
    )
    html = _generate(_gold_scale_df(), risk, tmp_path)
    assert '<div class="warning-banner">' in html
    assert "pip_size" in html


def test_clean_run_has_no_warning_banner(tmp_path):
    """A run with no execution-integrity issues must not show the banner at
    all -- it should be invisible by default, not an empty box."""
    risk = RiskConfig(
        initial_balance=50_000.0, risk_mode="percent", risk_value=2.0,
        pip_size=0.0001, spread_pips=1.0, slippage_pips=0.5, commission_per_trade=5.0,
    )
    html = _generate(_flat_fx_df(), risk, tmp_path)
    assert '<div class="warning-banner">' not in html


def test_verdict_and_final_parameters_appear_in_report(tmp_path):
    """Full Pipeline's verdict/reasons and winning parameter values must be
    baked into the saved HTML report itself, not just returned to the
    caller and logged to a live console that may not be open later."""
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=2.0, pip_size=0.0001)
    df = _flat_fx_df()
    strategy = _sma_strategy()
    bt = run_backtest(df, strategy, risk)
    rules = PropRules()
    pnls = [t.pnl for t in bt.trades]
    dates = [t.entry_time for t in bt.trades]
    single_run = simulate_account(pnls, dates, rules)
    mc = run_monte_carlo(bt.trades, rules, MonteCarloConfig(n_simulations=100, random_seed=1))
    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    paths = generate_full_report(
        output_dir=tmp_path, strategy_name="Test", strategy_source_type="manual",
        instrument="TEST.csv", timeframe="5m", backtest_period=period,
        backtest_result=bt, prop_rules=rules, prop_single_run=single_run,
        monte_carlo_result=mc, basename="test_report", risk_config=risk,
        verdict="NOT READY", verdict_reasons=["Did not clear the eval pass threshold."],
        final_parameters={"stop_loss_pips": "22", "take_profit_pips": "48"},
    )
    html = paths["html"].read_text(encoding="utf-8")
    assert '<div class="verdict-banner verdict-not-ready">' in html
    assert "Did not clear the eval pass threshold." in html
    assert "Final Parameters" in html
    assert "stop_loss_pips" in html and "22" in html


def test_no_verdict_section_when_not_provided(tmp_path):
    """Every non-Full-Pipeline report (the overwhelming majority of
    reports the app generates) must render exactly as before -- no empty
    verdict banner appearing out of nowhere."""
    risk = RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=2.0, pip_size=0.0001)
    html = _generate(_flat_fx_df(), risk, tmp_path)
    assert '<div class="verdict-banner' not in html
    assert "Final Parameters" not in html
