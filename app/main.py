"""
T58 Trading — Prop Algo Backtester
Entry point.

Usage:
    python -m app.main                 # launch desktop GUI
    python -m app.main --cli --csv data/examples/EURUSD_5M_sample.csv
                                        # run the full pipeline headlessly
                                        # (useful on machines without a display,
                                        # and for scripted/CI runs)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.data.importer import import_csv
from app.data.storage import list_stored_datasets, store_csv_path
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.manual import ManualStrategy
from app.ui.main_window import launch

DEFAULT_MANUAL_STRATEGY = {
    "name": "SMA 20/50 Cross",
    "indicators": [
        {"type": "sma", "period": 20, "column": "close", "as": "sma_fast"},
        {"type": "sma", "period": 50, "column": "close", "as": "sma_slow"},
    ],
    "long_entry": "sma_fast > sma_slow",
    "long_exit": "sma_fast < sma_slow",
    "short_entry": "sma_fast < sma_slow",
    "short_exit": "sma_fast > sma_slow",
    "stop_loss_pips": 20,
    "take_profit_pips": 40,
}


def _resolve_default_csv() -> str:
    """Prefer a dataset already stored in data/raw/ over the bundled example."""
    stored = list_stored_datasets()
    if stored:
        return str(stored[0].path)  # most recently added
    return "data/examples/EURUSD_5M_sample.csv"


def run_cli(csv_path: str | None, n_sims: int, output_dir: str) -> None:
    if csv_path is None:
        csv_path = _resolve_default_csv()
    else:
        stored_path = store_csv_path(csv_path)
        csv_path = str(stored_path)

    import_result = import_csv(csv_path)
    if not import_result.is_valid:
        print("Import failed:")
        for e in import_result.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    df = import_result.dataframe
    for w in import_result.warnings:
        print(f"  WARNING: {w}")
    print(f"Loaded {len(df)} bars from {csv_path}")

    strategy = ManualStrategy(DEFAULT_MANUAL_STRATEGY)
    risk = RiskConfig()
    rules = PropRules()

    print("Running historical backtest...")
    bt_result = run_backtest(df, strategy, risk)
    print(f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}  "
          f"Win rate: {bt_result.statistics.win_rate:.1f}%")

    print("Running prop-firm simulation on historical sequence...")
    trade_pnls = [t.pnl for t in bt_result.trades]
    trade_dates = [t.entry_time for t in bt_result.trades]
    single_run = simulate_account(trade_pnls, trade_dates, rules)
    print(f"  Passed evaluation: {single_run.passed_evaluation}  Reached payout: {single_run.reached_first_payout}")

    print(f"Running Monte Carlo simulation ({n_sims:,} runs)...")
    mc_result = run_monte_carlo(bt_result.trades, rules, MonteCarloConfig(n_simulations=n_sims))
    print(f"  Evaluation pass probability: {mc_result.evaluation_pass_probability:.1f}%")
    print(f"  First payout probability: {mc_result.first_payout_probability:.1f}%")
    print(f"  Expected payout: ${mc_result.expected_payout:,.2f}")

    print("Running out-of-sample holdout check...")
    try:
        holdout_comparison = run_holdout_comparison(df, strategy, risk, holdout_frac=0.2)
    except Exception as exc:
        print(f"  Holdout check skipped: {exc}")
        holdout_comparison = None

    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    paths = generate_full_report(
        output_dir=output_dir,
        strategy_name=bt_result.strategy_name,
        strategy_source_type=strategy.source_type,
        instrument=Path(csv_path).name,
        timeframe="unknown",
        backtest_period=period,
        backtest_result=bt_result,
        prop_rules=rules,
        prop_single_run=single_run,
        monte_carlo_result=mc_result,
        holdout_comparison=holdout_comparison,
    )
    print("\nReport written to:")
    for k, p in paths.items():
        print(f"  {k}: {p}")


def main():
    parser = argparse.ArgumentParser(description="T58 Trading — Prop Algo Backtester")
    parser.add_argument("--cli", action="store_true", help="run headlessly instead of launching the GUI")
    parser.add_argument("--csv", default=None, help="path to a market data CSV (--cli mode); if omitted, uses the "
                                                       "most recently stored dataset in data/raw/, or the bundled sample")
    parser.add_argument("--sims", type=int, default=10000, help="number of Monte Carlo simulations (--cli mode)")
    parser.add_argument("--output", default="reports", help="output directory for the report (--cli mode)")
    args = parser.parse_args()

    if args.cli:
        run_cli(args.csv, args.sims, args.output)
    else:
        launch()


if __name__ == "__main__":
    main()
