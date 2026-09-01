"""
Multi-strategy batch testing.

The Strategy Library lets Python / PineScript / MQL5 strategies pile up
over time. This module runs the exact same single-strategy pipeline every
other path in this app already trusts -- backtest -> prop-firm simulation
-> Monte Carlo -> generate_full_report() -- against every strategy in a
supplied list, one after another, producing ONE saved report per strategy
(never a merged/averaged one), and records each result back onto that
strategy's own Strategy Library metadata (record_backtest_result) exactly
the way a single Run & Report run already does. Because every report
funnels through generate_full_report(), every result is automatically
picked up by the Dashboard/run_history the same way Search Lab's Bulk
Backtest mode already is -- no separate wiring needed here.

Deliberately a thin, serial reuse of existing, already-validated pipeline
code -- the only new behavior is looping it over more than one strategy,
with one bad strategy (zero trades, a parse error, whatever) never
stopping the rest of the batch.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.backtest.engine import run_backtest, run_holdout_comparison
from app.backtest.risk import RiskConfig
from app.monte_carlo.engine import MonteCarloConfig, run_monte_carlo
from app.prop.simulator import PropRules, simulate_account
from app.reports.generator import generate_full_report
from app.strategy.base import Strategy

ProgressCallback = Callable[[str], None]


@dataclass
class BatchTestItem:
    label: str                                     # display name (e.g. the library filename)
    strategy: Strategy
    library_ref: tuple[str, str] | None = None      # (strategy_type, filename) -- set for library-sourced items only


@dataclass
class BatchTestOutcome:
    label: str
    ok: bool
    reason: str | None = None                       # set when ok is False
    trades: int = 0
    net_profit: float = 0.0
    eval_pass_probability: float = 0.0
    report_html: Path | None = None
    report_json: Path | None = None


@dataclass
class BatchTestSummary:
    outcomes: list = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> list:
        return [o for o in self.outcomes if o.ok]

    @property
    def failed(self) -> list:
        return [o for o in self.outcomes if not o.ok]


def run_batch_test(
    df: pd.DataFrame,
    items: list[BatchTestItem],
    risk: RiskConfig,
    prop_rules: PropRules,
    output_dir: str | Path,
    instrument: str = "unknown",
    mc_sims: int = 10_000,
    mc_method: str = "bootstrap",
    holdout_frac: float = 0.2,
    basename_prefix: str = "batch",
    progress_cb: ProgressCallback | None = None,
) -> BatchTestSummary:
    """Runs each item's strategy through the standard single-strategy
    pipeline and writes one report per item. A single bad strategy
    (backtest error, zero trades) is recorded as a failed outcome and the
    rest of the batch continues -- it never aborts the whole run."""
    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    t0 = time.time()
    period = (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    outcomes: list[BatchTestOutcome] = []

    for i, item in enumerate(items, start=1):
        log(f"[{i}/{len(items)}] {item.label}")
        try:
            bt_result = run_backtest(df, item.strategy, risk)
        except Exception as exc:  # noqa: BLE001 -- one bad strategy must not stop the batch
            log(f"  Skipped -- backtest error: {exc}")
            outcomes.append(BatchTestOutcome(item.label, ok=False, reason=f"backtest error: {exc}"))
            continue

        if not bt_result.trades:
            log("  Skipped -- 0 trades generated on this data.")
            outcomes.append(BatchTestOutcome(item.label, ok=False, reason="0 trades generated on this data"))
            continue

        trade_pnls = [t.pnl for t in bt_result.trades]
        trade_dates = [t.entry_time for t in bt_result.trades]
        single_run = simulate_account(trade_pnls, trade_dates, prop_rules)
        mc_cfg = MonteCarloConfig(n_simulations=mc_sims, method=mc_method)
        mc_result = run_monte_carlo(bt_result.trades, prop_rules, mc_cfg)

        try:
            holdout_comparison = run_holdout_comparison(df, item.strategy, risk, holdout_frac=holdout_frac)
        except Exception:  # noqa: BLE001 -- a holdout that can't run isn't a reason to fail the item
            holdout_comparison = None

        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", item.label) or f"strategy_{i}"
        paths = generate_full_report(
            output_dir=output_dir,
            basename=f"{basename_prefix}_{i:02d}_{safe_name}",
            strategy_name=item.label,
            strategy_source_type=item.strategy.source_type,
            instrument=instrument,
            timeframe="unknown",
            backtest_period=period,
            backtest_result=bt_result,
            prop_rules=prop_rules,
            prop_single_run=single_run,
            monte_carlo_result=mc_result,
            holdout_comparison=holdout_comparison,
            risk_config=risk,
            price_df=df,
        )

        if item.library_ref is not None:
            try:
                from app.strategy.library import record_backtest_result
                record_backtest_result(item.library_ref[0], item.library_ref[1], {
                    "trades": len(bt_result.trades),
                    "net_profit": round(bt_result.statistics.net_profit, 2),
                    "win_rate": round(bt_result.statistics.win_rate, 1),
                    "max_dd": round(bt_result.statistics.max_drawdown_pct, 2),
                    "eval_pass_probability": round(mc_result.evaluation_pass_probability, 1),
                    "report_html": str(paths["html"]),
                })
            except Exception:  # noqa: BLE001 -- recording to the library is a convenience, not core output
                pass

        try:
            from app.ai.experiment_memory import record_experiment

            record_experiment(
                origin="batch_test",
                strategy_name=item.label,
                source_type=item.strategy.source_type,
                instrument=instrument,
                verdict="TESTED",
                trades=len(bt_result.trades),
                net_profit=bt_result.statistics.net_profit,
                win_rate=bt_result.statistics.win_rate,
                profit_factor=bt_result.statistics.profit_factor,
                max_drawdown_pct=bt_result.statistics.max_drawdown_pct,
                eval_pass_probability=mc_result.evaluation_pass_probability,
                first_payout_probability=mc_result.first_payout_probability,
                risk_of_ruin_pct=mc_result.risk_of_ruin_pct,
            )
        except Exception:  # noqa: BLE001 -- T58 Research Memory is a bonus record, not core output
            pass

        log(
            f"  Trades: {len(bt_result.trades)}  Net profit: ${bt_result.statistics.net_profit:,.2f}  "
            f"Eval pass probability: {mc_result.evaluation_pass_probability:.1f}%  Report: {paths['html'].name}"
        )
        outcomes.append(BatchTestOutcome(
            item.label, ok=True, trades=len(bt_result.trades),
            net_profit=bt_result.statistics.net_profit,
            eval_pass_probability=mc_result.evaluation_pass_probability,
            report_html=paths["html"], report_json=paths.get("json"),
        ))

    elapsed = time.time() - t0
    log(
        f"\nBatch test complete in {elapsed:.1f}s. {len(items)} strategy(ies) attempted, "
        f"{sum(1 for o in outcomes if o.ok)} produced a report."
    )
    return BatchTestSummary(outcomes=outcomes, elapsed_seconds=elapsed)
