"""
Risk Sweep -- "Risk Is a Strategy Variable" (T58 Quant Trading Masterclass
PDF, Lesson 7).

The Masterclass material's own point: don't assume 1% risk-per-trade is
optimal, and don't optimize for the return of any ONE trade -- optimize
P(reach payout target before hitting a loss constraint), which a smaller
risk level can sometimes dramatically improve at the cost of taking
longer. That's a question about the FULL evaluation-to-payout path, not
about a single backtest's net profit, so this module re-runs the
strategy at each candidate risk level (position sizing changes every
trade's dollar P&L, so this can't be approximated by rescaling one
backtest's numbers after the fact) and scores each level with
app.prop.survival_engine.run_prop_survival_analysis -- the same
full-lifecycle survival model app.ai.research_loop and the Search Lab
already trust, not a new metric invented here.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.prop.simulator import PropRules
from app.prop.survival_engine import PropSurvivalConfig, run_prop_survival_analysis

DEFAULT_RISK_VALUES: list[float] = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]


@dataclass
class RiskSweepPoint:
    risk_value: float                 # % of equity risked per trade (RiskConfig.risk_value, risk_mode="percent")
    n_trades: int
    net_profit: float
    max_drawdown_pct: float
    prop_survival_score: float        # 0-100, from app.prop.survival_engine
    probability_pass_evaluation: float
    probability_first_payout: float
    median_days_to_first_payout: float | None
    never_recovered_pct: float        # % of simulated lives whose worst drawdown was never reclaimed

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RiskSweepResult:
    points: list                # list[RiskSweepPoint], one per risk_value tested, in the order tested
    best_point: "RiskSweepPoint | None"   # highest prop_survival_score; None if every level failed to trade
    metric: str                 # what "best" was chosen by -- always "prop_survival_score" today
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "best_point": self.best_point.to_dict() if self.best_point else None,
            "metric": self.metric, "notes": self.notes,
        }

    def render_table(self) -> str:
        lines = ["Risk Sweep (risk % per trade vs. prop-survival outcomes)", ""]
        header = f"{'Risk %':>8}{'Trades':>9}{'Net Profit':>13}{'Max DD':>9}{'T58 Score':>11}{'Pass %':>9}{'Payout %':>10}"
        lines.append(header)
        lines.append("-" * len(header))
        for p in self.points:
            best_marker = "  <-- best" if self.best_point is p else ""
            lines.append(
                f"{p.risk_value:>7.2f}%{p.n_trades:>9}{p.net_profit:>13,.2f}{p.max_drawdown_pct:>8.1f}%"
                f"{p.prop_survival_score:>10.1f}{p.probability_pass_evaluation:>8.1f}%"
                f"{p.probability_first_payout:>9.1f}%{best_marker}"
            )
        if self.best_point:
            lines.append("")
            lines.append(
                f"Best risk level for this strategy/data: {self.best_point.risk_value:.2f}% per trade "
                f"(T58 Prop Survival Score {self.best_point.prop_survival_score:.1f}/100)."
            )
        return "\n".join(lines)


def run_risk_sweep(
    df,
    strategy_builder,
    base_risk: RiskConfig,
    prop_rules: PropRules,
    risk_values: list[float] | None = None,
    survival_cfg: PropSurvivalConfig | None = None,
) -> RiskSweepResult:
    """
    strategy_builder: zero-arg callable returning a FRESH Strategy
        instance per call -- same "always build fresh" convention as
        app.search.robustness.run_walk_forward, since some strategy
        sources cache internal state keyed to the data they last saw.
    base_risk: template RiskConfig -- every field is reused UNCHANGED
        except risk_value, which is overwritten with each candidate from
        `risk_values`. base_risk.risk_mode should be "percent" (the
        Masterclass material's own framing, and the only mode where
        sweeping a list of risk LEVELS is meaningful); a "fixed" $-mode
        base_risk still works but risk_values are then interpreted as
        flat dollars, not percentages, since risk_value's meaning
        depends on risk_mode -- see RiskConfig.risk_amount.
    risk_values: candidate risk levels to test, in the SAME units as
        base_risk.risk_value. Defaults to the exact list the Masterclass
        material itself suggests (0.10% through 1.00%).
    """
    risk_values = DEFAULT_RISK_VALUES if risk_values is None else risk_values
    if not risk_values:
        raise ValueError("risk_values must contain at least one candidate risk level.")

    points: list[RiskSweepPoint] = []
    notes: list[str] = []

    for value in risk_values:
        risk = dataclasses.replace(base_risk, risk_value=float(value))
        try:
            bt = run_backtest(df, strategy_builder(), risk)
        except Exception as exc:  # noqa: BLE001 -- one bad level must not kill the whole sweep
            notes.append(f"Risk {value:.2f}%: backtest failed ({exc}) -- skipped.")
            continue
        if not bt.trades:
            notes.append(f"Risk {value:.2f}%: produced zero trades -- skipped.")
            continue

        try:
            survival = run_prop_survival_analysis(bt.trades, prop_rules, survival_cfg)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Risk {value:.2f}%: survival analysis failed ({exc}) -- skipped.")
            continue

        points.append(RiskSweepPoint(
            risk_value=float(value), n_trades=len(bt.trades), net_profit=bt.statistics.net_profit,
            max_drawdown_pct=bt.statistics.max_drawdown_pct, prop_survival_score=survival.prop_survival_score,
            probability_pass_evaluation=survival.evaluation.probability_pass_evaluation,
            probability_first_payout=survival.funded.probability_first_payout,
            median_days_to_first_payout=survival.funded.median_days_to_first_payout,
            never_recovered_pct=survival.evaluation.never_recovered_pct,
        ))

    if not points:
        notes.append("No risk level produced a scoreable result -- every level was skipped.")
        return RiskSweepResult(points=[], best_point=None, metric="prop_survival_score", notes=notes)

    best_point = max(points, key=lambda p: p.prop_survival_score)
    return RiskSweepResult(points=points, best_point=best_point, metric="prop_survival_score", notes=notes)
