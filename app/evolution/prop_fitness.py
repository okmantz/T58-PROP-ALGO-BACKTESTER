"""
PROP FITNESS -- the composite score the Evolution Lab (app.evolution.engine)
selects and ranks candidates by, instead of raw profit.

    PROP FITNESS =
        pass_probability x payout_probability x robustness x oos_consistency
        / drawdown
        - penalties(too_few_trades, high_param_sensitivity, high_pbo,
                     is_oos_degradation, concentration, losing_streaks)

Every input is a signal this codebase already computes elsewhere (Monte
Carlo, app.search.robustness's parameter-neighborhood robustness and
walk-forward efficiency, app.validation.cpcv's genuine PBO/CPCV
degradation, app.backtest.statistics' own drawdown/streak/trade-count
numbers) -- this module's only job is combining them into one ranking
number, consistently, in one place, so the Evolution Lab's PRE-FILTER,
CLUSTER, and KEEP TOP N steps are all comparing candidates on the same
scale.

This is deliberately NOT the same thing as
app.optimize.refinement.compute_fitness's "composite_prop_score" metric,
which is cheap (Monte Carlo only) and used INSIDE the GA's inner loop
where it has to run hundreds of times per generation. PROP FITNESS is
the expensive, only-computed-once-per-surviving-candidate score used for
final ranking after robustness/OOS/CPCV/stress have already run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropFitnessBreakdown:
    """Every component that went into the final score, so the UI/journal
    can show *why* a candidate ranked where it did, not just the number."""
    pass_probability: float           # 0..1, Monte Carlo evaluation-pass probability
    payout_probability: float         # 0..1, Monte Carlo first-payout probability
    robustness: float                 # 0..1, parameter-neighborhood stability ratio
    oos_consistency: float            # 0..1, walk-forward efficiency (clipped)
    drawdown_pct: float                # raw max drawdown %, used as the divisor
    base_score: float                  # the multiplicative term before penalties

    penalty_too_few_trades: float = 0.0
    penalty_param_sensitivity: float = 0.0
    penalty_high_pbo: float = 0.0
    penalty_is_oos_degradation: float = 0.0
    penalty_concentration: float = 0.0
    penalty_losing_streak: float = 0.0

    final_score: float = 0.0
    notes: list = field(default_factory=list)

    def total_penalty(self) -> float:
        return (
            self.penalty_too_few_trades + self.penalty_param_sensitivity
            + self.penalty_high_pbo + self.penalty_is_oos_degradation
            + self.penalty_concentration + self.penalty_losing_streak
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _trade_concentration(trade_pnls: list[float]) -> float:
    """Fraction of total GROSS PROFIT contributed by the single best trade.
    High = a small number of lucky trades are propping up the whole result
    (the strategy hasn't actually demonstrated a repeatable edge yet)."""
    wins = [p for p in trade_pnls if p > 0]
    gross_profit = sum(wins)
    if gross_profit <= 0 or not wins:
        return 0.0
    return max(wins) / gross_profit


def compute_prop_fitness(
    stats: dict,
    mc_summary: dict,
    robustness_dict: dict | None,
    walk_forward_dict: dict | None,
    trade_pnls: list[float],
    pbo: float | None = None,               # 0..1, from app.validation.cpcv.compute_pbo, pooled per-generation
    cpcv_degradation: float | None = None,  # in-sample minus out-of-sample metric, from app.validation.cpcv.run_cpcv
    min_trades_target: int = 30,
) -> PropFitnessBreakdown:
    pass_probability = max(0.0, min(1.0, mc_summary.get("evaluation_pass_probability", 0.0) / 100.0))
    payout_probability = max(0.0, min(1.0, mc_summary.get("first_payout_probability", 0.0) / 100.0))
    robustness = float(robustness_dict["stability_ratio"]) if robustness_dict else 0.5
    robustness = max(0.0, min(1.0, robustness))
    # walk-forward efficiency is test/train metric ratio -- can run above 1
    # (test outperformed train) or deeply negative; clip to [0, 1] for use
    # as a multiplicative consistency factor (this function only wants
    # "how much of in-sample performance survived," not the raw ratio).
    oos_consistency = 0.5
    if walk_forward_dict is not None:
        oos_consistency = max(0.0, min(1.0, float(walk_forward_dict.get("walk_forward_efficiency", 0.5))))
    drawdown_pct = max(float(stats.get("max_drawdown_pct", 0.0) or 0.0), 1.0)  # floor at 1 to avoid div-by-~0 blowups

    base_score = (pass_probability * payout_probability * robustness * oos_consistency) / drawdown_pct * 100.0

    notes: list[str] = []
    n_trades = int(stats.get("total_trades", 0) or 0)

    penalty_too_few_trades = 0.0
    if n_trades < min_trades_target:
        shortfall = (min_trades_target - n_trades) / min_trades_target
        penalty_too_few_trades = shortfall * 15.0
        notes.append(f"Only {n_trades} trades (target {min_trades_target}+) -- fitness discounted for small sample size.")

    penalty_param_sensitivity = (1.0 - robustness) * 10.0
    if robustness_dict is not None and not robustness_dict.get("is_stable", True):
        notes.append(
            f"Parameter-neighborhood stability {robustness:.2f} -- nearby parameter values behave "
            "quite differently, a sign of fitting to noise rather than a real edge."
        )

    penalty_high_pbo = 0.0
    if pbo is not None:
        pbo = max(0.0, min(1.0, pbo))
        if pbo > 0.5:
            penalty_high_pbo = (pbo - 0.5) * 40.0
            notes.append(f"Probability of Backtest Overfitting {pbo:.0%} (>50%) -- likely overfit vs. its peer pool.")

    penalty_is_oos_degradation = 0.0
    if cpcv_degradation is not None and cpcv_degradation > 0:
        penalty_is_oos_degradation = min(cpcv_degradation, 5.0) * 4.0
        notes.append(f"CPCV in-sample vs out-of-sample degradation {cpcv_degradation:.2f}.")

    concentration = _trade_concentration(trade_pnls)
    penalty_concentration = 0.0
    if concentration > 0.2:
        penalty_concentration = (concentration - 0.2) * 30.0
        notes.append(f"Single best trade is {concentration:.0%} of gross profit -- result may hinge on one outlier.")

    penalty_losing_streak = 0.0
    max_losing_streak = int(stats.get("max_losing_streak", 0) or 0)
    if n_trades > 0:
        streak_frac = max_losing_streak / n_trades
        if streak_frac > 0.15:
            penalty_losing_streak = (streak_frac - 0.15) * 20.0
            notes.append(f"Max losing streak is {max_losing_streak}/{n_trades} trades ({streak_frac:.0%}).")

    breakdown = PropFitnessBreakdown(
        pass_probability=pass_probability, payout_probability=payout_probability,
        robustness=robustness, oos_consistency=oos_consistency, drawdown_pct=drawdown_pct,
        base_score=base_score,
        penalty_too_few_trades=penalty_too_few_trades,
        penalty_param_sensitivity=penalty_param_sensitivity,
        penalty_high_pbo=penalty_high_pbo,
        penalty_is_oos_degradation=penalty_is_oos_degradation,
        penalty_concentration=penalty_concentration,
        penalty_losing_streak=penalty_losing_streak,
        notes=notes,
    )
    breakdown.final_score = base_score - breakdown.total_penalty()
    return breakdown
