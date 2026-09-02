"""
T58 Strategy Scorecard.

Implements the T58 Quant Trading Masterclass material's Part XI scorecard
verbatim: a single 0-100 number built from EIGHT already-computed,
already-trusted signals this app produces elsewhere -- never a new
metric invented here, only a documented way of weighting and combining
what already exists:

    Metric                      Weight   Source
    Pass probability              25     app.monte_carlo.engine.MonteCarloResult.evaluation_pass_probability
    First payout probability      20     ...first_payout_probability
    Risk of ruin                 -20     ...risk_of_ruin_pct (subtracted)
    Walk-forward stability        15     app.search.robustness.WalkForwardResult.walk_forward_efficiency
    Monte Carlo robustness        10     see _mc_robustness_score below
    Parameter stability            5     app.validation.parameter_robustness.ParameterRobustnessResult
                                          (or the cheaper app.search.robustness.RobustnessResult, if
                                          that's the one already computed -- both are 0-100-like already)
    Expectancy                     5     app.backtest.statistics.BacktestStatistics.average_r
    Drawdown                      10     app.backtest.statistics.BacktestStatistics.max_drawdown_pct

Tiers, also verbatim from the Masterclass material:

    92+  Elite       85+  Strong       75+  Promising      65+  Research      <65  Reject

Three of the eight inputs (Monte Carlo robustness, parameter stability,
expectancy, drawdown) aren't already expressed on a 0-100 "higher is
better" scale in this app, so this module documents exactly how each is
mapped onto one -- see the docstring on each _*_score helper. Every
mapping is a heuristic; none of them invents a new backtest metric, they
only rescale ones that already exist. `score_from_results()` does that
mapping for you from the objects this app's own pipeline already
produces; `T58ScorecardInputs` is there for a caller that wants to
supply the eight 0-100 numbers itself (e.g. from records already
persisted in app.search.results_db, where a fresh Monte Carlo/
robustness object may not be sitting in memory anymore).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_WEIGHTS: dict[str, float] = {
    "pass_probability": 25.0,
    "first_payout_probability": 20.0,
    "risk_of_ruin": -20.0,
    "walk_forward_stability": 15.0,
    "monte_carlo_robustness": 10.0,
    "parameter_stability": 5.0,
    "expectancy": 5.0,
    "drawdown": 10.0,
}

_TIERS: list[tuple[float, str]] = [
    (92.0, "Elite"), (85.0, "Strong"), (75.0, "Promising"), (65.0, "Research"),
]


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class T58ScorecardInputs:
    """All eight components, already expressed 0-100 ('higher is always
    better', including risk_of_ruin and drawdown -- see score_from_results
    for how those two get inverted onto this scale before they arrive
    here). None for a component means 'not computed' -- the final score
    re-normalizes over only the weights actually present rather than
    silently treating a missing check as a zero, so a strategy that
    simply hasn't been walk-forward tested yet isn't penalized as if it
    had FAILED walk-forward testing."""
    pass_probability: float | None = None
    first_payout_probability: float | None = None
    risk_of_ruin: float | None = None
    walk_forward_stability: float | None = None
    monte_carlo_robustness: float | None = None
    parameter_stability: float | None = None
    expectancy: float | None = None
    drawdown: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class T58ScorecardResult:
    score: float                 # 0-100
    tier: str                    # "Elite" | "Strong" | "Promising" | "Research" | "Reject"
    components: dict             # component name -> {"value": float|None, "weight": float, "contribution": float|None}
    n_components_used: int
    n_components_total: int
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def render_line(self) -> str:
        used = f"{self.n_components_used}/{self.n_components_total} checks available"
        return f"T58 Score: {self.score:.1f}/100 -- {self.tier} ({used})"


def tier_for_score(score: float) -> str:
    for threshold, name in _TIERS:
        if score >= threshold:
            return name
    return "Reject"


def compute_t58_score(inputs: T58ScorecardInputs) -> T58ScorecardResult:
    """Weighted sum over whatever components are present, RE-NORMALIZED
    to the weight actually available (see T58ScorecardInputs docstring
    for why a missing check isn't scored as a failing one). All 8
    components present and maxed would score exactly 100; all 8 present
    and floored (0 everywhere, risk_of_ruin term also at its floor of 0)
    would score exactly 0. Positive and negative weights are handled
    the same way: a component's "value" is always 0-100 'higher is
    better' by this point (risk_of_ruin=0 means NO ruin risk, i.e. the
    best possible outcome), so `weight * value` is added for every
    component including the nominally-negative-weighted risk_of_ruin --
    the sign only matters for how score_from_results() maps the RAW
    risk-of-ruin percentage onto this inverted scale."""
    values = inputs.to_dict()
    components: dict = {}
    total_weight_available = 0.0
    weighted_sum = 0.0
    notes: list[str] = []

    for name, weight in _WEIGHTS.items():
        val = values.get(name)
        abs_weight = abs(weight)
        if val is None:
            components[name] = {"value": None, "weight": weight, "contribution": None}
            continue
        val_clipped = _clip(float(val))
        contribution = abs_weight * val_clipped
        components[name] = {"value": val_clipped, "weight": weight, "contribution": contribution}
        weighted_sum += contribution
        total_weight_available += abs_weight * 100.0

    n_total = len(_WEIGHTS)
    n_used = sum(1 for c in components.values() if c["value"] is not None)
    if total_weight_available <= 0:
        notes.append("No scorecard components were available -- score is 0 by default, not a real evaluation.")
        return T58ScorecardResult(score=0.0, tier="Reject", components=components,
                                   n_components_used=0, n_components_total=n_total, notes=notes)

    score = _clip(weighted_sum / total_weight_available * 100.0)
    if n_used < n_total:
        missing = [name for name, c in components.items() if c["value"] is None]
        notes.append(
            f"{n_total - n_used} of {n_total} checks weren't available ({', '.join(missing)}) -- "
            "score is re-normalized over what WAS checked, not penalized for what wasn't."
        )
    return T58ScorecardResult(
        score=score, tier=tier_for_score(score), components=components,
        n_components_used=n_used, n_components_total=n_total, notes=notes,
    )


# ---------------------------------------------------------------------------
# Mapping real pipeline objects onto the 0-100 'higher is better' scale
# ---------------------------------------------------------------------------

def _walk_forward_score(walk_forward_efficiency: float) -> float:
    """walk_forward_efficiency is mean_test_metric/mean_train_metric,
    already clipped to [-5, 5] by app.search.robustness -- 1.0 means
    the test period did exactly as well as train (ideal), > 1 is even
    better (rare), < 1 means some degradation out of sample. Maps
    linearly: 1.0+ -> 100, 0.0 (test performance vanished) -> 0,
    negative (test period LOST when train WON) -> 0 floor."""
    return _clip(walk_forward_efficiency * 100.0)


def _mc_robustness_score(mc_result) -> float | None:
    """'Monte Carlo robustness' isn't a metric the Masterclass material
    defines precisely beyond 'calculate distributions, not just expected
    profit' -- interpreted here as how TIGHT the simulated return
    distribution is: a strategy whose 25th-75th percentile return spread
    is small relative to its median return is more robust (every
    simulated path tells a similar story) than one where the outcome
    swings wildly simulation to simulation, even if the median is the
    same. Returns None (not 0) if the percentiles this needs weren't
    computed, so a missing input doesn't get scored as maximally
    UNROBUST."""
    pcts = getattr(mc_result, "return_percentiles", None) or {}
    p25, p50, p75 = pcts.get(25), pcts.get(50), pcts.get(75)
    if p25 is None or p50 is None or p75 is None or p50 == 0:
        return None
    spread_ratio = abs(p75 - p25) / abs(p50)
    # A spread as wide as the median itself (ratio 1.0) scores 0; no
    # spread at all (ratio 0.0) scores 100 -- linear between.
    return _clip(100.0 * (1.0 - min(spread_ratio, 1.0)))


def _expectancy_score(average_r: float | None) -> float | None:
    """average_r is the strategy's mean R-multiple per trade (see
    app.backtest.statistics). 0R or worse scores 0; 2R or better scores
    100 (the Masterclass material's own Lesson 1 example strategy,
    +0.44R expected, would score 22/100 here -- a real, working edge
    can still be a LOW number on this component alone, which is why
    it's only weighted 5/90ths of the total score)."""
    if average_r is None:
        return None
    return _clip(average_r / 2.0 * 100.0)


def _drawdown_score(max_drawdown_pct: float | None, prop_max_drawdown_pct: float | None) -> float | None:
    """Inverted and scaled against the account's OWN max-drawdown rule
    (PropRules.max_drawdown_pct) rather than an arbitrary fixed number,
    since 'how much drawdown is acceptable' is defined by the firm's own
    rules, not a universal constant: using exactly the full allowance
    scores 0 (no safety margin left), using none of it scores 100."""
    if max_drawdown_pct is None or not prop_max_drawdown_pct:
        return None
    return _clip(100.0 * (1.0 - abs(max_drawdown_pct) / abs(prop_max_drawdown_pct)))


def _risk_of_ruin_score(risk_of_ruin_pct: float | None) -> float | None:
    """Simple inversion onto the 'higher is better' scale every other
    component uses: 0% risk of ruin -> 100, 100% risk of ruin -> 0."""
    if risk_of_ruin_pct is None:
        return None
    return _clip(100.0 - risk_of_ruin_pct)


def score_from_results(
    mc_result=None,
    walk_forward_result=None,
    robustness_result=None,
    statistics=None,
    prop_max_drawdown_pct: float | None = None,
) -> T58ScorecardResult:
    """Convenience entry point: pass whichever of this app's own result
    objects you already have in hand (any/all may be None -- a partial
    scorecard, correctly re-normalized, beats refusing to score at
    all). `robustness_result` accepts either
    app.validation.parameter_robustness.ParameterRobustnessResult
    (reads .parameter_robustness_score directly) or
    app.search.robustness.RobustnessResult (reads .stability_ratio,
    scaled x100 and clipped -- it's a ratio centered near 1.0, not
    already a 0-100 score)."""
    pass_probability = getattr(mc_result, "evaluation_pass_probability", None)
    first_payout_probability = getattr(mc_result, "first_payout_probability", None)
    risk_of_ruin_pct = getattr(mc_result, "risk_of_ruin_pct", None)

    walk_forward_stability = None
    if walk_forward_result is not None:
        walk_forward_stability = _walk_forward_score(walk_forward_result.walk_forward_efficiency)

    parameter_stability = None
    if robustness_result is not None:
        if hasattr(robustness_result, "parameter_robustness_score"):
            parameter_stability = _clip(robustness_result.parameter_robustness_score)
        elif hasattr(robustness_result, "stability_ratio"):
            parameter_stability = _clip(robustness_result.stability_ratio * 100.0)

    average_r = getattr(statistics, "average_r", None) if statistics is not None else None
    max_drawdown_pct = getattr(statistics, "max_drawdown_pct", None) if statistics is not None else None

    inputs = T58ScorecardInputs(
        pass_probability=pass_probability,
        first_payout_probability=first_payout_probability,
        risk_of_ruin=_risk_of_ruin_score(risk_of_ruin_pct),
        walk_forward_stability=walk_forward_stability,
        monte_carlo_robustness=_mc_robustness_score(mc_result) if mc_result is not None else None,
        parameter_stability=parameter_stability,
        expectancy=_expectancy_score(average_r),
        drawdown=_drawdown_score(max_drawdown_pct, prop_max_drawdown_pct),
    )
    return compute_t58_score(inputs)
