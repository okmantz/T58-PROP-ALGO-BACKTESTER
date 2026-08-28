"""
Multi-strategy ensemble backtesting -- several DIFFERENT, weakly-correlated
strategies combined on the SAME instrument. The mirror case of
app.portfolio.portfolio, which already does multi-ASSET/single-strategy;
this module is multi-STRATEGY/single-asset.

Two combination modes:

  "blend" (run_ensemble_blend) -- each strategy leg keeps trading
    independently at its own correlation-adjusted share of one account's
    risk budget, exactly like a Portfolio leg does today, just pointed at
    the SAME df for every leg instead of a different instrument each.
    Reuses app.portfolio.portfolio.run_portfolio_backtest UNMODIFIED --
    its measure -> correlate -> re-weight -> chronologically-merge pipeline
    doesn't care whether what varies between legs is the instrument or the
    strategy, so there is no reason to reimplement it. More than one leg
    CAN have a position open at the same time (each leg is really trading
    its own slice of the account independently); this is the right model
    whenever every leg's own stop/target/trailing-stop rules should keep
    applying unchanged.

  "vote" (run_ensemble_vote) -- combines every leg's own raw signal series
    bar-by-bar into ONE composite long/flat/short signal (only entering
    once at least `min_agreement` legs agree on direction), then runs that
    single combined signal through the ordinary one-position-at-a-time
    engine, same as any other strategy. True to how a real single-account,
    single-position "voted" ensemble actually trades -- unlike blend mode,
    only one position is ever open at a time. Risk management (stop,
    target, trailing stop, break-even) is inherited from the FIRST-listed
    leg only, since voting combines entry TIMING across legs, not each
    leg's own exit rules -- reach for blend mode instead whenever each
    leg's own risk management should keep mattering.

Both modes require at least 2 strategy legs, and both are aimed at the
same goal this app's Portfolio feature already demonstrated for
instruments: several imperfectly-correlated return streams smooth the
combined equity curve (and therefore raise real eval-pass probability)
in a way no amount of re-tuning a single strategy's own parameters can.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.backtest.engine import BacktestResult, run_backtest
from app.backtest.risk import RiskConfig
from app.portfolio.portfolio import InstrumentLeg, PortfolioConfig, PortfolioResult, run_portfolio_backtest
from app.strategy.base import Strategy, StrategyResult


class EnsembleError(Exception):
    """Raised when an ensemble backtest cannot proceed."""


def _default_leg_name(strategy: Strategy, index: int) -> str:
    if strategy.source_type == "manual":
        name = getattr(strategy, "config", {}).get("name")
        if name:
            return str(name)
    return f"{strategy.source_type}_leg_{index + 1}"


def _validate_legs(strategies: list[Strategy], names: list[str] | None, weights: list[float] | None) -> tuple[list[str], list[float]]:
    if len(strategies) < 2:
        raise EnsembleError("An ensemble requires at least 2 strategy legs.")
    if names is not None and len(names) != len(strategies):
        raise EnsembleError("`names` must have exactly one entry per strategy leg.")
    if weights is not None and len(weights) != len(strategies):
        raise EnsembleError("`weights` must have exactly one entry per strategy leg.")
    resolved_names = list(names) if names else [_default_leg_name(s, i) for i, s in enumerate(strategies)]
    if len(set(resolved_names)) != len(resolved_names):
        raise EnsembleError(f"Ensemble leg names must be unique; got {resolved_names}.")
    resolved_weights = list(weights) if weights else [1.0] * len(strategies)
    return resolved_names, resolved_weights


# ---------------------------------------------------------------------------
# Blend mode
# ---------------------------------------------------------------------------

def build_ensemble_legs(
    df: pd.DataFrame,
    strategies: list[Strategy],
    risk: RiskConfig,
    names: list[str] | None = None,
    weights: list[float] | None = None,
) -> list[InstrumentLeg]:
    """Wraps N different strategies, all pointed at the SAME df, into the
    InstrumentLeg list app.portfolio.portfolio.run_portfolio_backtest
    expects -- the only difference from a real multi-asset Portfolio call
    is that every leg's `df` is identical."""
    resolved_names, resolved_weights = _validate_legs(strategies, names, weights)
    return [
        InstrumentLeg(name=name, df=df, strategy=strat, risk=risk, weight=weight)
        for name, strat, weight in zip(resolved_names, strategies, resolved_weights)
    ]


def run_ensemble_blend(
    df: pd.DataFrame,
    strategies: list[Strategy],
    risk: RiskConfig,
    names: list[str] | None = None,
    weights: list[float] | None = None,
    config: PortfolioConfig | None = None,
) -> PortfolioResult:
    """See module docstring. Returns the exact same PortfolioResult shape
    the Portfolio feature already reports (per-leg stats, correlation
    matrix, combined equity curve/statistics, diversification ratio) --
    reused as-is so the existing Portfolio report template can render an
    ensemble result unmodified, just relabeled "strategies" instead of
    "instruments" in the UI layer."""
    legs = build_ensemble_legs(df, strategies, risk, names, weights)
    return run_portfolio_backtest(legs, config)


# ---------------------------------------------------------------------------
# Vote mode
# ---------------------------------------------------------------------------

@dataclass
class EnsembleVoteConfig:
    min_agreement: int = 2   # how many legs must agree on direction before a combined entry fires

    def __post_init__(self):
        self.min_agreement = max(int(self.min_agreement), 1)


class _VoteEnsembleStrategy(Strategy):
    """Combines N sub-strategies' signals into one majority/threshold-vote
    signal. Not intended to be constructed directly outside this module --
    use run_ensemble_vote()."""

    source_type = "ensemble_vote"

    def __init__(self, strategies: list[Strategy], min_agreement: int, leg_names: list[str]):
        self.strategies = strategies
        self.min_agreement = min_agreement
        self.leg_names = leg_names

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        results = [s.generate(df) for s in self.strategies]
        signals = pd.concat([r.signals for r in results], axis=1)
        longs = (signals == 1).sum(axis=1)
        shorts = (signals == -1).sum(axis=1)
        combined = pd.Series(0, index=signals.index)
        combined[longs >= self.min_agreement] = 1
        combined[shorts >= self.min_agreement] = -1
        # If both directions somehow clear the threshold on the same bar
        # (possible when min_agreement is low relative to leg count and
        # legs split roughly evenly), that is not a consensus -- stay flat
        # rather than arbitrarily picking a side.
        conflict = (longs >= self.min_agreement) & (shorts >= self.min_agreement)
        combined[conflict] = 0

        primary = results[0]
        return StrategyResult(
            name=(
                f"Vote Ensemble ({self.min_agreement}-of-{len(self.strategies)} legs: "
                f"{', '.join(self.leg_names)}; risk management from '{self.leg_names[0]}')"
            ),
            source_type=self.source_type,
            signals=combined,
            stop_loss_pips=primary.stop_loss_pips,
            take_profit_pips=primary.take_profit_pips,
            stop_loss_distance=primary.stop_loss_distance,
            take_profit_distance=primary.take_profit_distance,
            trailing_stop_distance=primary.trailing_stop_distance,
            breakeven_trigger_r=primary.breakeven_trigger_r,
        )


def run_ensemble_vote(
    df: pd.DataFrame,
    strategies: list[Strategy],
    risk: RiskConfig,
    names: list[str] | None = None,
    vote_config: EnsembleVoteConfig | None = None,
) -> BacktestResult:
    """See module docstring. Returns an ordinary BacktestResult -- an
    ensemble in vote mode is, from the engine's point of view, just one
    more strategy with one signal series."""
    resolved_names, _ = _validate_legs(strategies, names, None)
    cfg = vote_config or EnsembleVoteConfig()
    if cfg.min_agreement > len(strategies):
        raise EnsembleError(
            f"min_agreement ({cfg.min_agreement}) cannot exceed the number of strategy legs "
            f"({len(strategies)})."
        )
    vote_strategy = _VoteEnsembleStrategy(strategies, cfg.min_agreement, resolved_names)
    return run_backtest(df, vote_strategy, risk)
