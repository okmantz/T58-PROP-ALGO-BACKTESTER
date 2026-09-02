"""
Regime Router -- Multi-Regime Strategy (T58 Quant Trading Masterclass PDF,
Strategy Research #7: "Instead of ONE STRATEGY, build a regime classifier
that determines which strategy is allowed to trade").

This is what turns "regime_switching" in app.strategy.family_taxonomy from
a taxonomy label into an actual, runnable Strategy: a mapping of
{regime_label: Strategy} plus the classification dimension to route on
(any one of app.validation.regime_matrix's four -- trend/volatility/
session/environment) produces ONE combined signal series, where each
bar's signal comes from whichever sub-strategy is assigned to the regime
active at that bar. A bar in a regime with no assigned sub-strategy is
forced FLAT (0), never defaulted to some other sub-strategy's opinion --
an unassigned regime means "no strategy has been vetted for this
condition," a strictly stronger statement than "trade the fallback
strategy anyway, hope for the best."

Every sub-strategy's generate() is run over the FULL dataset (not just
its regime's bars), so its own rolling indicators warm up correctly --
only the resulting SIGNAL is masked down to its regime's bars afterward.
This is the same "generate on the whole history, then select" pattern
app.validation.regime_matrix itself uses for trade attribution, and for
the same reason: an indicator computed only on a regime's own
disconnected bars would be warming up from scratch at the start of every
segment, which isn't how the strategy would actually see the data live.

Risk management: each sub-strategy's own stop/target settings are
respected inside its own regime (a trend sub-strategy's ATR multiple and
a mean-reversion sub-strategy's ATR multiple don't have to match) by
combining their PER-BAR stop/target distance series the same way signals
are combined. A sub-strategy that only set a scalar stop_loss_pips/
take_profit_pips (no distance series -- see app.strategy.base.
StrategyResult) is converted to an equivalent constant-value distance
series using `pip_size`, so every regime's risk settings can be merged
bar-by-bar regardless of which style each sub-strategy used. `pip_size`
is the one thing the router needs its caller to supply that a plain
Strategy.generate(df) call doesn't otherwise need -- pass the SAME
pip_size you're about to run the backtest's RiskConfig with.
"""
from __future__ import annotations

import pandas as pd

from app.strategy.base import Strategy, StrategyError, StrategyResult
from app.validation.regime_matrix import RegimeThresholds, label_regimes

_VALID_DIMENSIONS = {"trend", "volatility", "session", "environment"}


class RegimeRouterStrategy(Strategy):
    source_type = "regime_router"

    def __init__(
        self,
        regime_dimension: str,
        strategies_by_regime: dict[str, Strategy],
        pip_size: float = 0.0001,
        thresholds: RegimeThresholds | None = None,
        name: str | None = None,
    ):
        """
        regime_dimension: which of app.validation.regime_matrix's four
            classification dimensions to route on.
        strategies_by_regime: {regime_label: Strategy}. Valid labels
            depend on `regime_dimension` -- see
            app.validation.regime_matrix._TREND_NAMES / _VOL_NAMES /
            _SESSION_WINDOWS / _ENV_NAMES for the exact label sets. A
            label not present in the classifier's output for this
            dataset is simply never active (not an error) -- e.g. a
            "power_hour" entry on data with no bars in that window.
        pip_size: used only to convert a sub-strategy's scalar
            stop_loss_pips/take_profit_pips into a per-bar distance
            series when it didn't already produce one -- pass the same
            value you're about to run the backtest's RiskConfig with.
        thresholds: reuse previously-fit RegimeThresholds (e.g. from a
            prior app.validation.regime_matrix run on this same
            instrument) instead of re-fitting fresh quantiles on
            whatever df this router happens to see -- important for a
            forward/live run, where fitting fresh cut points on a short
            recent window would silently redefine what "extreme
            volatility" even means each time it runs.
        """
        if regime_dimension not in _VALID_DIMENSIONS:
            raise StrategyError(
                f"Unknown regime dimension '{regime_dimension}'. Must be one of {sorted(_VALID_DIMENSIONS)}."
            )
        if not strategies_by_regime:
            raise StrategyError("RegimeRouterStrategy needs at least one {regime_label: Strategy} mapping.")
        self.regime_dimension = regime_dimension
        self.strategies_by_regime = strategies_by_regime
        self.pip_size = pip_size
        self.thresholds = thresholds
        self.name = name or f"Regime Router ({regime_dimension}: {', '.join(strategies_by_regime)})"

    def generate(self, df: pd.DataFrame) -> StrategyResult:
        labels, fit_thresholds = label_regimes(df, thresholds=self.thresholds)
        regime_series = labels[self.regime_dimension]

        combined_signal = pd.Series(0.0, index=df.index)
        stop_distance = pd.Series(float("nan"), index=df.index)
        target_distance = pd.Series(float("nan"), index=df.index)
        any_distance_used = False

        for regime_label, sub_strategy in self.strategies_by_regime.items():
            result = sub_strategy.generate(df)
            sub_signals = self._validate_signals(result.signals, df)
            mask = (regime_series == regime_label).fillna(False)
            if not mask.any():
                continue  # this regime never occurred in this data -- nothing to route

            combined_signal = combined_signal.where(~mask, sub_signals.astype(float))

            sub_stop = self._as_distance_series(result.stop_loss_distance, result.stop_loss_pips, df)
            if sub_stop is not None:
                stop_distance = stop_distance.where(~mask, sub_stop)
                any_distance_used = True
            sub_target = self._as_distance_series(result.take_profit_distance, result.take_profit_pips, df)
            if sub_target is not None:
                target_distance = target_distance.where(~mask, sub_target)
                any_distance_used = True

        return StrategyResult(
            name=self.name, source_type=self.source_type, signals=combined_signal,
            stop_loss_distance=stop_distance if any_distance_used else None,
            take_profit_distance=target_distance if any_distance_used else None,
        )

    def _as_distance_series(self, distance: pd.Series | None, pips: float | None, df: pd.DataFrame) -> pd.Series | None:
        if distance is not None:
            return distance
        if pips is not None:
            return pd.Series(float(pips) * self.pip_size, index=df.index)
        return None

    def active_regimes(self) -> list[str]:
        return list(self.strategies_by_regime.keys())
