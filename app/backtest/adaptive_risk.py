"""
Declarative, config-driven adaptive risk rules.

No strategy source this app supports (Manual, Python, PineScript, MQL5)
can see its own realized trade outcomes at signal-generation time --
generate_signals() / ManualStrategy.generate() run once, statelessly, over
the whole price series before any P&L exists. Money-management behavior
that depends on the account's own trading history -- "cut size after 2
losses in a row", "coast once 80% of the way to the profit target",
"de-risk once today's realized P&L is down X%" -- is the same category of
gap that RiskConfig.daily_loss_limit_pct was added to close for a hard
circuit breaker: it cannot be expressed inside a strategy's own signal
logic, no matter which source type, so it has to be a first-class ENGINE
feature instead. This module makes that possible for graduated,
size-scaling money management (not just a binary stop-trading breaker),
so any strategy can be backtested with or without a given adaptive-risk
overlay applied on top, without changing a single line of the strategy
itself.

Rules are declarative and stack MULTIPLICATIVELY: every rule whose
condition is currently true contributes a position-size multiplier for
the NEXT entry, and the multiplier actually applied is the PRODUCT of
every active rule's multiplier. This keeps each rule simple and
independent -- no rule needs to know about any other rule -- while still
letting several de-risking conditions compound, the way a real trader
stacking "I'm down today" with "I've lost twice in a row" would actually
behave more cautiously than either alone.

Multipliers only ever affect the SIZE of new entries. They never touch
stop-loss/take-profit placement, and they are evaluated only at the
moment a new position is opened (this engine trades one position at a
time -- see app.backtest.execution's module docstring -- so there is
never a "resize an open position" case to handle).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_VALID_TRIGGERS = {"consecutive_losses", "daily_loss_pct", "daily_profit_pct", "progress_to_target_pct"}


class AdaptiveRiskError(Exception):
    """Raised for an invalid adaptive-risk rule or config."""


@dataclass
class AdaptiveRiskRule:
    trigger: str
    # "consecutive_losses"     -- threshold = number of losing trades in a row (resets to 0 on any winning trade)
    # "daily_loss_pct"         -- threshold = % of initial_balance realized-lost so far TODAY
    # "daily_profit_pct"       -- threshold = % of initial_balance realized-gained so far TODAY
    # "progress_to_target_pct" -- threshold = % of the way from 0 to profit_target_amount the account has reached
    #                             so far, ALL-TIME (not reset daily) -- requires
    #                             AdaptiveRiskConfig.profit_target_amount to be set; a rule using this trigger is
    #                             simply never active (multiplier 1.0) if that amount isn't supplied.
    threshold: float
    risk_multiplier: float   # position-size multiplier for new entries while this rule is active (e.g. 0.5 = half size)
    label: str = ""          # optional human-readable name, surfaced in reports/UI only

    def __post_init__(self):
        if self.trigger not in _VALID_TRIGGERS:
            raise AdaptiveRiskError(
                f"Unknown adaptive risk trigger '{self.trigger}'. Supported: {sorted(_VALID_TRIGGERS)}"
            )
        self.threshold = float(self.threshold)
        self.risk_multiplier = max(float(self.risk_multiplier), 0.0)
        if not self.label:
            self.label = f"{self.trigger} >= {self.threshold} -> x{self.risk_multiplier}"


@dataclass
class AdaptiveRiskConfig:
    enabled: bool = False
    rules: list = field(default_factory=list)   # list[AdaptiveRiskRule]
    profit_target_amount: float | None = None   # required only by "progress_to_target_pct" rules; in account
    # currency (e.g. initial_balance * evaluation_profit_target_pct / 100 -- the caller computes this from
    # whatever PropRules/target the backtest is being run against, since this module has no PropRules dependency).

    def __post_init__(self):
        if self.rules and not all(isinstance(r, AdaptiveRiskRule) for r in self.rules):
            self.rules = [r if isinstance(r, AdaptiveRiskRule) else AdaptiveRiskRule(**r) for r in self.rules]


@dataclass
class AdaptiveRiskState:
    """Mutable per-backtest bookkeeping the execution loop updates as
    trades close. Deliberately never exposed to a Strategy -- only
    app.backtest.execution.run_execution reads/writes this, which is the
    entire point: the strategy's own signal logic still can't see it."""
    initial_balance: float = 0.0
    consecutive_losses: int = 0
    day_realized_pnl: float = 0.0
    cumulative_realized_pnl: float = 0.0

    def record_trade_close(self, pnl: float, is_new_day: bool) -> None:
        if is_new_day:
            self.day_realized_pnl = 0.0
        self.day_realized_pnl += pnl
        self.cumulative_realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0
        # a scratch trade (pnl == 0) leaves the streak counter unchanged

    def _rule_active(self, rule: AdaptiveRiskRule, profit_target_amount: float | None) -> bool:
        if rule.trigger == "consecutive_losses":
            return self.consecutive_losses >= rule.threshold
        if rule.trigger == "daily_loss_pct":
            loss_pct = (-self.day_realized_pnl / self.initial_balance * 100.0) if self.initial_balance else 0.0
            return loss_pct >= rule.threshold
        if rule.trigger == "daily_profit_pct":
            profit_pct = (self.day_realized_pnl / self.initial_balance * 100.0) if self.initial_balance else 0.0
            return profit_pct >= rule.threshold
        if rule.trigger == "progress_to_target_pct":
            if not profit_target_amount:
                return False
            progress_pct = self.cumulative_realized_pnl / profit_target_amount * 100.0
            return progress_pct >= rule.threshold
        return False

    def active_multiplier(self, config: AdaptiveRiskConfig) -> float:
        """The product of every currently-triggered rule's multiplier --
        1.0 (no de-risking) if this config is disabled or no rule fires."""
        if not config.enabled or not config.rules:
            return 1.0
        mult = 1.0
        for rule in config.rules:
            if self._rule_active(rule, config.profit_target_amount):
                mult *= rule.risk_multiplier
        return mult

    def active_rule_labels(self, config: AdaptiveRiskConfig) -> list[str]:
        """Which rules are currently firing -- used to annotate a trade
        with WHY its size was scaled, for report transparency."""
        if not config.enabled:
            return []
        return [rule.label for rule in config.rules if self._rule_active(rule, config.profit_target_amount)]
